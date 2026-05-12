use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chrono::Utc;
use clap::{Args, ValueEnum};
use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::EdwardsPoint;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::Serialize;

#[cfg(feature = "gpu")]
use crate::search::GpuSearcher;
use crate::search::{advance_scalar, clamp_scalar, PrefixMatcher};

const SCHEMA_VERSION: u32 = 1;

/// Number of points compressed under one batched inversion. Mirrors
/// `CHAIN_BATCH` in src/search.rs so the bench worker measures the same hot
/// loop the real search uses.
const CHAIN_BATCH: usize = 16;

/// Per-worker flush threshold for atomic counter updates. Keeping it batched
/// avoids dominating the loop with cache-line contention at >1 MH/s.
const FLUSH_EVERY: u64 = 1024;

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Seconds to run each mode.
    #[arg(short, long, default_value_t = 10)]
    pub duration: u64,

    /// Which mode(s) to run.
    #[arg(short, long, value_enum, default_value = "all")]
    pub mode: BenchMode,

    /// CPU threads for cpu-n / hybrid (default: available_parallelism).
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Hex prefix(es) to count matches against. Default "abc" yields enough
    /// matches in a 10s run at any rate for the Poisson sanity check.
    #[arg(short, long, default_value = "abc")]
    pub prefix: Vec<String>,

    /// Append one JSONL record per mode to this file. If omitted, records are
    /// printed to stdout instead.
    #[arg(short, long)]
    pub out: Option<PathBuf>,

    /// Human-friendly machine label embedded in each record (e.g. "M4 Max MBP").
    #[arg(long)]
    pub machine_label: Option<String>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchMode {
    Cpu1,
    CpuN,
    Gpu,
    Hybrid,
    All,
}

#[derive(Serialize)]
struct BenchRecord {
    schema_version: u32,
    timestamp: String,
    git: BenchGit,
    machine: BenchMachine,
    config: BenchConfig,
    results: BenchResults,
}

#[derive(Serialize, Clone)]
struct BenchGit {
    sha: Option<String>,
    dirty: Option<bool>,
    branch: Option<String>,
}

#[derive(Serialize, Clone)]
struct BenchMachine {
    label: Option<String>,
    hostname: String,
    os: String,
    cpu_model: String,
    cpu_physical_cores: Option<usize>,
    cpu_logical_cores: usize,
    memory_gb: u64,
    gpus: Vec<String>,
}

#[derive(Serialize)]
struct BenchConfig {
    mode: &'static str,
    threads: usize,
    gpus_used: usize,
    prefixes: Vec<String>,
    duration_secs: u64,
    build_features: Vec<&'static str>,
}

#[derive(Serialize)]
struct BenchResults {
    elapsed_secs: f64,
    total_keys: u64,
    rate_keys_per_sec: f64,
    matches: u64,
    expected: f64,
    ratio: f64,
    poisson_sigma_pct: f64,
}

struct BenchTotals {
    keys: u64,
    matches: u64,
    elapsed: f64,
}

/// Handle for a running benchmark workload. Mirrors `SearchHandle` but counts
/// matches without exiting, so a single run can validate observed-vs-expected.
struct BenchHandle {
    stop: Arc<AtomicBool>,
    keys: Arc<AtomicU64>,
    matches: Arc<AtomicU64>,
    workers: Vec<JoinHandle<()>>,
    start: Instant,
}

impl BenchHandle {
    fn empty() -> Self {
        BenchHandle {
            stop: Arc::new(AtomicBool::new(false)),
            keys: Arc::new(AtomicU64::new(0)),
            matches: Arc::new(AtomicU64::new(0)),
            workers: Vec::new(),
            start: Instant::now(),
        }
    }

    fn spawn_cpu_workers(&mut self, matchers: &Arc<Vec<PrefixMatcher>>, count: usize) {
        self.workers.reserve(count);
        for _ in 0..count {
            self.workers.push(spawn_cpu_bench_worker(
                Arc::clone(matchers),
                Arc::clone(&self.stop),
                Arc::clone(&self.keys),
                Arc::clone(&self.matches),
            ));
        }
    }

    #[cfg(feature = "gpu")]
    fn spawn_gpu_workers(&mut self, gpus: Vec<Box<dyn GpuSearcher>>) {
        self.workers.reserve(gpus.len());
        for searcher in gpus {
            self.workers.push(spawn_gpu_bench_worker(
                searcher,
                Arc::clone(&self.stop),
                Arc::clone(&self.keys),
                Arc::clone(&self.matches),
            ));
        }
    }

    fn start_cpu(matchers: Arc<Vec<PrefixMatcher>>, threads: usize) -> Self {
        let mut h = Self::empty();
        h.spawn_cpu_workers(&matchers, threads);
        h
    }

    #[cfg(feature = "gpu")]
    fn start_gpu(gpus: Vec<Box<dyn GpuSearcher>>) -> Self {
        let mut h = Self::empty();
        h.spawn_gpu_workers(gpus);
        h
    }

    #[cfg(feature = "gpu")]
    fn start_hybrid(
        matchers: Arc<Vec<PrefixMatcher>>,
        threads: usize,
        gpus: Vec<Box<dyn GpuSearcher>>,
    ) -> Self {
        let mut h = Self::empty();
        h.spawn_cpu_workers(&matchers, threads);
        h.spawn_gpu_workers(gpus);
        h
    }

    fn stop(self) -> BenchTotals {
        self.stop.store(true, Ordering::Relaxed);
        for w in self.workers {
            let _ = w.join();
        }
        BenchTotals {
            keys: self.keys.load(Ordering::Relaxed),
            matches: self.matches.load(Ordering::Relaxed),
            elapsed: self.start.elapsed().as_secs_f64(),
        }
    }
}

/// CPU bench worker: mirrors `spawn_cpu_worker` in src/search.rs but counts
/// matches and continues until `stop` is signaled, instead of returning on
/// first match. The `+8B` chained-compress hot loop is byte-identical, so
/// we measure the same code path real searches use.
fn spawn_cpu_bench_worker(
    matchers: Arc<Vec<PrefixMatcher>>,
    stop: Arc<AtomicBool>,
    keys: Arc<AtomicU64>,
    matches: Arc<AtomicU64>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let eight_b = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);

        let mut scalar = [0u8; 32];
        OsRng.fill_bytes(&mut scalar);
        clamp_scalar(&mut scalar);
        let mut point: EdwardsPoint = EdwardsPoint::mul_base_clamped(scalar);

        let mut local_keys: u64 = 0;
        let mut local_matches: u64 = 0;

        while !stop.load(Ordering::Relaxed) {
            let mut next_point = point;
            let batch_points: [EdwardsPoint; CHAIN_BATCH] = core::array::from_fn(|_| {
                let cur = next_point;
                next_point += eight_b;
                cur
            });

            let compressed = EdwardsPoint::compress_batch::<CHAIN_BATCH>(&batch_points);

            for c in compressed.iter() {
                let public_key = c.to_bytes();
                if public_key[0] == 0x00 || public_key[0] == 0xFF {
                    continue;
                }
                if matchers.iter().any(|m| m.matches(&public_key)) {
                    local_matches += 1;
                }
            }

            local_keys += CHAIN_BATCH as u64;
            if local_keys >= FLUSH_EVERY {
                keys.fetch_add(local_keys, Ordering::Relaxed);
                if local_matches > 0 {
                    matches.fetch_add(local_matches, Ordering::Relaxed);
                    local_matches = 0;
                }
                local_keys = 0;
            }
            point = next_point;
            advance_scalar(&mut scalar, 8 * CHAIN_BATCH as u64);
        }

        keys.fetch_add(local_keys, Ordering::Relaxed);
        if local_matches > 0 {
            matches.fetch_add(local_matches, Ordering::Relaxed);
        }
    })
}

#[cfg(feature = "gpu")]
fn spawn_gpu_bench_worker(
    mut searcher: Box<dyn GpuSearcher>,
    stop: Arc<AtomicBool>,
    keys: Arc<AtomicU64>,
    matches: Arc<AtomicU64>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            match searcher.count_batch() {
                Ok((k, m)) => {
                    keys.fetch_add(k, Ordering::Relaxed);
                    matches.fetch_add(m as u64, Ordering::Relaxed);
                }
                Err(e) => {
                    eprintln!(
                        "GPU error during bench ({}): {}",
                        searcher.device_name(),
                        e
                    );
                    return;
                }
            }
        }
    })
}

fn expand_modes(mode: BenchMode) -> Vec<BenchMode> {
    match mode {
        BenchMode::All => vec![
            BenchMode::Cpu1,
            BenchMode::CpuN,
            BenchMode::Gpu,
            BenchMode::Hybrid,
        ],
        m => vec![m],
    }
}

fn mode_str(mode: BenchMode) -> &'static str {
    match mode {
        BenchMode::Cpu1 => "cpu1",
        BenchMode::CpuN => "cpuN",
        BenchMode::Gpu => "gpu",
        BenchMode::Hybrid => "hybrid",
        BenchMode::All => "all",
    }
}

fn build_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut f = Vec::new();
    #[cfg(feature = "cuda")]
    f.push("cuda");
    #[cfg(feature = "metal")]
    f.push("metal");
    f
}

fn vergen_env(v: Option<&'static str>) -> Option<String> {
    v.filter(|s| !s.is_empty() && *s != "VERGEN_IDEMPOTENT_OUTPUT")
        .map(|s| s.to_string())
}

fn gather_git() -> BenchGit {
    BenchGit {
        sha: vergen_env(option_env!("VERGEN_GIT_SHA")),
        dirty: vergen_env(option_env!("VERGEN_GIT_DIRTY")).map(|v| v == "true"),
        branch: vergen_env(option_env!("VERGEN_GIT_BRANCH")),
    }
}

fn gather_machine(label: Option<String>) -> BenchMachine {
    let sys = sysinfo::System::new_all();
    let hostname = sysinfo::System::host_name().unwrap_or_else(|| "unknown".to_string());
    let os = sysinfo::System::long_os_version()
        .or_else(sysinfo::System::name)
        .unwrap_or_else(|| "unknown".to_string());
    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let cpu_logical_cores = sys.cpus().len();
    let cpu_physical_cores = sys.physical_core_count();
    let memory_gb = sys.total_memory() / (1024 * 1024 * 1024);

    BenchMachine {
        label,
        hostname,
        os,
        cpu_model,
        cpu_physical_cores,
        cpu_logical_cores,
        memory_gb,
        gpus: Vec::new(), // populated per-mode from searcher.device_name()
    }
}

pub fn run(args: BenchArgs) -> Result<(), Box<dyn Error>> {
    // Validate prefixes via the same rules the vanity search uses.
    let prefixes: Vec<String> = args
        .prefix
        .iter()
        .map(|p| crate::validate_prefix(p))
        .collect::<Result<_, _>>()
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    let matchers: Arc<Vec<PrefixMatcher>> = Arc::new(
        prefixes
            .iter()
            .map(|p| PrefixMatcher::new(p))
            .collect(),
    );

    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let machine_template = gather_machine(args.machine_label.clone());
    let git = gather_git();
    let duration = Duration::from_secs(args.duration);

    eprintln!(
        "Machine: {} ({}, {} logical / {:?} physical cores, {} GB RAM)",
        machine_template
            .label
            .as_deref()
            .unwrap_or(machine_template.hostname.as_str()),
        machine_template.cpu_model,
        machine_template.cpu_logical_cores,
        machine_template.cpu_physical_cores,
        machine_template.memory_gb,
    );
    if let Some(sha) = git.sha.as_deref() {
        let dirty_tag = if git.dirty == Some(true) { " (dirty)" } else { "" };
        eprintln!(
            "Commit:  {}{} on {}",
            &sha[..sha.len().min(12)],
            dirty_tag,
            git.branch.as_deref().unwrap_or("?")
        );
    }

    let ctx = BenchContext {
        threads,
        prefixes: &prefixes,
        matchers,
        duration,
        git: &git,
        machine_template: &machine_template,
        out: args.out.as_ref(),
    };

    for mode in expand_modes(args.mode) {
        if let Some(reason) = mode_skip_reason(mode) {
            eprintln!("Skipping {}: {}", mode_str(mode), reason);
            continue;
        }
        run_one_mode(&ctx, mode)?;
    }

    Ok(())
}

/// Per-run inputs shared across every mode in a single `bench` invocation.
struct BenchContext<'a> {
    threads: usize,
    prefixes: &'a [String],
    matchers: Arc<Vec<PrefixMatcher>>,
    duration: Duration,
    git: &'a BenchGit,
    machine_template: &'a BenchMachine,
    out: Option<&'a PathBuf>,
}

/// Returns Some(reason) if the mode can't run on this build / hardware.
fn mode_skip_reason(mode: BenchMode) -> Option<&'static str> {
    match mode {
        BenchMode::Gpu | BenchMode::Hybrid => {
            #[cfg(feature = "gpu")]
            {
                None
            }
            #[cfg(not(feature = "gpu"))]
            {
                Some("built without gpu feature")
            }
        }
        _ => None,
    }
}

fn run_one_mode(ctx: &BenchContext, mode: BenchMode) -> Result<(), Box<dyn Error>> {
    eprintln!(
        "Running bench: mode={} duration={}s prefixes={:?}",
        mode_str(mode),
        ctx.duration.as_secs(),
        ctx.prefixes
    );

    let (handle, threads_used, gpu_names) = match mode {
        BenchMode::Cpu1 => (
            BenchHandle::start_cpu(Arc::clone(&ctx.matchers), 1),
            1,
            Vec::<String>::new(),
        ),
        BenchMode::CpuN => (
            BenchHandle::start_cpu(Arc::clone(&ctx.matchers), ctx.threads),
            ctx.threads,
            Vec::<String>::new(),
        ),
        BenchMode::Gpu | BenchMode::Hybrid => {
            #[cfg(feature = "gpu")]
            {
                let gpus = crate::try_init_gpu(ctx.prefixes);
                if gpus.is_empty() {
                    eprintln!("  No GPU available, skipping.");
                    return Ok(());
                }
                let names: Vec<String> =
                    gpus.iter().map(|g| g.device_name().to_string()).collect();
                if mode == BenchMode::Gpu {
                    (BenchHandle::start_gpu(gpus), 0, names)
                } else {
                    (
                        BenchHandle::start_hybrid(Arc::clone(&ctx.matchers), ctx.threads, gpus),
                        ctx.threads,
                        names,
                    )
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                unreachable!("filtered by mode_skip_reason")
            }
        }
        BenchMode::All => unreachable!("expanded by expand_modes"),
    };

    thread::sleep(ctx.duration);
    let totals = handle.stop();

    let rate = if totals.elapsed > 0.0 {
        totals.keys as f64 / totals.elapsed
    } else {
        0.0
    };
    let min_len = ctx.prefixes.iter().map(|p| p.len()).min().unwrap_or(1);
    let same_len_count = ctx.prefixes.iter().filter(|p| p.len() == min_len).count() as f64;
    let p_match = same_len_count / 16f64.powi(min_len as i32);
    let expected = totals.keys as f64 * p_match;
    let ratio = if expected > 0.0 {
        totals.matches as f64 / expected
    } else {
        0.0
    };
    let poisson_sigma_pct = if totals.matches > 0 {
        100.0 / (totals.matches as f64).sqrt()
    } else {
        f64::INFINITY
    };

    let gpus_used = gpu_names.len();
    let machine = BenchMachine {
        gpus: gpu_names,
        ..ctx.machine_template.clone()
    };

    let record = BenchRecord {
        schema_version: SCHEMA_VERSION,
        timestamp: Utc::now().to_rfc3339(),
        git: ctx.git.clone(),
        machine,
        config: BenchConfig {
            mode: mode_str(mode),
            threads: threads_used,
            gpus_used,
            prefixes: ctx.prefixes.to_vec(),
            duration_secs: ctx.duration.as_secs(),
            build_features: build_features(),
        },
        results: BenchResults {
            elapsed_secs: totals.elapsed,
            total_keys: totals.keys,
            rate_keys_per_sec: rate,
            matches: totals.matches,
            expected,
            ratio,
            poisson_sigma_pct,
        },
    };

    let rate_label = if rate >= 1e9 {
        format!("{:.2} GH/s", rate / 1e9)
    } else if rate >= 1e6 {
        format!("{:.2} MH/s", rate / 1e6)
    } else {
        format!("{:.2} kH/s", rate / 1e3)
    };
    eprintln!(
        "  -> {:.2}s, {} keys, {}, {} matches (expected {:.0}, ratio {:.3} ±{:.1}%)",
        totals.elapsed,
        totals.keys,
        rate_label,
        totals.matches,
        expected,
        ratio,
        poisson_sigma_pct,
    );

    let json = serde_json::to_string(&record)?;
    if let Some(path) = ctx.out {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", json)?;
    } else {
        println!("{}", json);
    }

    Ok(())
}
