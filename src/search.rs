use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use rand::rngs::OsRng;
use rand::RngCore;

use crate::keygen::generate_keypair;
use crate::types::{MeshCoreKeypair, SearchResult, SearchStats};

const BATCH_SIZE: u64 = 1024;

/// Parsed prefix for fast nibble-level matching.
/// Avoids hex-encoding every public key in the hot loop.
pub struct PrefixMatcher {
    /// Full bytes to match (pairs of hex chars).
    pub(crate) full_bytes: Vec<u8>,
    /// If prefix has odd length, the high nibble of the trailing hex char.
    pub(crate) trailing_nibble: Option<u8>,
}

impl PrefixMatcher {
    /// Parse a hex prefix string into a matcher.
    /// Assumes input is already validated as uppercase hex.
    pub fn new(prefix: &str) -> Self {
        let prefix = prefix.to_ascii_uppercase();
        let mut full_bytes = Vec::new();
        let chars: Vec<u8> = prefix.bytes().collect();

        let mut i = 0;
        while i + 1 < chars.len() {
            let hi = nibble_from_ascii(chars[i]);
            let lo = nibble_from_ascii(chars[i + 1]);
            full_bytes.push((hi << 4) | lo);
            i += 2;
        }

        let trailing_nibble = if i < chars.len() {
            Some(nibble_from_ascii(chars[i]))
        } else {
            None
        };

        PrefixMatcher {
            full_bytes,
            trailing_nibble,
        }
    }

    /// Check if a 32-byte public key matches this prefix.
    #[inline(always)]
    pub fn matches(&self, public_key: &[u8; 32]) -> bool {
        // Check full bytes
        for (i, &expected) in self.full_bytes.iter().enumerate() {
            if public_key[i] != expected {
                return false;
            }
        }

        // Check trailing nibble (high nibble of next byte)
        if let Some(nibble) = self.trailing_nibble {
            let idx = self.full_bytes.len();
            if (public_key[idx] >> 4) != nibble {
                return false;
            }
        }

        true
    }
}

fn nibble_from_ascii(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'A'..=b'F' => c - b'A' + 10,
        b'a'..=b'f' => c - b'a' + 10,
        _ => unreachable!("validated hex input"),
    }
}

/// Check if a public key should be skipped (starts with 00 or FF).
#[inline(always)]
fn should_skip(public_key: &[u8; 32]) -> bool {
    public_key[0] == 0x00 || public_key[0] == 0xFF
}

/// Internal result containing keypair + which prefix matched.
struct MatchResult {
    keypair: MeshCoreKeypair,
    matched_prefix: String,
}

/// Handle for a running vanity key search.
/// Exposes atomics so a TUI render loop can poll progress directly.
pub struct SearchHandle {
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
    result: Arc<Mutex<Option<MatchResult>>>,
    start: Instant,
    workers: Vec<JoinHandle<()>>,
}

impl SearchHandle {
    /// Start a vanity key search in background threads.
    pub fn start(prefixes: &[String], num_threads: usize) -> Self {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let matchers = Arc::clone(&matchers);
            let found = Arc::clone(&found);
            let total_attempts = Arc::clone(&attempts);
            let result = Arc::clone(&result);

            workers.push(thread::spawn(move || {
                let mut seed = [0u8; 32];
                let mut local_count: u64 = 0;

                while !found.load(Ordering::Relaxed) {
                    OsRng.fill_bytes(&mut seed);
                    let kp = generate_keypair(&seed);
                    local_count += 1;

                    if local_count % BATCH_SIZE == 0 {
                        total_attempts.fetch_add(BATCH_SIZE, Ordering::Relaxed);
                    }

                    if should_skip(&kp.public_key) {
                        continue;
                    }

                    if let Some(matched) = matchers.iter().find(|(_, m)| m.matches(&kp.public_key))
                    {
                        total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
                        found.store(true, Ordering::Relaxed);
                        *result.lock().unwrap() = Some(MatchResult {
                            keypair: kp,
                            matched_prefix: matched.0.clone(),
                        });
                        return;
                    }
                }

                total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
            }));
        }

        SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        }
    }

    /// Check if a match has been found.
    pub fn is_done(&self) -> bool {
        self.found.load(Ordering::Relaxed)
    }

    /// Get current search statistics.
    pub fn stats(&self, expected: u64) -> SearchStats {
        let attempts = self.attempts.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_secs_f64();
        SearchStats {
            attempts,
            expected_attempts: expected,
            elapsed_secs: elapsed,
            keys_per_sec: if elapsed > 0.0 {
                attempts as f64 / elapsed
            } else {
                0.0
            },
        }
    }

    /// Join all worker threads and return the result.
    pub fn finish(self) -> SearchResult {
        for h in self.workers {
            h.join().unwrap();
        }

        let elapsed = self.start.elapsed().as_secs_f64();
        let attempts = self.attempts.load(Ordering::Relaxed);
        let m = self.result.lock().unwrap().take().expect("search found a key");

        SearchResult {
            public_key: hex::encode_upper(m.keypair.public_key),
            private_key: hex::encode_upper(m.keypair.private_key),
            matched_prefix: m.matched_prefix,
            attempts,
            elapsed_secs: elapsed,
        }
    }

    /// Start a vanity key search on GPU. Spawns a single thread that runs the
    /// GPU launch loop, updating the same atomics as the CPU path.
    #[cfg(feature = "cuda")]
    pub fn start_gpu(prefixes: &[String]) -> Result<Self, crate::gpu::GpuError> {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut gpu_searcher = crate::gpu::GpuSearcher::new(prefixes)?;
        let prefixes_owned: Vec<String> = prefixes.to_vec();

        let found_clone = Arc::clone(&found);
        let attempts_clone = Arc::clone(&attempts);
        let result_clone = Arc::clone(&result);

        let worker = thread::spawn(move || {
            // Use OS randomness for the initial nonce so different runs don't overlap
            let mut nonce_bytes = [0u8; 8];
            OsRng.fill_bytes(&mut nonce_bytes);
            let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

            // Build CPU-side matchers for identifying which prefix matched
            let matchers: Vec<(String, PrefixMatcher)> = prefixes_owned
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect();

            while !found_clone.load(Ordering::Relaxed) {
                match gpu_searcher.search_batch(base_nonce) {
                    Ok(batch_result) => {
                        attempts_clone.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                        if let Some(kp) = batch_result.keypair {
                            found_clone.store(true, Ordering::Relaxed);
                            // Identify which prefix matched
                            let matched_prefix = matchers
                                .iter()
                                .find(|(_, m)| m.matches(&kp.public_key))
                                .map(|(p, _)| p.clone())
                                .unwrap_or_else(|| prefixes_owned[0].clone());
                            *result_clone.lock().unwrap() = Some(MatchResult {
                                keypair: kp,
                                matched_prefix,
                            });
                            return;
                        }
                        base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
                    }
                    Err(e) => {
                        eprintln!("GPU error: {}", e);
                        return;
                    }
                }
            }
        });

        Ok(SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers: vec![worker],
        })
    }

    /// Start a Metal GPU-only vanity key search.
    #[cfg(feature = "metal")]
    pub fn start_metal(prefixes: &[String]) -> Result<Self, crate::metal_gpu::MetalError> {
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let metal_searcher = crate::metal_gpu::MetalSearcher::new(prefixes)?;
        let prefixes_owned: Vec<String> = prefixes.to_vec();

        let found_clone = Arc::clone(&found);
        let attempts_clone = Arc::clone(&attempts);
        let result_clone = Arc::clone(&result);

        let worker = thread::spawn(move || {
            let mut nonce_bytes = [0u8; 8];
            OsRng.fill_bytes(&mut nonce_bytes);
            let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

            let matchers: Vec<(String, PrefixMatcher)> = prefixes_owned
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect();

            while !found_clone.load(Ordering::Relaxed) {
                match metal_searcher.search_batch(base_nonce) {
                    Ok(batch_result) => {
                        attempts_clone.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                        if let Some(kp) = batch_result.keypair {
                            found_clone.store(true, Ordering::Relaxed);
                            let matched_prefix = matchers
                                .iter()
                                .find(|(_, m)| m.matches(&kp.public_key))
                                .map(|(p, _)| p.clone())
                                .unwrap_or_else(|| prefixes_owned[0].clone());
                            *result_clone.lock().unwrap() = Some(MatchResult {
                                keypair: kp,
                                matched_prefix,
                            });
                            return;
                        }
                        base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
                    }
                    Err(e) => {
                        eprintln!("Metal GPU error: {}", e);
                        return;
                    }
                }
            }
        });

        Ok(SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers: vec![worker],
        })
    }

    /// Start a hybrid vanity key search: CPU threads + Metal GPU concurrently.
    #[cfg(feature = "metal")]
    pub fn start_hybrid(
        prefixes: &[String],
        cpu_threads: usize,
    ) -> Result<Self, crate::metal_gpu::MetalError> {
        let matchers: Arc<Vec<(String, PrefixMatcher)>> = Arc::new(
            prefixes
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect(),
        );
        let found = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(AtomicU64::new(0));
        let result: Arc<Mutex<Option<MatchResult>>> = Arc::new(Mutex::new(None));

        let mut workers = Vec::with_capacity(cpu_threads + 1);

        // Spawn CPU workers — same loop body as start()
        for _ in 0..cpu_threads {
            let matchers = Arc::clone(&matchers);
            let found = Arc::clone(&found);
            let total_attempts = Arc::clone(&attempts);
            let result = Arc::clone(&result);

            workers.push(thread::spawn(move || {
                let mut seed = [0u8; 32];
                let mut local_count: u64 = 0;

                while !found.load(Ordering::Relaxed) {
                    OsRng.fill_bytes(&mut seed);
                    let kp = generate_keypair(&seed);
                    local_count += 1;

                    if local_count % BATCH_SIZE == 0 {
                        total_attempts.fetch_add(BATCH_SIZE, Ordering::Relaxed);
                    }

                    if should_skip(&kp.public_key) {
                        continue;
                    }

                    if let Some(matched) =
                        matchers.iter().find(|(_, m)| m.matches(&kp.public_key))
                    {
                        total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
                        found.store(true, Ordering::Relaxed);
                        *result.lock().unwrap() = Some(MatchResult {
                            keypair: kp,
                            matched_prefix: matched.0.clone(),
                        });
                        return;
                    }
                }

                total_attempts.fetch_add(local_count % BATCH_SIZE, Ordering::Relaxed);
            }));
        }

        // Spawn Metal GPU dispatch thread
        let metal_searcher = crate::metal_gpu::MetalSearcher::new(prefixes)?;
        let prefixes_owned: Vec<String> = prefixes.to_vec();
        let found_clone = Arc::clone(&found);
        let attempts_clone = Arc::clone(&attempts);
        let result_clone = Arc::clone(&result);

        workers.push(thread::spawn(move || {
            let mut nonce_bytes = [0u8; 8];
            OsRng.fill_bytes(&mut nonce_bytes);
            let mut base_nonce: u64 = u64::from_le_bytes(nonce_bytes);

            let matchers: Vec<(String, PrefixMatcher)> = prefixes_owned
                .iter()
                .map(|p| (p.clone(), PrefixMatcher::new(p)))
                .collect();

            while !found_clone.load(Ordering::Relaxed) {
                match metal_searcher.search_batch(base_nonce) {
                    Ok(batch_result) => {
                        attempts_clone.fetch_add(batch_result.keys_checked, Ordering::Relaxed);
                        if let Some(kp) = batch_result.keypair {
                            found_clone.store(true, Ordering::Relaxed);
                            let matched_prefix = matchers
                                .iter()
                                .find(|(_, m)| m.matches(&kp.public_key))
                                .map(|(p, _)| p.clone())
                                .unwrap_or_else(|| prefixes_owned[0].clone());
                            *result_clone.lock().unwrap() = Some(MatchResult {
                                keypair: kp,
                                matched_prefix,
                            });
                            return;
                        }
                        base_nonce = base_nonce.wrapping_add(batch_result.keys_checked);
                    }
                    Err(e) => {
                        eprintln!("Metal GPU error: {}", e);
                        return;
                    }
                }
            }
        }));

        Ok(SearchHandle {
            found,
            attempts,
            result,
            start: Instant::now(),
            workers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matcher_full_bytes() {
        let m = PrefixMatcher::new("AB");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(m.matches(&key));

        key[0] = 0xAC;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_odd_nibble() {
        let m = PrefixMatcher::new("A");
        let mut key = [0u8; 32];
        key[0] = 0xA0;
        assert!(m.matches(&key));

        key[0] = 0xAF;
        assert!(m.matches(&key));

        key[0] = 0xB0;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_multi_byte() {
        let m = PrefixMatcher::new("ABCD");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        key[1] = 0xCD;
        assert!(m.matches(&key));

        key[1] = 0xCE;
        assert!(!m.matches(&key));
    }

    #[test]
    fn prefix_matcher_three_nibbles() {
        let m = PrefixMatcher::new("ABC");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        key[1] = 0xC0;
        assert!(m.matches(&key));

        key[1] = 0xCF;
        assert!(m.matches(&key));

        key[1] = 0xD0;
        assert!(!m.matches(&key));
    }

    #[test]
    fn skip_00_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0x00;
        assert!(should_skip(&key));
    }

    #[test]
    fn skip_ff_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0xFF;
        assert!(should_skip(&key));
    }

    #[test]
    fn no_skip_normal_prefix() {
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(!should_skip(&key));
    }

    #[test]
    fn prefix_matcher_case_insensitive() {
        let m = PrefixMatcher::new("ab");
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        assert!(m.matches(&key));
    }

    #[test]
    fn search_handle_finds_single_char_prefix() {
        let handle = SearchHandle::start(&["A".to_string()], 2);
        let result = handle.finish();
        assert!(
            result.public_key.starts_with('A'),
            "expected public key starting with A, got {}",
            result.public_key
        );
        assert_eq!(result.matched_prefix, "A");
    }

    #[test]
    fn search_handle_multiple_prefixes() {
        let handle = SearchHandle::start(&["A".to_string(), "B".to_string()], 2);
        let result = handle.finish();
        assert!(
            result.public_key.starts_with('A') || result.public_key.starts_with('B'),
            "expected public key starting with A or B, got {}",
            result.public_key
        );
        assert!(
            result.matched_prefix == "A" || result.matched_prefix == "B",
            "expected matched_prefix A or B, got {}",
            result.matched_prefix
        );
    }
}
