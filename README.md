# mc-keygen

Vanity Ed25519 key generator for [MeshCore](https://github.com/ripplebiz/MeshCore). Find keys whose public key starts with a chosen hex prefix.

## Usage

```
mc-keygen <PREFIX>... [OPTIONS]
```

**Options:**
- `-t, --threads <N>` — worker threads (default: all cores)
- `--json` — output result as JSON (no TUI, no color)
- `--gpu` — use CUDA GPU acceleration (requires `cuda` feature)
- `--verify` — cross-check GPU keygen against CPU (requires `cuda` or `metal` feature)

**Examples:**
```bash
mc-keygen AB             # find a key starting with AB
mc-keygen AB CD EF       # find a key matching ANY of these prefixes
mc-keygen DEAD -t 4      # use 4 threads
mc-keygen AB --json      # machine-readable output
mc-keygen ABCDEF --gpu   # use GPU for longer prefixes
```

Multiple prefixes can be passed in a single invocation — every key is checked against all of them, so searching for N prefixes is ~N× more efficient than N separate runs. The JSON output includes a `matched_prefix` field.

### Search difficulty

Each hex character multiplies expected attempts by 16:

| Prefix | Expected attempts | CPU time¹ | GPU time¹ |
|--------|-------------------|-----------|-----------|
| 1 char | 16                | instant   | instant   |
| 2 char | 256               | instant   | instant   |
| 3 char | 4,096             | instant   | instant   |
| 4 char | 65,536            | instant   | instant   |
| 5 char | ~1M               | <1s       | instant   |
| 6 char | ~16M              | ~12s      | <1s       |
| 7 char | ~268M             | ~3.5m     | ~4s       |
| 8 char | ~4.3B             | ~55m      | ~63s      |

¹ CPU: Ryzen 9 7950X3D, 32 threads (~1.3M keys/s). GPU: RTX 4090 (~68M keys/s). See [docs/gpu.md](docs/gpu.md) for details.

## Building

```bash
cargo build --release                    # CPU only
cargo build --release --features cuda    # NVIDIA GPU (requires CUDA Toolkit)
cargo build --release --features metal   # Apple GPU (macOS, Apple Silicon)
```

NVIDIA GPU support requires the CUDA Toolkit. See [docs/gpu.md](docs/gpu.md) for setup details.

Apple Metal builds auto-detect the GPU and run hybrid CPU+GPU mode by default.

## How it works

1. Generate random 32-byte seed
2. SHA-512 hash, clamp scalar (per Ed25519 spec), derive public key
3. Check if public key hex starts with the target prefix
4. Repeat across all cores (or GPU threads) until a match is found

Keys starting with `00` or `FF` are skipped (reserved by MeshCore).

## Sources

- [MeshCore](https://github.com/ripplebiz/MeshCore) — the mesh networking firmware these keys are for
- [MeshCore mc-keygen web tool](https://gessaman.com/mc-keygen/) — reference implementation of the key generation algorithm
- [Ed25519 / RFC 8032](https://datatracker.ietf.org/doc/html/rfc8032) — the signature scheme spec
- [curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek) — Rust Ed25519 elliptic curve library
- [ratatui](https://github.com/ratatui/ratatui) — TUI framework for the progress display
