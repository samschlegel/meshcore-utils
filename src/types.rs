use std::fmt;

use serde::Serialize;

/// A MeshCore Ed25519 keypair with 32-byte public key and 64-byte private key.
pub struct MeshCoreKeypair {
    pub public_key: [u8; 32],
    pub private_key: [u8; 64],
}

/// Result of a successful vanity key search.
#[derive(Serialize)]
pub struct SearchResult {
    pub public_key: String,
    pub private_key: String,
    pub matched_prefix: String,
    pub attempts: u64,
    pub elapsed_secs: f64,
}

/// Error returned when a search completes without finding a match
/// (e.g., all GPU workers errored out).
#[derive(Debug)]
pub struct SearchError {
    pub attempts: u64,
    pub elapsed_secs: f64,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "search ended without finding a match after {} attempts ({:.1}s)",
            self.attempts, self.elapsed_secs
        )
    }
}

impl std::error::Error for SearchError {}

/// Live statistics for progress reporting.
pub struct SearchStats {
    pub attempts: u64,
    pub expected_attempts: u64,
    pub elapsed_secs: f64,
    pub keys_per_sec: f64,
}
