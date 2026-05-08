use curve25519_dalek::EdwardsPoint;
use sha2::{Digest, Sha512};

use crate::types::MeshCoreKeypair;

/// Generate a MeshCore Ed25519 keypair from a 32-byte seed.
///
/// Algorithm (matching the mc-keygen web tool):
/// 1. SHA-512(seed) -> 64 bytes
/// 2. Clamp first 32 bytes: [0] &= 248, [31] &= 63, [31] |= 64
/// 3. Multiply Ed25519 base point by clamped scalar -> 32-byte public key
/// 4. Private key = clamped_scalar[0..32] || sha512_digest[32..64]
///
/// Kept for compatibility with anyone reproducing keys via the web tool's
/// seed-based derivation; the search loop uses `generate_keypair_from_random_bytes`.
#[allow(dead_code)]
pub fn generate_keypair(seed: &[u8; 32]) -> MeshCoreKeypair {
    let hash = Sha512::digest(seed);

    let mut scalar_bytes = [0u8; 32];
    scalar_bytes.copy_from_slice(&hash[..32]);

    // Clamp the scalar
    scalar_bytes[0] &= 248;
    scalar_bytes[31] &= 63;
    scalar_bytes[31] |= 64;

    // Multiply base point by clamped scalar
    // mul_base_clamped applies clamping internally, but clamping is idempotent
    let public_point: EdwardsPoint = EdwardsPoint::mul_base_clamped(scalar_bytes);
    let public_key = public_point.compress().to_bytes();

    // Private key = clamped scalar || second half of SHA-512 digest
    let mut private_key = [0u8; 64];
    private_key[..32].copy_from_slice(&scalar_bytes);
    private_key[32..].copy_from_slice(&hash[32..]);

    MeshCoreKeypair {
        public_key,
        private_key,
    }
}

/// Build a MeshCore keypair directly from 64 bytes of cryptographic randomness:
/// `scalar_src` (clamped to form the scalar) and `prefix` (used as the second half
/// of the expanded private key, fed into the signing nonce hash).
///
/// This skips the SHA-512 derivation step in `generate_keypair`, since the seed
/// it would expand from is not exposed. Used by both the CPU search loop (with
/// bytes drawn directly from `OsRng`) and the GPU search path (with bytes from
/// Philox(key, counter)).
pub fn generate_keypair_from_random_bytes(
    scalar_src: &[u8; 32],
    prefix: &[u8; 32],
) -> MeshCoreKeypair {
    let mut scalar_bytes = *scalar_src;
    scalar_bytes[0] &= 248;
    scalar_bytes[31] &= 63;
    scalar_bytes[31] |= 64;

    let public_point: EdwardsPoint = EdwardsPoint::mul_base_clamped(scalar_bytes);
    let public_key = public_point.compress().to_bytes();

    let mut private_key = [0u8; 64];
    private_key[..32].copy_from_slice(&scalar_bytes);
    private_key[32..].copy_from_slice(prefix);

    MeshCoreKeypair {
        public_key,
        private_key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_keygen() {
        let seed = [42u8; 32];
        let kp1 = generate_keypair(&seed);
        let kp2 = generate_keypair(&seed);
        assert_eq!(kp1.public_key, kp2.public_key);
        assert_eq!(kp1.private_key, kp2.private_key);
    }

    #[test]
    fn different_seeds_different_keys() {
        let kp1 = generate_keypair(&[1u8; 32]);
        let kp2 = generate_keypair(&[2u8; 32]);
        assert_ne!(kp1.public_key, kp2.public_key);
    }

    #[test]
    fn clamping_applied() {
        let seed = [0xFFu8; 32];
        let hash = Sha512::digest(&seed);
        let kp = generate_keypair(&seed);

        // Verify clamping was applied to the private key scalar
        assert_eq!(kp.private_key[0] & 7, 0, "low 3 bits should be cleared");
        assert_eq!(kp.private_key[31] & 128, 0, "high bit should be cleared");
        assert_eq!(kp.private_key[31] & 64, 64, "second-highest bit should be set");

        // Second half should match SHA-512 digest
        assert_eq!(&kp.private_key[32..], &hash[32..]);
    }

    #[test]
    fn public_key_is_32_bytes() {
        let kp = generate_keypair(&[7u8; 32]);
        assert_eq!(kp.public_key.len(), 32);
    }

    #[test]
    fn private_key_is_64_bytes() {
        let kp = generate_keypair(&[7u8; 32]);
        assert_eq!(kp.private_key.len(), 64);
    }
}
