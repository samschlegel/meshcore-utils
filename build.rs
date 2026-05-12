use vergen_gix::{Emitter, GixBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gix = GixBuilder::all_git()?;

    // Default behavior is to NOT fail on error (source tarballs without .git
    // still compile); env!() reads of missing vars are guarded with
    // option_env!() in src/bench.rs.
    Emitter::default().add_instructions(&gix)?.emit()?;

    Ok(())
}
