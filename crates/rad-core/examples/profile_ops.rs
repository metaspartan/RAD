use anyhow::Result;
use rad_core::InklingEngine;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::var("RAD_MODEL")
        .unwrap_or_else(|_| "./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let mut eng = InklingEngine::load_multi(Path::new(&model_path), 2048, 2)?;
    eng.profile_batch_cold(8)?;
    eng.profile_batch_cold(32)?;
    eng.profile_batch_cold(1024)?;
    Ok(())
}
