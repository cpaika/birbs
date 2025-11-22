use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub epoch: usize,
    pub train_loss: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
    pub best_accuracy: f32,
    pub timestamp: String,
}

impl CheckpointMetadata {
    pub fn new(
        epoch: usize,
        train_loss: f32,
        test_loss: f32,
        test_accuracy: f32,
        best_accuracy: f32,
    ) -> Self {
        use chrono::Local;
        Self {
            epoch,
            train_loss,
            test_loss,
            test_accuracy,
            best_accuracy,
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }

    /// Save metadata to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load metadata from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&json)?;
        Ok(metadata)
    }
}

/// Check if checkpoint exists
pub fn checkpoint_exists() -> bool {
    Path::new("checkpoint_model.safetensors").exists()
        && Path::new("checkpoint_metadata.json").exists()
}

/// Get checkpoint info if it exists
pub fn get_checkpoint_info() -> Option<CheckpointMetadata> {
    if checkpoint_exists() {
        CheckpointMetadata::load(Path::new("checkpoint_metadata.json")).ok()
    } else {
        None
    }
}
