use anyhow::Result;
use candle_nn::{Optimizer, VarBuilder, VarMap};

use imgs::checkpoint;
use imgs::dataset::BirdDataset;
use imgs::model::BirdCNN;
use imgs::train::{load_model, save_model, Trainer};
use imgs::tui::run_training_ui;

fn main() -> Result<()> {
    // Configuration
    let data_dir = "./data/birds";
    let image_size = 256; // 256x256 images for better detail
    let batch_size = 16; // Reduced batch size for larger images
    let epochs = 100;
    let learning_rate = 0.001;
    let train_ratio = 0.8;
    let min_non_bird_images = 3000;
    let checkpoint_every = 5; // Save checkpoint every 5 epochs

    // Check for existing checkpoint
    if let Some(checkpoint_info) = checkpoint::get_checkpoint_info() {
    }

    // Check and download non-bird images if needed
    let non_bird_dir = std::path::Path::new("./data/birds/not-bird");

    if !non_bird_dir.exists() {
        std::fs::create_dir_all(non_bird_dir)?;
    }

    let existing_count = std::fs::read_dir(non_bird_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file()
                && matches!(
                    e.path().extension().and_then(|s| s.to_str()),
                    Some("jpg") | Some("jpeg") | Some("png")
                )
        })
        .count();


    if existing_count < min_non_bird_images {
        let needed = min_non_bird_images - existing_count;

        BirdDataset::download_non_bird_images(non_bird_dir, needed)?;

    } else {
    }

    // Load and prepare the dataset
    let dataset = BirdDataset::load_from_directory(
        std::path::Path::new(data_dir),
        image_size as u32,
    )?;

    if dataset.image_paths.is_empty() {
        std::process::exit(1);
    }


    // Split into train and test sets
    let (train_data, test_data) = dataset.train_test_split(train_ratio)?;

    // Initialize the model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &candle_core::Device::cuda_if_available(0)?);
    let model = BirdCNN::new(vb, dataset.num_classes, image_size)?;

    // Load from checkpoint if it exists
    if checkpoint::checkpoint_exists() {
        load_model(&mut varmap, "checkpoint_model.safetensors")?;
    } else {
    }

    // Initialize optimizer
    let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), learning_rate)?;

    // Run training with TUI
    let varmap_clone = varmap.clone();
    run_training_ui(move |ui| {
        {
            let mut ui = ui.lock().unwrap();
            ui.add_log("=== Bird Image Classification ===".to_string());
            ui.add_log(format!("Total images: {}", dataset.image_paths.len()));
            ui.add_log(format!("Classes: {}", dataset.num_classes));
            ui.add_log(format!("Train images: {}", train_data.image_paths.len()));
            ui.add_log(format!("Test images: {}", test_data.image_paths.len()));
        }

        // Create trainer and train
        let trainer = Trainer::new(batch_size, epochs, learning_rate)?;
        trainer.train(&model, &mut optimizer, &train_data, &test_data, image_size, ui.clone(), &varmap_clone, checkpoint_every)?;

        // Save the model
        {
            let mut ui = ui.lock().unwrap();
            ui.add_log("Saving model...".to_string());
        }
        save_model(&varmap_clone, "bird_classifier.safetensors")?;
        {
            let mut ui = ui.lock().unwrap();
            ui.add_log("Model saved to bird_classifier.safetensors".to_string());
        }

        Ok(())
    })?;


    Ok(())
}
