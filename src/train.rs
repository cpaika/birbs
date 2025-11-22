use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use rand::seq::SliceRandom;
use std::sync::{Arc, Mutex};

use crate::{
    checkpoint::CheckpointMetadata,
    dataset::BirdDataset,
    model::{accuracy, cross_entropy_loss, BirdCNN},
    tui::{TrainingMetrics, TrainingUI},
};

pub struct Trainer {
    pub device: Device,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
}

impl Trainer {
    pub fn new(batch_size: usize, epochs: usize, learning_rate: f64) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;

        Ok(Self {
            device,
            batch_size,
            epochs,
            learning_rate,
        })
    }

    /// Trains the model on the given dataset with checkpointing
    pub fn train(
        &self,
        model: &BirdCNN,
        optimizer: &mut impl Optimizer,
        train_data: &BirdDataset,
        test_data: &BirdDataset,
        image_size: usize,
        ui: Arc<Mutex<TrainingUI>>,
        varmap: &VarMap,
        checkpoint_every: usize,  // Save checkpoint every N epochs
    ) -> Result<()> {
        // Log initial info
        {
            let mut ui = ui.lock().unwrap();
            ui.add_log(format!("Using device: {:?}", self.device));
            ui.add_log(format!("Training samples: {}", train_data.image_paths.len()));
            ui.add_log(format!("Test samples: {}", test_data.image_paths.len()));
            ui.add_log(format!("Batch size: {}", self.batch_size));
            ui.add_log(format!("Epochs: {}", self.epochs));
            ui.add_log(format!("Learning rate: {}", self.learning_rate));
            ui.add_log("Starting training...".to_string());
        }

        let mut best_accuracy = 0.0f32;

        for epoch in 1..=self.epochs {
            // Training phase
            let train_loss = self.train_epoch(
                model,
                optimizer,
                train_data,
                image_size,
                epoch,
                ui.clone(),
            )?;

            // Evaluation phase
            let (test_loss, test_acc) = self.evaluate(
                model,
                test_data,
                image_size,
            )?;

            // Track best accuracy
            if test_acc > best_accuracy {
                best_accuracy = test_acc;
            }

            // Update UI with epoch metrics
            {
                let mut ui = ui.lock().unwrap();
                ui.add_metrics(TrainingMetrics {
                    epoch,
                    total_epochs: self.epochs,
                    train_loss,
                    test_loss,
                    test_accuracy: test_acc,
                    batch: 0,
                    total_batches: 0,
                    current_batch_loss: None,
                });
            }

            // Save checkpoint periodically
            if epoch % checkpoint_every == 0 || epoch == self.epochs {
                let checkpoint = CheckpointMetadata::new(
                    epoch,
                    train_loss,
                    test_loss,
                    test_acc,
                    best_accuracy,
                );

                // Save model weights
                save_model(varmap, "checkpoint_model.safetensors")?;

                // Save metadata
                checkpoint.save(std::path::Path::new("checkpoint_metadata.json"))?;

                {
                    let mut ui = ui.lock().unwrap();
                    ui.add_log(format!("💾 Checkpoint saved at epoch {} (acc: {:.2}%)", epoch, test_acc * 100.0));
                }
            }
        }

        {
            let mut ui = ui.lock().unwrap();
            ui.add_log("Training complete!".to_string());
            ui.add_log(format!("Best accuracy: {:.2}%", best_accuracy * 100.0));
        }

        Ok(())
    }

    /// Trains for one epoch
    fn train_epoch(
        &self,
        model: &BirdCNN,
        optimizer: &mut impl Optimizer,
        dataset: &BirdDataset,
        image_size: usize,
        _epoch: usize,
        ui: Arc<Mutex<TrainingUI>>,
    ) -> Result<f32> {
        let num_batches = (dataset.image_paths.len() + self.batch_size - 1) / self.batch_size;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..dataset.image_paths.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;
        let mut num_samples = 0;

        for batch_idx in 0..num_batches {
            if batch_idx % 10 == 0 {
                // Update UI with batch progress
                let mut ui_lock = ui.lock().unwrap();
                ui_lock.add_log(format!("Epoch batch {}/{} ({:.1}%)", batch_idx + 1, num_batches, (batch_idx as f32 / num_batches as f32) * 100.0));
                drop(ui_lock);
            }
            let start_idx = batch_idx * self.batch_size;
            let end_idx = ((batch_idx + 1) * self.batch_size).min(dataset.image_paths.len());
            let batch_indices = &indices[start_idx..end_idx];

            // Prepare batch - loads images on demand
            let (batch_images, valid_indices) = self.prepare_batch(batch_indices, dataset, image_size)?;
            let batch_labels = self.prepare_labels(&valid_indices, &dataset.labels)?;

            // Forward pass
            let logits = model.forward(&batch_images, true)?;

            // Compute loss
            let loss = cross_entropy_loss(&logits, &batch_labels)?;

            // Backward pass and update weights
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            // Track metrics (using actual number of valid images)
            let loss_val = loss.to_scalar::<f32>()?;
            total_loss += loss_val * valid_indices.len() as f32;
            num_samples += valid_indices.len();

            // Update UI with batch progress (every 10 batches to reduce lock contention)
            if batch_idx % 10 == 0 || batch_idx == num_batches - 1 {
                let mut ui = ui.lock().unwrap();
                ui.update_batch_progress(batch_idx + 1, num_batches, loss_val);
            }
        }

        Ok(total_loss / num_samples as f32)
    }

    /// Evaluates the model on test data
    fn evaluate(
        &self,
        model: &BirdCNN,
        dataset: &BirdDataset,
        image_size: usize,
    ) -> Result<(f32, f32)> {
        let num_batches = (dataset.image_paths.len() + self.batch_size - 1) / self.batch_size;
        let indices: Vec<usize> = (0..dataset.image_paths.len()).collect();

        let mut total_loss = 0.0;
        let mut total_correct = 0.0;
        let mut num_samples = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = ((batch_idx + 1) * self.batch_size).min(dataset.image_paths.len());
            let batch_indices = &indices[start_idx..end_idx];

            // Prepare batch - loads images on demand
            let (batch_images, valid_indices) = self.prepare_batch(batch_indices, dataset, image_size)?;
            let batch_labels = self.prepare_labels(&valid_indices, &dataset.labels)?;

            // Forward pass (no dropout in eval mode)
            let logits = model.forward(&batch_images, false)?;

            // Compute loss and accuracy
            let loss = cross_entropy_loss(&logits, &batch_labels)?;
            let acc = accuracy(&logits, &batch_labels)?;

            total_loss += loss.to_scalar::<f32>()? * valid_indices.len() as f32;
            total_correct += acc * valid_indices.len() as f32;
            num_samples += valid_indices.len();
        }

        Ok((
            total_loss / num_samples as f32,
            total_correct / num_samples as f32,
        ))
    }

    /// Prepares a batch of images as a tensor - loads images on demand for memory efficiency
    /// Returns (tensor, valid_indices) where valid_indices are the indices that successfully loaded
    fn prepare_batch(
        &self,
        indices: &[usize],
        dataset: &BirdDataset,
        image_size: usize,
    ) -> Result<(Tensor, Vec<usize>)> {
        let mut batch_data = Vec::new();
        let mut valid_indices = Vec::new();

        // Load each image on demand, skipping any that fail to load
        for &idx in indices {
            match dataset.load_image(idx) {
                Ok(img_tensor) => {
                    batch_data.extend_from_slice(&img_tensor);
                    valid_indices.push(idx);
                }
                Err(e) => {
                    // Continue with next image
                }
            }
        }

        // If we have no valid images in the batch, return an error
        if valid_indices.is_empty() {
            anyhow::bail!("All images in batch failed to load");
        }

        let actual_batch_size = valid_indices.len();
        let tensor = Tensor::from_vec(
            batch_data,
            &[actual_batch_size, 3, image_size, image_size],
            &self.device,
        )?;

        Ok((tensor, valid_indices))
    }

    /// Prepares a batch of labels as a tensor
    fn prepare_labels(&self, indices: &[usize], labels: &[usize]) -> Result<Tensor> {
        let batch_labels: Vec<u32> = indices.iter().map(|&idx| labels[idx] as u32).collect();

        let tensor = Tensor::from_vec(batch_labels, &[indices.len()], &self.device)?;

        Ok(tensor)
    }
}

/// Saves model weights to a file
pub fn save_model(varmap: &VarMap, path: &str) -> Result<()> {
    println!("Saving model to {}", path);
    varmap.save(path)?;
    println!("Model saved successfully!");
    Ok(())
}

/// Loads model weights from a file
pub fn load_model(varmap: &mut VarMap, path: &str) -> Result<()> {
    println!("Loading model from {}", path);
    varmap.load(path)?;
    println!("Model loaded successfully!");
    Ok(())
}
