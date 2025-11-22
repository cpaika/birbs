use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{conv2d, linear, ops, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

/// Convolutional Neural Network for bird classification
pub struct BirdCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout_prob: f32,
}

impl BirdCNN {
    /// Creates a new CNN model
    ///
    /// Architecture:
    /// - Conv1: 3 -> 32 channels, 3x3 kernel
    /// - MaxPool 2x2
    /// - Conv2: 32 -> 64 channels, 3x3 kernel
    /// - MaxPool 2x2
    /// - Conv3: 64 -> 128 channels, 3x3 kernel
    /// - MaxPool 2x2
    /// - Flatten
    /// - FC1: flattened -> 256
    /// - Dropout
    /// - FC2: 256 -> num_classes
    pub fn new(vb: VarBuilder, num_classes: usize, image_size: usize) -> Result<Self> {
        let conv1 = conv2d(
            3,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let conv2 = conv2d(
            32,
            64,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        let conv3 = conv2d(
            64,
            128,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv3"),
        )?;

        // Calculate the size after 3 max-pooling layers (each reduces size by 2)
        let feature_size = image_size / 8; // 3 max-pool layers: size / 2^3
        let flattened_size = 128 * feature_size * feature_size;

        let fc1 = linear(flattened_size, 256, vb.pp("fc1"))?;
        let fc2 = linear(256, num_classes, vb.pp("fc2"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            dropout_prob: 0.5,
        })
    }

    /// Forward pass through the network
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // Conv block 1: Conv -> ReLU -> MaxPool
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?;

        // Conv block 2: Conv -> ReLU -> MaxPool
        let xs = self.conv2.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?;

        // Conv block 3: Conv -> ReLU -> MaxPool
        let xs = self.conv3.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?;

        // Flatten
        let (b, c, h, w) = xs.dims4()?;
        let xs = xs.reshape(&[b, c * h * w])?;

        // Fully connected block 1: Linear -> ReLU -> Dropout
        let xs = self.fc1.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = if train {
            ops::dropout(&xs, self.dropout_prob)?
        } else {
            xs
        };

        // Fully connected block 2: Linear (output)
        let xs = self.fc2.forward(&xs)?;

        Ok(xs)
    }
}

/// Computes cross-entropy loss
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs = ops::log_softmax(logits, 1)?;

    // Reshape targets from [batch_size] to [batch_size, 1] for gather
    let targets_reshaped = targets.unsqueeze(1)?;

    // Gather the log probabilities for the correct classes
    let nll = log_probs.gather(&targets_reshaped, 1)?;

    // Squeeze to remove the extra dimension and compute mean
    let loss = nll.squeeze(1)?.neg()?.mean_all()?;
    Ok(loss)
}

/// Computes accuracy
pub fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f32> {
    let predictions = logits.argmax(1)?;
    let correct = predictions.eq(targets)?.to_dtype(DType::F32)?.sum_all()?;
    let total = targets.elem_count() as f32;
    Ok(correct.to_scalar::<f32>()? / total)
}
