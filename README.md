# Bird Image Classifier

A Rust-based binary image classifier using Hugging Face's Candle deep learning framework. This project trains a CNN to distinguish between birds and non-birds, demonstrating binary classification with deep learning in pure Rust.

## Features

- **Binary Classification**: Distinguishes birds from non-birds (trees, landscapes, etc.)
- **CNN Architecture**: 3-layer convolutional neural network with max pooling and dropout
- **Hugging Face Candle**: Pure Rust ML framework for efficient training
- **Live Training UI**: Real-time loss curves and metrics visualization
- **Dataset Loading**: Automated setup with balanced positive/negative examples
- **Inference CLI**: Easy-to-use command for classifying new images
- **Model Persistence**: Save and load trained models using SafeTensors format
- **GPU Support**: Automatically uses CUDA if available, falls back to CPU

## Architecture

```
Input: 3x64x64 RGB Image
    ↓
Conv1: 3 → 32 channels (3x3, padding=1) + ReLU + MaxPool(2x2)
    ↓
Conv2: 32 → 64 channels (3x3, padding=1) + ReLU + MaxPool(2x2)
    ↓
Conv3: 64 → 128 channels (3x3, padding=1) + ReLU + MaxPool(2x2)
    ↓
Flatten
    ↓
FC1: 8192 → 256 + ReLU + Dropout(0.5)
    ↓
FC2: 256 → num_classes
```

## Quick Start

### 1. Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- (Optional) CUDA toolkit for GPU acceleration

### 2. Get the Datasets

This project trains a **binary classifier** to distinguish birds from non-birds.

**Download Required Datasets:**

1. **NABirds dataset** (48,000+ bird images):
   - **[Download NABirds](https://dl.allaboutbirds.org/merlin---computer-vision--terms-of-use?submissionGuid=18ceb8c4-7fde-4651-98ff-90dba028bccd)**
   - File: `nabirds.tar.gz`

2. **Trees dataset** (non-bird images for negative examples):
   - **[Download from Kaggle](https://www.kaggle.com/datasets/mexwell/5m-trees-dataset)**
   - Requires Kaggle account (free)
   - File: `archive.zip` or similar

### 3. Setup Dataset & Verify

Run the setup script to extract and prepare both datasets:

```bash
./setup_dataset.sh ~/Downloads/nabirds.tar.gz ~/Downloads/archive.zip
```

This will:
- ✓ Extract bird images to `./data/birds/bird/` (~48,000 images)
- ✓ Extract tree images to `./data/birds/not-bird/` (balanced to match bird count)
- ✓ Create binary classification structure (2 classes: bird, not-bird)
- ✓ Run tests to verify the dataset loads correctly

**Note:** You can run with just the NABirds dataset initially:
```bash
./setup_dataset.sh ~/Downloads/nabirds.tar.gz
```
Then add the trees dataset later.

### 4. Train the Model

```bash
cargo run --release
```

The training will **automatically**:
1. **Check non-bird dataset**: Verify you have at least 3,000 non-bird images
2. **Auto-download if needed**: If you have fewer than 3,000 non-bird images, it will automatically download high-quality diverse photos from Unsplash (via Picsum)
3. **Filter bird images**: When using HuggingFace datasets, automatically filters out any images labeled with bird-related keywords
4. **Load all images**: Load images from both classes (bird and not-bird)
5. **Split dataset**: 80% train / 20% test
6. **Train for 100 epochs**: With progress tracking in a live TUI
7. **Save model**: Save the trained model to `bird_classifier.safetensors`

**No manual dataset preparation needed for non-bird images!** Just run the training and it will handle everything automatically.

**Expected output:** The model will learn to classify images as either "bird" or "not-bird" with high accuracy.

### 5. Run Inference

Classify any bird image with your trained model:

```bash
cargo run --bin predict --release -- path/to/bird.jpg
```

---

## Alternative Dataset Options

### Option 1: Use Your Own Images

For binary classification, organize your images in this structure:

```
data/birds/
├── bird/
│   ├── robin.jpg
│   ├── eagle.jpg
│   ├── sparrow.jpg
│   └── ... (all bird images)
└── not-bird/
    ├── tree1.jpg
    ├── cat.jpg
    ├── car.jpg
    └── ... (all non-bird images)
```

**Important:** Balance your dataset! Aim for roughly equal numbers of bird and not-bird images.

### Option 2: Other Non-Bird Datasets

For negative examples (not-bird), you can use:

- **CIFAR-10/CIFAR-100**: Contains cats, dogs, cars, trucks, etc.
- **ImageNet subsets**: Animals, objects, scenes
- **COCO dataset**: Various objects and scenes
- **Your own photos**: Landscapes, objects, people, etc.

Just place them in `data/birds/not-bird/` directory.

## Usage

### Training

Run the training script:

```bash
cargo run --release
```

The program will:
1. Load images from `./data/birds/`
2. Split into 80% train, 20% test
3. Train for 100 epochs
4. Save the model to `bird_classifier.safetensors`

### Configuration

You can modify training parameters in `src/main.rs`:

```rust
let image_size = 64;               // Image dimensions (64x64)
let batch_size = 32;               // Batch size
let epochs = 100;                  // Number of training epochs
let learning_rate = 0.001;         // Learning rate for SGD
let train_ratio = 0.8;             // Train/test split ratio
let min_non_bird_images = 3000;    // Minimum non-bird images (auto-downloads if needed)
```

### Manual Non-Bird Image Download

If you want to manually download non-bird images ahead of time:

```bash
cargo run --bin download_samples --release -- 3000
```

This will download 3,000 diverse images from Unsplash in parallel (much faster than sequential).

### Inference

After training, use the prediction command to classify bird images:

```bash
cargo run --bin predict --release -- path/to/bird_image.jpg
```

**Example:**
```bash
cargo run --bin predict --release -- ~/Downloads/robin.jpg
```

**Output:**
```
🔍 Bird Species Classifier

📂 Loading class names...
   Found 12 bird species

🖼️  Loading and preprocessing image: ~/Downloads/robin.jpg
   ✓ Image preprocessed to 64x64

🧠 Loading model from bird_classifier.safetensors...
   Using device: Cpu
   ✓ Model loaded

🚀 Running inference...
   ✓ Inference complete

📊 Top 2 Predictions:
┌─────┬─────────────────────────────┬─────────────┬────────────┐
│ Rank│ Class                       │ Probability │ Confidence │
├─────┼─────────────────────────────┼─────────────┼────────────┤
│   1 │ bird                        │      92.34% │ ██████████████████│
│   2 │ not-bird                    │       7.66% │ █         │
└─────┴─────────────────────────────┴─────────────┴────────────┘

🎯 Prediction: bird (92.3% - High confidence)
```

The command will:
- Load the trained model weights
- Preprocess your image to 64x64
- Run inference with softmax
- Display predictions for bird vs not-bird with confidence bars
- Warn if confidence is low

## Project Structure

```
src/
├── lib.rs             # Library exports
├── main.rs            # Training entry point with TUI + auto-download
├── dataset.rs         # Dataset loading, preprocessing, and downloading
├── model.rs           # CNN architecture
├── train.rs           # Training loop and evaluation
├── tui.rs             # Terminal UI for training visualization
└── bin/
    ├── predict.rs         # Inference command for classifying images
    ├── download_samples.rs # Manual non-bird image downloader
    └── download_hf.rs     # HuggingFace dataset downloader (with bird filtering)

Cargo.toml          # Dependencies
README.md           # This file
setup_dataset.sh    # Dataset setup script (for bird images)
```

## Dependencies

- **candle-core** & **candle-nn**: Deep learning framework
- **image**: Image loading and processing
- **reqwest**: HTTP client for dataset downloading
- **serde**: Serialization framework
- **indicatif**: Progress bars
- **anyhow**: Error handling
- **rand**: Random number generation

## Training Output

Example training output:

```
=== Bird Image Classification with Rust and Candle ===

Step 1: Loading bird dataset...
Found 10 bird species/classes
Loaded 1000 images across 10 classes

Step 2: Splitting dataset...
  Train: 800 images | Test: 200 images

Step 3: Initializing CNN model...
  Model initialized successfully!

Step 4: Initializing optimizer...
  Using SGD with learning rate: 0.001

Step 5: Training the model...

Epoch 1/100
  [00:00:45] ████████████████████████ 1200/1200 loss: 2.1234
  Train Loss: 2.2145 | Test Loss: 2.0987 | Test Accuracy: 25.50%

Epoch 2/100
  [00:00:43] ████████████████████████ 1200/1200 loss: 1.8765
  Train Loss: 1.9234 | Test Loss: 1.8654 | Test Accuracy: 35.00%

...

Epoch 100/100
  [00:00:41] ████████████████████████ 1200/1200 loss: 0.3421
  Train Loss: 0.3521 | Test Loss: 0.4123 | Test Accuracy: 89.50%

=== Training Complete! ===
```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is installed for faster training
2. **Batch Size**: Increase if you have more GPU memory
3. **Image Size**: Larger images (128x128, 224x224) improve accuracy but slow training
4. **Data Augmentation**: Add random flips, rotations, and color jitter
5. **Transfer Learning**: Use pretrained models like ResNet for better results

## Troubleshooting

**"No images found!"**
- Check that `data/birds/` exists and contains subdirectories
- Verify images are in JPG, JPEG, or PNG format

**Out of memory errors**
- Reduce `batch_size` or `image_size`
- Train on CPU if GPU memory is limited

**Slow training**
- Use `--release` flag when running
- Enable CUDA support
- Reduce image size or dataset

## Future Improvements

- [ ] Data augmentation
- [ ] Transfer learning with pretrained models
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Confusion matrix visualization
- [ ] Web interface for inference
- [ ] Export to ONNX format

## License

MIT

## Acknowledgments

- Hugging Face for the Candle framework
- Caltech-UCSD for the CUB-200-2011 dataset
- The Rust ML community
