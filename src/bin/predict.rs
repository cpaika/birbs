use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{ops, VarBuilder, VarMap};
use std::env;

// Import from the library
use birbs::model::BirdCNN;

fn preprocess_image(img_path: &str, size: u32) -> Result<Vec<f32>> {
    // Load image
    let img = image::open(img_path)
        .with_context(|| format!("Failed to open image: {}", img_path))?;

    // Resize to expected size
    let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Convert to CHW format (Channels, Height, Width) and normalize to [0, 1]
    let (width, height) = rgb.dimensions();
    let mut tensor = vec![0.0f32; (3 * width * height) as usize];

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let base_idx = (y * width + x) as usize;
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // CHW format: all reds, then all greens, then all blues
        tensor[base_idx] = r;
        tensor[(width * height) as usize + base_idx] = g;
        tensor[(2 * width * height) as usize + base_idx] = b;
    }

    Ok(tensor)
}

fn load_class_names() -> Result<Vec<String>> {
    // Load class names from the data directory
    let data_path = std::path::Path::new("./data/birds");

    if !data_path.exists() {
        anyhow::bail!("Dataset directory not found: {:?}. Make sure you have the training data.", data_path);
    }

    let mut class_names: Vec<String> = std::fs::read_dir(data_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();

    class_names.sort();

    if class_names.is_empty() {
        anyhow::bail!("No class directories found in {:?}", data_path);
    }

    Ok(class_names)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        std::process::exit(1);
    }

    let image_path = &args[1];
    let model_path = "bird_classifier.safetensors";
    let image_size: usize = 64;

    println!("🔍 Binary Bird Classifier\n");

    // Check if model exists
    if !std::path::Path::new(model_path).exists() {
        std::process::exit(1);
    }

    // Load class names
    println!("📂 Loading classes...");
    let class_names = load_class_names()?;
    let num_classes = class_names.len();
    if num_classes == 2 {
        println!("   Binary classification: {} classes (bird, not-bird)\n", num_classes);
    } else {
        println!("   Found {} classes\n", num_classes);
    }

    // Preprocess image
    println!("🖼️  Loading and preprocessing image: {}", image_path);
    let image_tensor = preprocess_image(image_path, image_size as u32)?;
    println!("   ✓ Image preprocessed to {}x{}\n", image_size, image_size);

    // Load model
    println!("🧠 Loading model from {}...", model_path);
    let device = Device::cuda_if_available(0)?;
    println!("   Using device: {:?}", device);

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;

    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = BirdCNN::new(vb, num_classes, image_size)?;
    println!("   ✓ Model loaded\n");

    // Prepare tensor with batch dimension
    let input_tensor = Tensor::from_vec(
        image_tensor,
        (1, 3, image_size, image_size), // [batch=1, channels=3, height, width]
        &device,
    )?;

    // Run inference
    println!("🚀 Running inference...");
    let logits = model.forward(&input_tensor, false)?; // train=false for inference
    let probabilities = ops::softmax(&logits, 1)?;

    // Get predictions
    let probs_vec = probabilities.to_vec2::<f32>()?;
    let probs = &probs_vec[0]; // Get first (and only) batch

    // Find top 5 predictions
    let mut predictions: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(idx, &prob)| (idx, prob))
        .collect();
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("   ✓ Inference complete\n");

    // Display results
    let num_to_show = predictions.len().min(5);
    println!("📊 Top {} Predictions:", num_to_show);
    println!("┌─────┬─────────────────────────────┬─────────────┬────────────────────┐");
    println!("│ Rank│ Class                       │ Probability │ Confidence         │");
    println!("├─────┼─────────────────────────────┼─────────────┼────────────────────┤");

    for (rank, &(class_idx, prob)) in predictions.iter().take(num_to_show).enumerate() {
        let class = &class_names[class_idx];
        let percentage = prob * 100.0;
        let bar_length = ((percentage / 100.0) * 20.0) as usize;
        let bar = "█".repeat(bar_length);

        println!(
            "│ {:>3} │ {:<27} │ {:>10.2}% │ {:<18} │",
            rank + 1,
            if class.len() > 27 {
                &class[..27]
            } else {
                class
            },
            percentage,
            bar
        );
    }

    println!("└─────┴─────────────────────────────┴─────────────┴────────────────────┘");

    // Show most confident prediction
    let (top_class, top_prob) = predictions[0];
    let confidence = if top_prob > 0.8 {
        "High"
    } else if top_prob > 0.5 {
        "Medium"
    } else {
        "Low"
    };

    let is_bird = &class_names[top_class] == "bird";
    let verdict = if is_bird {
        "🐦 This is a BIRD!"
    } else {
        "🌳 This is NOT a bird"
    };

    println!("\n{}", verdict);
    println!("🎯 Prediction: {} ({:.1}% - {} confidence)",
             class_names[top_class], top_prob * 100.0, confidence);

    if top_prob < 0.6 {
        println!("\n⚠️  Low confidence - the model is uncertain about this prediction.");
        println!("   This could mean:");
        println!("   • The image quality is poor or unclear");
        println!("   • The subject is partially obscured");
        println!("   • The model needs more training");
        println!("   • The image contains both birds and non-birds");
    } else if top_prob > 0.9 {
        println!("\n✅ High confidence - the model is very certain about this prediction!");
    }

    Ok(())
}
