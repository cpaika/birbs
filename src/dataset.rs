use anyhow::Result;
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;

pub struct BirdDataset {
    pub image_paths: Vec<std::path::PathBuf>,
    pub labels: Vec<usize>,
    pub class_names: Vec<String>,
    pub num_classes: usize,
    pub use_augmentation: bool,
    pub image_size: u32,
}

impl BirdDataset {
    /// Downloads non-bird images (main entry point for training)
    pub fn download_non_bird_images(output_dir: &Path, count: usize) -> Result<()> {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::sync::{Arc, Mutex};
        use std::thread;

        println!("Downloading {} diverse non-bird images...", count);
        println!("Using Unsplash photos (via Picsum) for high-quality, diverse examples.\n");

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let downloaded = Arc::new(Mutex::new(0usize));
        let pb = ProgressBar::new(count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg} ({per_sec})")
                .unwrap(),
        );

        // Download in parallel using threads
        let num_threads = 10;
        let chunk_size = (count + num_threads - 1) / num_threads;
        let mut handles = vec![];

        let start_offset = std::fs::read_dir(output_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .count();

        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = ((thread_id + 1) * chunk_size).min(count);

            if start >= count {
                break;
            }

            let client = client.clone();
            let output_dir = output_dir.to_path_buf();
            let downloaded = Arc::clone(&downloaded);
            let pb = pb.clone();

            let handle = thread::spawn(move || {
                for i in start..end {
                    let url = format!("https://picsum.photos/256/256?random={}", start_offset + i + 1000);

                    match client.get(&url).send() {
                        Ok(response) if response.status().is_success() => {
                            if let Ok(bytes) = response.bytes() {
                                if image::load_from_memory(&bytes).is_ok() {
                                    let filepath = output_dir.join(format!("sample_{:06}.jpg", start_offset + i));
                                    if let Ok(mut file) = fs::File::create(&filepath) {
                                        use std::io::Write;
                                        if file.write_all(&bytes).is_ok() {
                                            let mut count = downloaded.lock().unwrap();
                                            *count += 1;
                                            pb.set_position(*count as u64);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    thread::sleep(std::time::Duration::from_millis(30));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        let final_count = *downloaded.lock().unwrap();
        pb.finish_with_message(format!("Downloaded {} images", final_count));

        Ok(())
    }

    /// Downloads non-bird images from HuggingFace dataset
    pub fn download_hf_dataset(
        output_dir: &Path,
        dataset_name: &str,
        max_images: usize,
    ) -> Result<()> {
        use hf_hub::api::sync::Api;

        println!("Downloading non-bird dataset from HuggingFace: {}", dataset_name);
        println!("This may take a while for large datasets...");

        fs::create_dir_all(output_dir)?;

        // Initialize HF API
        let api = Api::new()?;
        let repo = api.dataset(dataset_name.to_string());

        println!("Fetching dataset files...");

        // For the Honey-Data-15M dataset, we'll download parquet files
        // The dataset contains image URLs that we need to download
        let info = repo.info()?;
        let parquet_files: Vec<_> = info
            .siblings
            .iter()
            .filter(|f| f.rfilename.ends_with(".parquet"))
            .take(1) // Just use first parquet file for now
            .collect();

        if parquet_files.is_empty() {
            anyhow::bail!("No parquet files found in dataset");
        }

        println!("Found {} parquet files, processing first one...", parquet_files.len());

        for file_info in parquet_files {
            println!("Downloading: {}", file_info.rfilename);
            let file_path = repo.get(&file_info.rfilename)?;

            // Read parquet file and extract image URLs
            Self::process_parquet_file(&file_path, output_dir, max_images)?;
            break; // Only process first file for now
        }

        Ok(())
    }

    /// Process parquet file and download images
    fn process_parquet_file(
        parquet_path: &Path,
        output_dir: &Path,
        max_images: usize,
    ) -> Result<()> {
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let file = fs::File::open(parquet_path)?;
        let reader = SerializedFileReader::new(file)?;

        let pb = ProgressBar::new(max_images as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap(),
        );

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        // Get the schema to understand the structure
        let metadata = reader.metadata();
        let schema = metadata.file_metadata().schema_descr();

        println!("Parquet schema has {} columns", schema.num_columns());

        // Print column names to understand the structure
        println!("Columns:");
        for i in 0..schema.num_columns() {
            println!("  {}: {}", i, schema.column(i).name());
        }

        println!("\nExtracting image URLs from parquet file...");

        let mut downloaded = 0;
        let mut row_iter = reader.get_row_iter(None)?;

        // Process rows and extract image URLs
        while let Some(Ok(row)) = row_iter.next() {
            if downloaded >= max_images {
                break;
            }

            // Try to find the URL column (common names: "url", "image_url", "URL")
            let url = Self::extract_url_from_row(&row);

            if let Some(url_str) = url {
                match Self::download_image(&client, &url_str, output_dir, downloaded) {
                    Ok(true) => {
                        downloaded += 1;
                        if downloaded % 100 == 0 {
                            pb.set_position(downloaded as u64);
                            pb.set_message(format!("Downloaded {}/{}", downloaded, max_images));
                        }
                    }
                    Ok(false) => {
                        // Failed to download, skip
                    }
                    Err(_) => {
                        // Error downloading, skip
                    }
                }
            }
        }

        pb.finish_with_message(format!("Downloaded {} images from parquet", downloaded));

        if downloaded == 0 {
            println!("\n⚠️  No images were downloaded from the parquet file.");
            println!("The dataset structure doesn't match expected format (no URL columns found).");
            println!("\nFalling back to downloading sample images from Unsplash...\n");

            // Fallback: Download sample images from Picsum/Unsplash
            Self::download_sample_images(&client, output_dir, max_images)?;
        }

        Ok(())
    }

    /// Extract URL from a parquet row, filtering out bird-related images
    fn extract_url_from_row(row: &parquet::record::Row) -> Option<String> {
        // First, check if this row contains bird-related content
        if Self::contains_bird_keywords(row) {
            return None; // Skip bird images
        }

        // Try different common column names for image URLs
        let url_column_names = ["url", "URL", "image_url", "img_url", "link"];

        for col_name in &url_column_names {
            if let Some((_, field)) = row.get_column_iter().find(|(name, _)| name == col_name) {
                if let Some(url_str) = field.to_string().strip_prefix("\"").and_then(|s| s.strip_suffix("\"")) {
                    return Some(url_str.to_string());
                }
                return Some(field.to_string());
            }
        }

        None
    }

    /// Check if a parquet row contains bird-related keywords
    fn contains_bird_keywords(row: &parquet::record::Row) -> bool {
        let bird_keywords = [
            "bird", "birds", "avian", "eagle", "hawk", "sparrow", "robin",
            "crow", "raven", "owl", "parrot", "peacock", "penguin", "duck",
            "goose", "swan", "hummingbird", "woodpecker", "cardinal", "pigeon",
            "seagull", "pelican", "flamingo", "ostrich", "emu", "chicken",
            "rooster", "turkey", "pheasant", "quail", "dove", "finch",
        ];

        // Check all text fields in the row for bird keywords
        for (name, field) in row.get_column_iter() {
            // Check caption, description, text, or value fields
            if matches!(name.as_str(), "caption" | "text" | "description" | "value" | "label" | "category") {
                let field_str = field.to_string().to_lowercase();

                // Check if any bird keyword appears in the text
                for keyword in &bird_keywords {
                    if field_str.contains(keyword) {
                        return true; // This row contains bird-related content
                    }
                }
            }
        }

        false
    }

    /// Download sample non-bird images from public sources
    fn download_sample_images(
        client: &reqwest::blocking::Client,
        output_dir: &Path,
        count: usize,
    ) -> Result<()> {
        use std::sync::{Arc, Mutex};
        use std::thread;

        println!("Downloading {} sample non-bird images using Unsplash (via Picsum)...", count);
        println!("This provides diverse, high-quality photos perfect for non-bird examples.");

        let downloaded = Arc::new(Mutex::new(0usize));
        let pb = ProgressBar::new(count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap(),
        );

        // Download in parallel using threads
        let num_threads = 8;
        let chunk_size = (count + num_threads - 1) / num_threads;
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = ((thread_id + 1) * chunk_size).min(count);

            if start >= count {
                break;
            }

            let client = client.clone();
            let output_dir = output_dir.to_path_buf();
            let downloaded = Arc::clone(&downloaded);
            let pb = pb.clone();

            let handle = thread::spawn(move || {
                for i in start..end {
                    // Use Unsplash API via Picsum for diverse, high-quality images
                    let url = format!("https://picsum.photos/256/256?random={}", i);

                    match client.get(&url).send() {
                        Ok(response) if response.status().is_success() => {
                            if let Ok(bytes) = response.bytes() {
                                if image::load_from_memory(&bytes).is_ok() {
                                    let filepath = output_dir.join(format!("sample_{:06}.jpg", i));
                                    if let Ok(mut file) = fs::File::create(filepath) {
                                        use std::io::Write;
                                        let _ = file.write_all(&bytes);

                                        let mut count = downloaded.lock().unwrap();
                                        *count += 1;
                                        pb.set_position(*count as u64);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    // Small delay to avoid overwhelming the API
                    thread::sleep(std::time::Duration::from_millis(50));
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }

        let final_count = *downloaded.lock().unwrap();
        pb.finish_with_message(format!("Downloaded {} images", final_count));

        Ok(())
    }

    /// Download single image from URL
    fn download_image(
        client: &reqwest::blocking::Client,
        url: &str,
        output_dir: &Path,
        index: usize,
    ) -> Result<bool> {
        use std::io::Write;

        let response = client.get(url).send()?;

        if !response.status().is_success() {
            return Ok(false);
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        // Determine file extension
        let ext = if content_type.contains("jpeg") || content_type.contains("jpg") {
            "jpg"
        } else if content_type.contains("png") {
            "png"
        } else {
            "jpg" // default
        };

        let filename = format!("hf_image_{:06}.{}", index, ext);
        let filepath = output_dir.join(filename);

        let bytes = response.bytes()?;

        // Verify it's actually an image by trying to load it
        match image::load_from_memory(&bytes) {
            Ok(_) => {
                let mut file = fs::File::create(filepath)?;
                file.write_all(&bytes)?;
                Ok(true)
            }
            Err(_) => Ok(false), // Not a valid image, skip
        }
    }

    /// Recursively collect all image files from a directory
    fn collect_image_files_recursive(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Recursively search subdirectories
                Self::collect_image_files_recursive(&path, files)?;
            } else if path.is_file() {
                // Check if it's an image file
                if matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("jpg") | Some("jpeg") | Some("png")
                ) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    /// Loads bird images from a directory structure with class balancing
    /// Expected structure: data_dir/class_name/image.jpg
    /// Also supports nested subdirectories (e.g., data_dir/bird/species1/image.jpg)
    pub fn load_from_directory(data_dir: &Path, image_size: u32) -> Result<Self> {
        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();

        // Find all class directories
        let mut class_dirs: Vec<_> = fs::read_dir(data_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .collect();

        class_dirs.sort_by_key(|dir| dir.file_name());

        if class_dirs.is_empty() {
            anyhow::bail!(
                "No class directories found in {}. Please organize images by class/species.",
                data_dir.display()
            );
        }

        println!("Found {} classes", class_dirs.len());

        // First pass: count images per class to find the minimum (including subdirectories)
        let mut class_counts = Vec::new();
        for class_dir in &class_dirs {
            let mut image_files = Vec::new();
            Self::collect_image_files_recursive(&class_dir.path(), &mut image_files)?;
            class_counts.push(image_files.len());
        }

        // Find the minimum class size for balancing
        let min_class_size = *class_counts.iter().min().unwrap();

        println!("\n⚖️  Class Balance:");
        for (i, class_dir) in class_dirs.iter().enumerate() {
            let class_name = class_dir.file_name().to_string_lossy().to_string();
            println!("  {}: {} images", class_name, class_counts[i]);
        }
        println!("\n🎯 Balancing dataset: Using {} images per class (smallest class size)", min_class_size);
        println!("   This prevents the model from just predicting the majority class.\n");

        let pb = ProgressBar::new(class_dirs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap(),
        );

        // Second pass: load and validate images with balancing
        for (class_idx, class_dir) in class_dirs.iter().enumerate() {
            let class_name = class_dir.file_name().to_string_lossy().to_string();
            class_names.push(class_name.clone());
            pb.set_message(format!("Validating {}", class_name));

            // Load all image files for this class (recursively if there are subdirectories)
            let mut image_files = Vec::new();
            Self::collect_image_files_recursive(&class_dir.path(), &mut image_files)?;

            // Shuffle to get random selection
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            image_files.shuffle(&mut rng);

            // Take only min_class_size images to balance classes
            for image_path in image_files.iter().take(min_class_size) {
                image_paths.push(image_path.clone());
                labels.push(class_idx);
            }

            pb.inc(1);
        }

        pb.finish_with_message("Dataset paths loaded and balanced!");

        let num_classes = class_names.len();
        println!(
            "\n✅ Loaded {} balanced image paths across {} classes",
            image_paths.len(),
            num_classes
        );

        Ok(BirdDataset {
            image_paths,
            labels,
            class_names,
            num_classes,
            use_augmentation: false,
            image_size,
        })
    }

    /// Preprocesses an image: resize and normalize
    fn preprocess_image(img: DynamicImage, size: u32) -> DynamicImage {
        // Resize to fixed size while maintaining aspect ratio
        img.resize_exact(size, size, image::imageops::FilterType::Lanczos3)
    }

    /// Apply data augmentation to an image
    fn augment_image(img: DynamicImage) -> DynamicImage {
        use image::imageops;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut result = img;

        // 50% chance: horizontal flip
        if rng.gen_bool(0.5) {
            result = DynamicImage::ImageRgb8(imageops::flip_horizontal(&result.to_rgb8()));
        }

        // 30% chance: small rotation (-15 to +15 degrees)
        if rng.gen_bool(0.3) {
            let angle = rng.gen_range(-15.0..15.0);
            // Note: Basic rotation - more complex rotation would require additional dependencies
            // For now, we'll skip rotation to keep dependencies simple
        }

        // 40% chance: brightness adjustment (0.7 to 1.3)
        if rng.gen_bool(0.4) {
            let factor = rng.gen_range(0.7..1.3);
            let rgb = result.to_rgb8();
            let (width, height) = rgb.dimensions();
            let mut adjusted = image::RgbImage::new(width, height);

            for (x, y, pixel) in rgb.enumerate_pixels() {
                let r = (pixel[0] as f32 * factor).min(255.0).max(0.0) as u8;
                let g = (pixel[1] as f32 * factor).min(255.0).max(0.0) as u8;
                let b = (pixel[2] as f32 * factor).min(255.0).max(0.0) as u8;
                adjusted.put_pixel(x, y, image::Rgb([r, g, b]));
            }

            result = DynamicImage::ImageRgb8(adjusted);
        }

        // 30% chance: contrast adjustment
        if rng.gen_bool(0.3) {
            let contrast = rng.gen_range(0.8..1.2);
            result = DynamicImage::ImageRgb8(imageops::contrast(&result.to_rgb8(), contrast));
        }

        result
    }

    /// Loads and processes a single image by index
    /// Returns tensor data in CHW format, normalized to [0, 1]
    pub fn load_image(&self, index: usize) -> Result<Vec<f32>> {
        let img = image::open(&self.image_paths[index])?;
        let resized = Self::preprocess_image(img, self.image_size);

        // Apply augmentation if enabled (training only)
        let processed_img = if self.use_augmentation {
            Self::augment_image(resized)
        } else {
            resized
        };

        let rgb = processed_img.to_rgb8();
        let (width, height) = rgb.dimensions();

        // Convert to CHW format (Channels, Height, Width) and normalize to [0, 1]
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

    /// Splits dataset into train and test sets
    /// Training set will have augmentation enabled, test set will not
    pub fn train_test_split(&self, train_ratio: f32) -> Result<(BirdDataset, BirdDataset)> {
        let mut indices: Vec<usize> = (0..self.image_paths.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let train_size = (self.image_paths.len() as f32 * train_ratio) as usize;

        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];

        let train_paths = train_indices.iter().map(|&i| self.image_paths[i].clone()).collect();
        let train_labels = train_indices.iter().map(|&i| self.labels[i]).collect();

        let test_paths = test_indices.iter().map(|&i| self.image_paths[i].clone()).collect();
        let test_labels = test_indices.iter().map(|&i| self.labels[i]).collect();

        Ok((
            BirdDataset {
                image_paths: train_paths,
                labels: train_labels,
                class_names: self.class_names.clone(),
                num_classes: self.num_classes,
                use_augmentation: true, // Enable augmentation for training
                image_size: self.image_size,
            },
            BirdDataset {
                image_paths: test_paths,
                labels: test_labels,
                class_names: self.class_names.clone(),
                num_classes: self.num_classes,
                use_augmentation: false, // No augmentation for testing
                image_size: self.image_size,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    /// Helper function to create a small test dataset (max 2 classes, max 5 images per class)
    fn load_small_test_dataset() -> Result<BirdDataset> {
        let data_path = Path::new("./data/birds");

        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();

        // Find all class directories
        let mut class_dirs: Vec<_> = fs::read_dir(data_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .take(2) // Only take 2 classes for testing
            .collect();

        class_dirs.sort_by_key(|dir| dir.file_name());

        for (class_idx, class_dir) in class_dirs.iter().enumerate() {
            let class_name = class_dir.file_name().to_string_lossy().to_string();
            class_names.push(class_name);

            // Collect max 5 image paths from this class
            let image_files: Vec<_> = fs::read_dir(class_dir.path())?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path().is_file()
                        && matches!(
                            entry.path().extension().and_then(|s| s.to_str()),
                            Some("jpg") | Some("jpeg") | Some("png")
                        )
                })
                .take(5) // Only take 5 images per class
                .collect();

            for image_file in image_files {
                image_paths.push(image_file.path());
                labels.push(class_idx);
            }
        }

        let num_classes = class_names.len();

        Ok(BirdDataset {
            image_paths,
            labels,
            class_names,
            num_classes,
            use_augmentation: false,
            image_size: 64,
        })
    }

    #[test]
    fn test_load_dataset() {
        // Test loading a small dataset from the data/birds directory
        let data_path = Path::new("./data/birds");

        // Skip test if data directory doesn't exist
        if !data_path.exists() {
            println!("Skipping test: ./data/birds directory not found");
            return;
        }

        let result = load_small_test_dataset();

        assert!(result.is_ok(), "Failed to load dataset: {:?}", result.err());

        let dataset = result.unwrap();

        // Verify we loaded some image paths
        assert!(dataset.image_paths.len() > 0, "No image paths loaded");
        assert!(dataset.labels.len() > 0, "No labels loaded");
        assert_eq!(
            dataset.image_paths.len(),
            dataset.labels.len(),
            "Mismatch between image paths and labels count"
        );

        // Verify we have multiple classes
        assert!(dataset.num_classes > 0, "No classes found");
        assert_eq!(
            dataset.num_classes,
            dataset.class_names.len(),
            "Mismatch between num_classes and class_names length"
        );

        // Verify labels are within valid range
        for (i, &label) in dataset.labels.iter().enumerate() {
            assert!(
                label < dataset.num_classes,
                "Image {} has invalid label: {} >= {}",
                i,
                label,
                dataset.num_classes
            );
        }

        println!("Dataset loaded successfully:");
        println!("  Total image paths: {}", dataset.image_paths.len());
        println!("  Number of classes: {}", dataset.num_classes);
        println!("  Classes: {:?}", dataset.class_names);
    }

    #[test]
    fn test_train_test_split() {
        let data_path = Path::new("./data/birds");

        if !data_path.exists() {
            println!("Skipping test: ./data/birds directory not found");
            return;
        }

        let dataset = load_small_test_dataset()
            .expect("Failed to load dataset");

        let train_ratio = 0.8;
        let result = dataset.train_test_split(train_ratio);

        assert!(result.is_ok(), "Failed to split dataset: {:?}", result.err());

        let (train_data, test_data) = result.unwrap();

        // Verify split sizes are reasonable
        let total = train_data.image_paths.len() + test_data.image_paths.len();
        assert_eq!(total, dataset.image_paths.len(), "Lost images during split");

        let expected_train_size = (dataset.image_paths.len() as f32 * train_ratio) as usize;
        let actual_train_size = train_data.image_paths.len();

        // Allow for some variance due to rounding
        assert!(
            (actual_train_size as i32 - expected_train_size as i32).abs() <= 1,
            "Train split size {} not close to expected {}",
            actual_train_size,
            expected_train_size
        );

        // Verify both splits have the same class info
        assert_eq!(train_data.num_classes, dataset.num_classes);
        assert_eq!(test_data.num_classes, dataset.num_classes);
        assert_eq!(train_data.class_names, dataset.class_names);
        assert_eq!(test_data.class_names, dataset.class_names);

        println!("Train/test split successful:");
        println!("  Train: {} images", train_data.image_paths.len());
        println!("  Test: {} images", test_data.image_paths.len());
    }

    #[test]
    fn test_load_image() {
        let data_path = Path::new("./data/birds");

        if !data_path.exists() {
            println!("Skipping test: ./data/birds directory not found");
            return;
        }

        let dataset = load_small_test_dataset()
            .expect("Failed to load dataset");

        if dataset.image_paths.is_empty() {
            println!("Skipping test: No images in dataset");
            return;
        }

        // Test loading a single image
        let result = dataset.load_image(0);
        assert!(result.is_ok(), "Failed to load image: {:?}", result.err());

        let tensor = result.unwrap();

        // Verify tensor shape (CHW format)
        let image_size = dataset.image_size as usize;
        let expected_size = 3 * image_size * image_size;
        assert_eq!(
            tensor.len(),
            expected_size,
            "Tensor has incorrect size: {} != {}",
            tensor.len(),
            expected_size
        );

        // Verify values are normalized to [0, 1]
        for &val in &tensor {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Tensor value out of range [0, 1]: {}",
                val
            );
        }

        println!("Image loading successful:");
        println!("  Image size: {}x{}", image_size, image_size);
        println!("  Tensor size: {}", expected_size);
    }
}
