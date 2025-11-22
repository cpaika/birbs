use anyhow::Result;
use std::env;

use imgs::dataset::BirdDataset;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let dataset_name = if args.len() > 1 {
        &args[1]
    } else {
        "Open-Bee/Honey-Data-15M"
    };

    let max_images = if args.len() > 2 {
        args[2].parse().unwrap_or(10000)
    } else {
        10000 // Sufficient for binary classification
    };

    println!("🤖 HuggingFace Dataset Downloader\n");
    println!("Dataset: {}", dataset_name);
    println!("Target images: {}\n", max_images);

    let output_dir = std::path::Path::new("./data/birds/not-bird");

    println!("Downloading to: {:?}\n", output_dir);

    match BirdDataset::download_hf_dataset(output_dir, dataset_name, max_images) {
        Ok(()) => {
            println!("\n✅ Download complete!");

            // Count downloaded images
            let count = std::fs::read_dir(output_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_file()
                        && matches!(
                            e.path().extension().and_then(|s| s.to_str()),
                            Some("jpg") | Some("jpeg") | Some("png")
                        )
                })
                .count();

            println!("Total images downloaded: {}", count);
            println!("\nYou can now run training:");
            println!("  cargo run --release");
        }
        Err(e) => {
            std::process::exit(1);
        }
    }

    Ok(())
}
