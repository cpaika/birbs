use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fs;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let max_images = if args.len() > 1 {
        args[1].parse().unwrap_or(10000)
    } else {
        10000
    };

    println!("🖼️  Sample Image Downloader for Non-Bird Dataset\n");
    println!("Target images: {}\n", max_images);

    let output_dir = std::path::Path::new("./data/birds/not-bird");
    fs::create_dir_all(output_dir)?;

    println!("Downloading {} diverse images from Unsplash (via Picsum)...", max_images);
    println!("These provide high-quality, diverse photos perfect for non-bird examples.\n");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let downloaded = Arc::new(Mutex::new(0usize));
    let pb = ProgressBar::new(max_images as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg} ({per_sec})")
            .unwrap(),
    );

    // Download in parallel using threads
    let num_threads = 10;
    let chunk_size = (max_images + num_threads - 1) / num_threads;
    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let start = thread_id * chunk_size;
        let end = ((thread_id + 1) * chunk_size).min(max_images);

        if start >= max_images {
            break;
        }

        let client = client.clone();
        let output_dir = output_dir.to_path_buf();
        let downloaded = Arc::clone(&downloaded);
        let pb = pb.clone();

        let handle = thread::spawn(move || {
            for i in start..end {
                // Use Unsplash API via Picsum for diverse, high-quality images
                let url = format!("https://picsum.photos/256/256?random={}", i + 1000);

                match client.get(&url).send() {
                    Ok(response) if response.status().is_success() => {
                        if let Ok(bytes) = response.bytes() {
                            if image::load_from_memory(&bytes).is_ok() {
                                let filepath = output_dir.join(format!("sample_{:06}.jpg", i));
                                if let Ok(mut file) = fs::File::create(&filepath) {
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

                // Small delay to avoid overwhelming the API
                thread::sleep(std::time::Duration::from_millis(30));
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        let _ = handle.join();
    }

    let final_count = *downloaded.lock().unwrap();
    pb.finish_with_message(format!("✅ Downloaded {} images", final_count));

    println!("\n✅ Download complete!");
    println!("Downloaded {} images to {:?}", final_count, output_dir);
    println!("\nYou can now run training:");
    println!("  cargo run --release");

    Ok(())
}
