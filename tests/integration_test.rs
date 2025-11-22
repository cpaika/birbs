use birbs::dataset::BirdDataset;

#[test]
fn test_augmentation_flag() {
    // Test that augmentation flag is properly set after train/test split
    let data_path = std::path::Path::new("./data/birds");

    if !data_path.exists() {
        println!("Skipping test: dataset not found");
        return;
    }

    let dataset = match BirdDataset::load_from_directory(data_path, 64) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: failed to load dataset");
            return;
        }
    };

    if dataset.images.len() < 10 {
        println!("Skipping test: not enough images");
        return;
    }

    let (train_data, test_data) = dataset.train_test_split(0.8).unwrap();

    // Training data should have augmentation enabled
    assert_eq!(train_data.use_augmentation, true, "Training set should have augmentation enabled");

    // Test data should not have augmentation
    assert_eq!(test_data.use_augmentation, false, "Test set should not have augmentation");

    println!("✓ Augmentation flags set correctly");
}
