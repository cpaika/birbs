#!/bin/bash

# Script to set up datasets for binary bird classification (bird vs not-bird)
# Usage: ./setup_dataset.sh <path_to_nabirds.tar.gz> [path_to_trees.zip]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bird Binary Classifier Dataset Setup ===${NC}"
echo ""
echo "This will set up a binary classification dataset:"
echo "  - Class 0: birds (from NABirds dataset)"
echo "  - Class 1: not-birds (from trees/other dataset)"
echo ""

# Check if nabirds tar file path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No dataset paths provided${NC}"
    echo "Usage: ./setup_dataset.sh <path_to_nabirds.tar.gz> [path_to_trees.zip]"
    echo ""
    echo "Download datasets:"
    echo "  NABirds: https://dl.allaboutbirds.org/merlin---computer-vision--terms-of-use"
    echo "  Trees:   https://www.kaggle.com/datasets/mexwell/5m-trees-dataset"
    echo ""
    echo "Note: Trees dataset is optional. If not provided, only bird class will be set up."
    exit 1
fi

NABIRDS_TAR="$1"
TREES_ZIP="${2:-}"

# Check if nabirds file exists
if [ ! -f "$NABIRDS_TAR" ]; then
    echo -e "${RED}Error: NABirds file not found: $NABIRDS_TAR${NC}"
    exit 1
fi

# Check if trees file exists (optional)
if [ -n "$TREES_ZIP" ] && [ ! -f "$TREES_ZIP" ]; then
    echo -e "${YELLOW}Warning: Trees file not found: $TREES_ZIP${NC}"
    echo "Continuing with birds only..."
    TREES_ZIP=""
fi

# Create data directory with binary classification structure
echo -e "${GREEN}Step 1: Creating binary classification directory structure...${NC}"
mkdir -p ./data/birds/bird
mkdir -p ./data/birds/not-bird
echo "✓ Created ./data/birds/bird"
echo "✓ Created ./data/birds/not-bird"

# ============================================================
# STEP 2: Extract and setup bird images
# ============================================================
echo ""
echo -e "${GREEN}Step 2: Processing NABirds dataset...${NC}"
TEMP_BIRDS=$(mktemp -d)
echo "Extracting to: $TEMP_BIRDS"

tar -xzf "$NABIRDS_TAR" -C "$TEMP_BIRDS"
echo "✓ Extracted successfully"

# Find the images directory
BIRD_IMAGES_DIR=""
if [ -d "$TEMP_BIRDS/nabirds/images" ]; then
    BIRD_IMAGES_DIR="$TEMP_BIRDS/nabirds/images"
elif [ -d "$TEMP_BIRDS/images" ]; then
    BIRD_IMAGES_DIR="$TEMP_BIRDS/images"
else
    BIRD_IMAGES_DIR=$(find "$TEMP_BIRDS" -type d -name "images" | head -n 1)
fi

if [ -z "$BIRD_IMAGES_DIR" ] || [ ! -d "$BIRD_IMAGES_DIR" ]; then
    echo -e "${RED}Error: Could not find 'images' directory in NABirds tar${NC}"
    rm -rf "$TEMP_BIRDS"
    exit 1
fi

echo "Found bird images at: $BIRD_IMAGES_DIR"

# Count and copy bird species
TOTAL_BIRD_SPECIES=$(find "$BIRD_IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "Found $TOTAL_BIRD_SPECIES bird species"

COPIED=0
for species_dir in "$BIRD_IMAGES_DIR"/*; do
    if [ -d "$species_dir" ]; then
        cp -r "$species_dir" "./data/birds/bird/"
        COPIED=$((COPIED + 1))
        if [ $((COPIED % 50)) -eq 0 ]; then
            echo "  Copied $COPIED/$TOTAL_BIRD_SPECIES species..."
        fi
    fi
done

BIRD_IMAGES=$(find ./data/birds/bird -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
echo "✓ Copied $COPIED species directories"
echo "✓ Total bird images: $BIRD_IMAGES"

rm -rf "$TEMP_BIRDS"

# ============================================================
# STEP 3: Extract and setup non-bird images (trees)
# ============================================================
if [ -n "$TREES_ZIP" ]; then
    echo ""
    echo -e "${GREEN}Step 3: Processing trees dataset (non-birds)...${NC}"
    TEMP_TREES=$(mktemp -d)
    echo "Extracting to: $TEMP_TREES"

    unzip -q "$TREES_ZIP" -d "$TEMP_TREES"
    echo "✓ Extracted successfully"

    # Find tree images - the dataset structure may vary
    TREE_IMAGES_DIR=$(find "$TEMP_TREES" -type d -name "*tree*" -o -name "*Tree*" | head -n 1)

    if [ -z "$TREE_IMAGES_DIR" ]; then
        # If no tree-specific directory, use the root
        TREE_IMAGES_DIR="$TEMP_TREES"
    fi

    echo "Found tree images at: $TREE_IMAGES_DIR"

    # Copy tree images to not-bird directory
    # Limit to similar number as birds to balance dataset
    TARGET_COUNT=$((BIRD_IMAGES))
    COPIED_TREES=0

    find "$TREE_IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | while read -r img; do
        if [ $COPIED_TREES -lt $TARGET_COUNT ]; then
            cp "$img" "./data/birds/not-bird/"
            COPIED_TREES=$((COPIED_TREES + 1))

            if [ $((COPIED_TREES % 1000)) -eq 0 ]; then
                echo "  Copied $COPIED_TREES tree images..."
            fi
        fi
    done

    TREE_COUNT=$(find ./data/birds/not-bird -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
    echo "✓ Total non-bird images: $TREE_COUNT"

    rm -rf "$TEMP_TREES"
else
    echo ""
    echo -e "${YELLOW}Step 3: Skipping non-bird dataset (not provided)${NC}"
    echo "To add non-bird examples later, download:"
    echo "  https://www.kaggle.com/datasets/mexwell/5m-trees-dataset"
    echo "And run: ./setup_dataset.sh $NABIRDS_TAR <path_to_trees.zip>"
fi

# ============================================================
# STEP 4: Summary
# ============================================================
echo ""
echo -e "${GREEN}Step 4: Dataset Summary${NC}"
TOTAL_IMAGES=$(find ./data/birds -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
echo "✓ Total images: $TOTAL_IMAGES"
echo "  - Bird images: $BIRD_IMAGES"
if [ -n "$TREES_ZIP" ]; then
    echo "  - Not-bird images: $TREE_COUNT"
fi

# ============================================================
# STEP 5: Run tests
# ============================================================
echo ""
echo -e "${GREEN}Step 5: Testing dataset loader...${NC}"
cargo test --release -- --nocapture 2>&1 | grep -E "(test result:|Total images:|Number of classes:)" || true

# Check if tests passed
if cargo test --release --quiet 2>&1 | grep -q "test result: ok"; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${BLUE}=== Setup Complete ===${NC}"
    echo ""
    echo "Binary classification dataset is ready!"
    echo "  Class 0: bird"
    echo "  Class 1: not-bird"
    echo ""
    echo "To start training, run:"
    echo "  cargo run --release"
    echo ""
    echo "The model will learn to distinguish birds from non-birds."
else
    echo ""
    echo -e "${RED}Warning: Some tests may have failed. Check output above.${NC}"
fi
