import os
import shutil
import random
import glob
from pathlib import Path

def split_unified_dataset(image_dir='DeepPCB/all_images', labels_dir='DeepPCB/labels', 
                         train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split the unified dataset into train, validation, and test sets
    """
    
    print("🔄 Splitting unified dataset...")
    
    # Create output directories
    output_dirs = {
        'train': {
            'images': 'DeepPCB/images/train',
            'labels': 'DeepPCB/labels/train'
        },
        'val': {
            'images': 'DeepPCB/images/val', 
            'labels': 'DeepPCB/labels/val'
        },
        'test': {
            'images': 'DeepPCB/images/test',
            'labels': 'DeepPCB/labels/test'
        }
    }
    
    # Create directories
    for split_dirs in output_dirs.values():
        for dir_path in split_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Get all label files (these determine which images we have)
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    if not label_files:
        print("❌ No label files found! Make sure you've run the conversion script first.")
        return False
    
    # Create pairs of (image_file, label_file) that actually exist
    valid_pairs = []
    
    for label_file in label_files:
        label_basename = os.path.splitext(os.path.basename(label_file))[0]
        
        # Find corresponding image file
        possible_image_names = []
        
        # Try different image extensions and naming patterns
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_image_names.extend([
                os.path.join(image_dir, label_basename + ext),
                os.path.join(image_dir, label_basename.replace('group', '') + ext),
            ])
        
        # Find which image file actually exists
        found_image = None
        for possible_image in possible_image_names:
            if os.path.exists(possible_image):
                found_image = possible_image
                break
        
        if found_image:
            valid_pairs.append((found_image, label_file))
        else:
            # Try to find any image that might match
            all_images = glob.glob(os.path.join(image_dir, '*'))
            for img_path in all_images:
                img_basename = os.path.splitext(os.path.basename(img_path))[0]
                if any(part in img_basename for part in label_basename.split('_') if len(part) > 3):
                    valid_pairs.append((img_path, label_file))
                    break
    
    print(f"📊 Found {len(valid_pairs)} valid image-label pairs")
    
    if not valid_pairs:
        print("❌ No valid image-label pairs found!")
        return False
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_pairs = len(valid_pairs)
    train_end = int(total_pairs * train_ratio)
    val_end = int(total_pairs * (train_ratio + val_ratio))
    
    # Split pairs
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Copy files to respective directories
    for split_name, pairs in splits.items():
        print(f"📁 Processing {split_name} split: {len(pairs)} pairs")
        
        for image_path, label_path in pairs:
            # Copy image
            image_filename = os.path.basename(image_path)
            dst_image = os.path.join(output_dirs[split_name]['images'], image_filename)
            shutil.copy2(image_path, dst_image)
            
            # Copy label
            label_filename = os.path.basename(label_path)
            dst_label = os.path.join(output_dirs[split_name]['labels'], label_filename)
            shutil.copy2(label_path, dst_label)
    
    print("\n✅ Dataset split completed!")
    print(f"📊 Final split:")
    print(f"   🏋️  Train: {len(splits['train'])} pairs")
    print(f"   🔍 Validation: {len(splits['val'])} pairs") 
    print(f"   🧪 Test: {len(splits['test'])} pairs")
    
    return True

def verify_dataset_structure():
    """Verify that the dataset is properly structured for training"""
    
    print("\n🔍 Verifying dataset structure...")
    
    required_dirs = [
        'DeepPCB/images/train',
        'DeepPCB/images/val',
        'DeepPCB/images/test',
        'DeepPCB/labels/train',
        'DeepPCB/labels/val',
        'DeepPCB/labels/test'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"  ✅ {dir_path}: {file_count} files")
            if file_count == 0:
                print(f"     ⚠️  Warning: Directory is empty!")
                all_good = False
        else:
            print(f"  ❌ {dir_path}: Missing!")
            all_good = False
    
    if all_good:
        print("\n🎉 Dataset structure is ready for training!")
    else:
        print("\n⚠️  There are issues with the dataset structure.")
    
    return all_good

def create_dataset_yaml():
    """Create the dataset.yaml file for YOLO training"""
    
    dataset_config = f"""# DeepPCB Dataset Configuration
path: ./DeepPCB  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
nc: 6  # number of classes
names: ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(dataset_config)
    
    print("📄 Created dataset.yaml configuration file")

def main():
    print("📊 DeepPCB Dataset Preparation")
    print("=" * 50)
    
    # Check if conversion has been done
    if not os.path.exists('DeepPCB/labels') or not os.path.exists('DeepPCB/all_images'):
        print("❌ Unified dataset not found!")
        print("Please run 'python convert_deeppcb_dataset.py' first to convert the dataset.")
        return
    
    # Split the dataset
    if split_unified_dataset():
        # Verify the structure
        verify_dataset_structure()
        
        # Create YAML config
        create_dataset_yaml()
        
        print("\n🚀 Dataset is ready for training!")
        print("Next step: Run 'python train.py' to start training")
    else:
        print("\n❌ Dataset preparation failed!")

if __name__ == "__main__":
    main()