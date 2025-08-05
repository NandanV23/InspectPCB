import os
import glob
import shutil
import random
from PIL import Image
from pathlib import Path

def step1_analyze_raw_data():
    """Analyze the raw DeepPCB data structure"""
    print("🔍 STEP 1: Analyzing raw DeepPCB data...")
    
    if not os.path.exists('DeepPCB/PCBData'):
        print("❌ DeepPCB/PCBData not found!")
        print("Please make sure you have:")
        print("1. Downloaded the DeepPCB dataset")
        print("2. Extracted it to your project directory")
        return False
    
    # Find all groups
    groups = [d for d in os.listdir('DeepPCB/PCBData') if d.startswith('group')]
    print(f"📁 Found {len(groups)} groups: {groups[:3]}...")
    
    # Analyze first group structure
    if groups:
        first_group = os.path.join('DeepPCB/PCBData', groups[0])
        subdirs = os.listdir(first_group)
        print(f"📂 Sample group structure ({groups[0]}): {subdirs}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(first_group, subdir)
            if os.path.isdir(subdir_path):
                files = os.listdir(subdir_path)
                print(f"   {subdir}: {len(files)} files")
                if files:
                    # Show file types
                    extensions = set(os.path.splitext(f)[1].lower() for f in files[:10])
                    print(f"      Extensions: {extensions}")
    
    return len(groups) > 0

def step2_collect_and_convert():
    """Collect all images and convert annotations"""
    print("\n🔄 STEP 2: Collecting and converting data...")
    
    # Create output directories
    os.makedirs('DeepPCB/unified_images', exist_ok=True)
    os.makedirs('DeepPCB/unified_labels', exist_ok=True)
    
    groups = [d for d in os.listdir('DeepPCB/PCBData') if d.startswith('group')]
    total_images = 0
    total_labels = 0
    
    for group in groups:
        print(f"📁 Processing {group}...")
        group_path = os.path.join('DeepPCB/PCBData', group)
        
        # Find subdirectories
        subdirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        
        image_dir = None
        annotation_dir = None
        
        for subdir in subdirs:
            if subdir.endswith('_not'):
                annotation_dir = os.path.join(group_path, subdir)
            else:
                image_dir = os.path.join(group_path, subdir)
        
        if not image_dir or not annotation_dir:
            print(f"   ⚠️ Skipping {group} - missing directories")
            continue
        
        # Process all annotation files
        annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))
        print(f"   📄 Found {len(annotation_files)} annotation files")
        
        for ann_file in annotation_files:
            base_name = os.path.splitext(os.path.basename(ann_file))[0]
            
            # Look for test image
            test_image_patterns = [
                os.path.join(image_dir, f"{base_name}_test.jpg"),
                os.path.join(image_dir, f"{base_name}_test.png"),
                os.path.join(image_dir, f"{base_name}.jpg"), 
                os.path.join(image_dir, f"{base_name}.png")
            ]
            
            found_image = None
            for pattern in test_image_patterns:
                if os.path.exists(pattern):
                    found_image = pattern
                    break
            
            if not found_image:
                continue
            
            # Copy image with unique name
            unique_name = f"{group}_{base_name}"
            dest_image = os.path.join('DeepPCB/unified_images', f"{unique_name}.jpg")
            dest_label = os.path.join('DeepPCB/unified_labels', f"{unique_name}.txt")
            
            # Copy image
            shutil.copy2(found_image, dest_image)
            
            # Convert annotation
            if convert_annotation(ann_file, found_image, dest_label):
                total_images += 1
                total_labels += 1
    
    print(f"✅ Collected {total_images} images and {total_labels} labels")
    return total_images > 0

def convert_annotation(ann_file, img_file, output_file):
    """Convert single annotation to YOLO format"""
    try:
        img = Image.open(img_file)
        w, h = img.size
        
        with open(ann_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # Try different formats
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                
                if len(parts) >= 5:
                    try:
                        x1, y1, x2, y2, class_id = map(int, parts[:5])
                        
                        # Convert to YOLO format
                        center_x = ((x1 + x2) / 2) / w
                        center_y = ((y1 + y2) / 2) / h
                        bbox_width = (x2 - x1) / w
                        bbox_height = (y2 - y1) / h
                        
                        # Adjust class ID (DeepPCB uses 1-6, YOLO needs 0-5)
                        if class_id > 0:
                            class_id -= 1
                        
                        f_out.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    except ValueError:
                        continue
        return True
    except Exception as e:
        print(f"   ❌ Error converting {ann_file}: {e}")
        return False

def step3_split_dataset():
    """Split unified dataset into train/val/test"""
    print("\n📊 STEP 3: Splitting dataset...")
    
    # Create directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(f'DeepPCB/images/{split}', exist_ok=True)
        os.makedirs(f'DeepPCB/labels/{split}', exist_ok=True)
    
    # Get all unified files
    image_files = glob.glob('DeepPCB/unified_images/*.jpg')
    
    if not image_files:
        print("❌ No unified images found!")
        return False
    
    print(f"📷 Found {len(image_files)} images to split")
    
    # Create matched pairs
    valid_pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = f'DeepPCB/unified_labels/{base_name}.txt'
        
        if os.path.exists(label_file):
            valid_pairs.append((img_file, label_file))
    
    print(f"📋 Found {len(valid_pairs)} valid image-label pairs")
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    
    train_end = int(len(valid_pairs) * 0.7)
    val_end = int(len(valid_pairs) * 0.9)
    
    splits_data = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Copy files
    for split_name, pairs in splits_data.items():
        print(f"📁 {split_name}: {len(pairs)} pairs")
        
        for img_path, label_path in pairs:
            # Copy image
            img_dest = os.path.join(f'DeepPCB/images/{split_name}', os.path.basename(img_path))
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            label_dest = os.path.join(f'DeepPCB/labels/{split_name}', os.path.basename(label_path))
            shutil.copy2(label_path, label_dest)
    
    return True

def step4_create_yaml():
    """Create dataset.yaml file"""
    print("\n📄 STEP 4: Creating dataset.yaml...")
    
    yaml_content = f"""# DeepPCB Dataset Configuration
path: {str(Path.cwd() / 'DeepPCB')}
train: images/train
val: images/val  
test: images/test

# Classes
nc: 6
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: pin-hole
  5: spurious_copper
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✅ Created dataset.yaml")

def step5_verify():
    """Verify everything is ready"""
    print("\n✅ STEP 5: Final verification...")
    
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
            count = len(os.listdir(dir_path))
            print(f"✅ {dir_path}: {count} files")
            if count == 0:
                all_good = False
        else:
            print(f"❌ {dir_path}: Missing!")
            all_good = False
    
    if os.path.exists('dataset.yaml'):
        print("✅ dataset.yaml: Present")
    else:
        print("❌ dataset.yaml: Missing!")
        all_good = False
    
    return all_good

def main():
    print("🚀 Complete DeepPCB Dataset Setup")
    print("=" * 50)
    
    if not step1_analyze_raw_data():
        return
    
    if not step2_collect_and_convert():
        return
    
    if not step3_split_dataset():
        return
    
    step4_create_yaml()
    
    if step5_verify():
        print("\n🎉 Dataset setup completed successfully!")
        print("\n🚀 Ready for training! Run:")
        print("python train_fixed.py")
    else:
        print("\n❌ Setup verification failed!")

if __name__ == "__main__":
    main()