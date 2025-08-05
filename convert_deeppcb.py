import os
import glob
import shutil
from PIL import Image
from pathlib import Path

def collect_deeppcb_dataset():
    """Collect all DeepPCB test images and annotations correctly"""
    
    print("🔄 Collecting DeepPCB dataset (test images only)...")
    
    # Create unified directories
    unified_image_dir = "DeepPCB/all_images"
    unified_labels_dir = "DeepPCB/labels"
    
    os.makedirs(unified_image_dir, exist_ok=True)
    os.makedirs(unified_labels_dir, exist_ok=True)
    
    # Find all group directories in PCBData
    pcb_data_dir = "DeepPCB/PCBData"
    if not os.path.exists(pcb_data_dir):
        print("❌ PCBData directory not found!")
        return False
    
    group_dirs = [d for d in os.listdir(pcb_data_dir) 
                  if d.startswith('group') and os.path.isdir(os.path.join(pcb_data_dir, d))]
    
    total_processed = 0
    total_converted = 0
    
    for group_dir in group_dirs:
        group_path = os.path.join(pcb_data_dir, group_dir)
        print(f"📁 Processing {group_dir}...")
        
        # Look for the image directory (without _not suffix)
        subdirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        
        image_subdir = None
        annotation_subdir = None
        
        for subdir in subdirs:
            if subdir.endswith('_not'):
                annotation_subdir = os.path.join(group_path, subdir)
            else:
                image_subdir = os.path.join(group_path, subdir)
        
        if not image_subdir or not annotation_subdir:
            print(f"  ⚠️  Skipping {group_dir} - missing image or annotation directory")
            continue
        
        # Get all annotation files
        annotation_files = glob.glob(os.path.join(annotation_subdir, '*.txt'))
        
        for ann_file in annotation_files:
            base_name = os.path.splitext(os.path.basename(ann_file))[0]
            
            # Look for corresponding test image (the key insight!)
            test_image_name = f"{base_name}_test.jpg"
            test_image_path = os.path.join(image_subdir, test_image_name)
            
            if not os.path.exists(test_image_path):
                # Try other possible formats
                for ext in ['.png', '.jpeg']:
                    alt_path = os.path.join(image_subdir, f"{base_name}_test{ext}")
                    if os.path.exists(alt_path):
                        test_image_path = alt_path
                        break
                else:
                    print(f"  ⚠️  No test image found for {base_name}")
                    continue
            
            # Copy the test image with a unique name
            unique_image_name = f"{group_dir}_{os.path.basename(test_image_path)}"
            dest_image_path = os.path.join(unified_image_dir, unique_image_name)
            shutil.copy2(test_image_path, dest_image_path)
            
            # Convert the annotation
            unique_label_name = f"{group_dir}_{base_name}.txt"
            dest_label_path = os.path.join(unified_labels_dir, unique_label_name)
            
            if convert_annotation_file(ann_file, test_image_path, dest_label_path):
                total_converted += 1
            
            total_processed += 1
    
    print(f"✅ Processed {total_processed} files, converted {total_converted} successfully")
    return total_converted > 0

def convert_annotation_file(annotation_file, image_file, output_file):
    """Convert a single annotation file to YOLO format"""
    
    try:
        # Get image dimensions
        img = Image.open(image_file)
        w, h = img.size
        
        # Read and convert annotations
        with open(annotation_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse DeepPCB format: x1,y1,x2,y2,class_id
                    # or space-separated: x1 y1 x2 y2 class_id
                    if ',' in line:
                        parts = line.split(',')
                    else:
                        parts = line.split()
                    
                    if len(parts) >= 5:
                        x1, y1, x2, y2, class_id = map(int, parts[:5])
                        
                        # Convert to YOLO format (normalized center coordinates)
                        center_x = ((x1 + x2) / 2) / w
                        center_y = ((y1 + y2) / 2) / h
                        bbox_width = (x2 - x1) / w
                        bbox_height = (y2 - y1) / h
                        
                        # DeepPCB uses 1-6 class IDs, convert to 0-5 for YOLO
                        if class_id > 0:
                            class_id -= 1
                        
                        f_out.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                        
                except (ValueError, IndexError) as e:
                    print(f"    ⚠️  Error parsing line '{line}': {e}")
                    continue
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error processing {annotation_file}: {e}")
        return False

def verify_conversion():
    """Verify that the conversion worked correctly"""
    
    print("\n🔍 Verifying conversion...")
    
    image_dir = "DeepPCB/all_images"
    label_dir = "DeepPCB/labels"
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("❌ Conversion directories not found!")
        return False
    
    image_files = glob.glob(os.path.join(image_dir, '*'))
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    print(f"📊 Conversion results:")
    print(f"   🖼️  Images: {len(image_files)}")
    print(f"   🏷️  Labels: {len(label_files)}")
    
    # Check some label files
    sample_labels = label_files[:3]
    for label_file in sample_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            print(f"   📄 {os.path.basename(label_file)}: {len(lines)} defects")
            if lines:
                # Show first annotation as example
                first_line = lines[0].strip()
                parts = first_line.split()
                if len(parts) >= 5:
                    class_id, cx, cy, w, h = parts[:5]
                    print(f"      Example: class={class_id}, center=({cx}, {cy}), size=({w}, {h})")
    
    return len(image_files) > 0 and len(label_files) > 0

def create_class_mapping():
    """Create a file showing the class mapping"""
    
    class_mapping = {
        0: "open",
        1: "short", 
        2: "mousebite",
        3: "spur",
        4: "pin-hole",
        5: "spurious_copper"
    }
    
    with open("DeepPCB/class_mapping.txt", "w") as f:
        f.write("DeepPCB Class Mapping (YOLO format):\n")
        f.write("=====================================\n")
        for class_id, class_name in class_mapping.items():
            f.write(f"{class_id}: {class_name}\n")
    
    print("📄 Created class_mapping.txt")

def main():
    print("🚀 DeepPCB Dataset Converter (Correct Version)")
    print("=" * 60)
    
    # Collect and convert the dataset
    if collect_deeppcb_dataset():
        # Verify the conversion
        if verify_conversion():
            # Create class mapping file
            create_class_mapping()
            
            print("\n🎉 Dataset conversion completed successfully!")
            print("\n📋 Summary:")
            print("   - Only '_test.jpg' images are used (not template images)")
            print("   - Annotations converted from DeepPCB to YOLO format")
            print("   - Class IDs converted from 1-6 to 0-5")
            print("   - All files unified in DeepPCB/all_images/ and DeepPCB/labels/")
            print("\n🚀 Next step: Run 'python prepare_data_updated.py'")
        else:
            print("\n❌ Conversion verification failed!")
    else:
        print("\n❌ Dataset conversion failed!")
        print("\nTroubleshooting:")
        print("1. Make sure DeepPCB/PCBData/ directory exists")
        print("2. Check that group directories contain both image and annotation subdirectories")
        print("3. Verify that '_test.jpg' images exist in the image directories")

if __name__ == "__main__":
    main()