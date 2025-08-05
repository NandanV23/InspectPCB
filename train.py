import torch
import os
import subprocess
import sys
import yaml
from pathlib import Path

def install_yolov5():
    """Install YOLOv5 repository if not present"""
    if not os.path.exists('yolov5'):
        print("📥 Cloning YOLOv5 repository...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], check=True)
            print("✅ YOLOv5 repository cloned successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to clone YOLOv5. Please install git or clone manually:")
            print("git clone https://github.com/ultralytics/yolov5.git")
            return False
    
    # Install YOLOv5 requirements
    requirements_file = 'yolov5/requirements.txt'
    if os.path.exists(requirements_file):
        print("📦 Installing YOLOv5 requirements...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
            print("✅ YOLOv5 requirements installed")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Could not install all YOLOv5 requirements")
    
    return True

def create_dataset_yaml():
    """Create dataset configuration file for training"""
    dataset_config = {
        'path': str(Path.cwd() / 'DeepPCB'),  # Absolute path to dataset
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 6,  # number of classes
        'names': {
            0: 'open',
            1: 'short', 
            2: 'mousebite',
            3: 'spur',
            4: 'pin-hole',
            5: 'spurious_copper'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("📄 Created dataset.yaml configuration file")
    return 'dataset.yaml'

def check_dataset_structure():
    """Check if dataset is properly prepared"""
    required_dirs = [
        'DeepPCB/images/train',
        'DeepPCB/images/val', 
        'DeepPCB/images/test',
        'DeepPCB/labels/train',
        'DeepPCB/labels/val',
        'DeepPCB/labels/test'
    ]
    
    missing_dirs = []
    empty_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        elif len(os.listdir(dir_path)) == 0:
            empty_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    if empty_dirs:
        print("⚠️  Empty directories:")
        for dir_path in empty_dirs:
            print(f"   - {dir_path}")
    
    # Count files in each directory
    print("📊 Dataset structure:")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"   {dir_path}: {file_count} files")
    
    return True

def train_yolov5():
    """Train YOLOv5 model using command line interface"""
    
    print("🚀 Starting YOLOv5 training...")
    
    # Training parameters
    train_args = [
        sys.executable, 'yolov5/train.py',
        '--img', '640',
        '--batch', '16', 
        '--epochs', '100',
        '--data', 'dataset.yaml',
        '--weights', 'yolov5s.pt',
        '--name', 'pcb_defect_detection',
        '--save-period', '10'
    ]
    
    # Add device parameter
    if torch.cuda.is_available():
        train_args.extend(['--device', '0'])
        print("🎮 Using GPU for training")
    else:
        train_args.extend(['--device', 'cpu'])
        print("💻 Using CPU for training (this will be slower)")
    
    print("🔧 Training command:")
    print(" ".join(train_args))
    print("\n" + "="*60)
    
    try:
        # Run training
        subprocess.run(train_args, check=True)
        
        # Copy the best model to the main directory
        best_model_path = 'yolov5/runs/train/pcb_defect_detection/weights/best.pt'
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, 'pcb_defect.pt')
            print("✅ Model saved as 'pcb_defect.pt'")
        
        print("\n🎉 Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False

def main():
    print("🏋️  YOLOv5 PCB Defect Detection Training")
    print("=" * 50)
    
    # Step 1: Install YOLOv5 if needed
    if not install_yolov5():
        return
    
    # Step 2: Check dataset structure
    if not check_dataset_structure():
        print("\n❌ Dataset not properly prepared!")
        print("Please run the following commands first:")
        print("1. python convert_deeppcb_correct.py")
        print("2. python prepare_data_updated.py")
        return
    
    # Step 3: Create dataset configuration
    dataset_yaml = create_dataset_yaml()
    
    # Step 4: Start training
    if train_yolov5():
        print("\n🎯 Training Summary:")
        print("- Model trained on PCB defect detection")
        print("- Best model saved as 'pcb_defect.pt'")
        print("- Training logs in 'yolov5/runs/train/pcb_defect_detection/'")
        print("\n🚀 Next step: Run 'streamlit run improved_app.py' to test the model!")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough disk space")
        print("2. Try reducing batch size if you get memory errors")
        print("3. Check that all dataset files are present")

if __name__ == "__main__":
    main()