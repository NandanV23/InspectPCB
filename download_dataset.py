import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {command}: {e}")
        return False

def download_deeppcb_dataset():
    """Download and setup the DeepPCB dataset"""
    
    print("🔄 Starting DeepPCB dataset download...")
    
    # Check if git is available
    if not run_command("git --version"):
        print("❌ Git is not installed or not available in PATH")
        print("Please install Git first: https://git-scm.com/downloads")
        return False
    
    # Remove existing DeepPCB directory if it exists
    if os.path.exists("DeepPCB"):
        print("🗑️  Removing existing DeepPCB directory...")
        shutil.rmtree("DeepPCB")
    
    # Clone the repository
    print("📥 Cloning DeepPCB repository...")
    clone_command = "git clone https://github.com/tangsanli5201/DeepPCB.git"
    
    if not run_command(clone_command):
        print("❌ Failed to clone repository")
        return False
    
    # Check if the required directories exist
    required_dirs = ["DeepPCB/PCBData"]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠️  Directory {dir_path} not found after cloning")
    
    # The dataset structure might be different, let's explore it
    print("\n📂 Exploring downloaded dataset structure...")
    deeppcb_path = Path("DeepPCB")
    
    if deeppcb_path.exists():
        print("DeepPCB directory contents:")
        for item in deeppcb_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # Show contents of subdirectories
                try:
                    sub_items = list(item.iterdir())[:5]  # Show first 5 items
                    for sub_item in sub_items:
                        print(f"    - {sub_item.name}")
                    if len(list(item.iterdir())) > 5:
                        print(f"    ... and {len(list(item.iterdir())) - 5} more items")
                except:
                    pass
            else:
                print(f"  📄 {item.name}")
    
    # Check for common dataset directory structures
    possible_data_dirs = [
        "DeepPCB/PCBData",
        "DeepPCB/dataset",
        "DeepPCB/data",
        "DeepPCB/images",
        "DeepPCB/test"
    ]
    
    found_data_dir = None
    for data_dir in possible_data_dirs:
        if os.path.exists(data_dir):
            found_data_dir = data_dir
            print(f"✅ Found data directory: {data_dir}")
            break
    
    if found_data_dir:
        # List contents of the data directory
        print(f"\n📋 Contents of {found_data_dir}:")
        data_path = Path(found_data_dir)
        for item in data_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/ ({len(list(item.iterdir()))} items)")
            else:
                print(f"  📄 {item.name}")
    
    print("\n✅ DeepPCB repository downloaded successfully!")
    print("\n📝 Next steps:")
    print("1. Check the directory structure above")
    print("2. You may need to adjust the paths in convert_labels.py")
    print("3. Look for 'image' and 'gt' (ground truth) directories")
    print("4. Run the conversion script once you've located the correct paths")
    
    return True

def create_directory_structure():
    """Create the expected directory structure if it doesn't exist"""
    directories = [
        "DeepPCB/image",
        "DeepPCB/gt",
        "DeepPCB/labels",
        "DeepPCB/images/train",
        "DeepPCB/images/val",
        "DeepPCB/images/test",
        "DeepPCB/labels/train",
        "DeepPCB/labels/val",
        "DeepPCB/labels/test"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Created directory structure")

if __name__ == "__main__":
    print("🚀 DeepPCB Dataset Setup")
    print("=" * 50)
    
    if download_deeppcb_dataset():
        create_directory_structure()
        print("\n🎉 Setup completed!")
        print("\nIf the dataset structure is different than expected:")
        print("1. Check the actual directory structure printed above")
        print("2. Update the paths in convert_labels.py accordingly")
        print("3. The dataset might be in DeepPCB/PCBData/ or another subdirectory")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        
        # Provide alternative download methods
        print("\n🔄 Alternative download methods:")
        print("1. Manual download:")
        print("   - Go to: https://github.com/tangsanli5201/DeepPCB")
        print("   - Click 'Code' > 'Download ZIP'")
        print("   - Extract to your project directory")
        print("\n2. Direct git clone:")
        print("   git clone https://github.com/tangsanli5201/DeepPCB.git")