import os
import glob

def check_dataset():
    """Quick check of dataset structure"""
    
    print("🔍 Quick Dataset Check")
    print("=" * 30)
    
    # Check all relevant directories
    dirs_to_check = [
        'DeepPCB/PCBData',
        'DeepPCB/all_images', 
        'DeepPCB/labels',
        'DeepPCB/images/train',
        'DeepPCB/images/val', 
        'DeepPCB/images/test',
        'DeepPCB/labels/train',
        'DeepPCB/labels/val',
        'DeepPCB/labels/test'
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✅ {dir_path}: {len(files)} files")
            
            # Show first few files as examples
            if len(files) > 0:
                examples = files[:3]
                for example in examples:
                    print(f"   - {example}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
        else:
            print(f"❌ {dir_path}: MISSING")
    
    # Check if we have any PCB groups
    if os.path.exists('DeepPCB/PCBData'):
        print(f"\n📁 PCBData Groups:")
        groups = [d for d in os.listdir('DeepPCB/PCBData') if d.startswith('group')]
        for group in groups[:5]:  # Show first 5 groups
            group_path = os.path.join('DeepPCB/PCBData', group)
            if os.path.isdir(group_path):
                subdirs = os.listdir(group_path)
                print(f"   {group}: {subdirs}")
    
    # Check dataset.yaml
    if os.path.exists('dataset.yaml'):
        print(f"\n📄 dataset.yaml exists")
        with open('dataset.yaml', 'r') as f:
            content = f.read()
            print("Content preview:")
            print(content[:200] + "..." if len(content) > 200 else content)
    else:
        print(f"\n❌ dataset.yaml: MISSING")

if __name__ == "__main__":
    check_dataset()