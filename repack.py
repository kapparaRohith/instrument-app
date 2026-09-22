import zipfile
import os

src_dir = "models/best_resnet.pth"
out_file = "models/best_resnet_repacked.pth"

def make_zip(prefix):
    with zipfile.ZipFile(out_file, 'w', zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, src_dir)
                arcname = os.path.join(prefix, rel_path) if prefix else rel_path
                zf.write(full_path, arcname)

# Try without a wrapper folder first
make_zip("")

import torch
try:
    torch.load(out_file, map_location='cpu')
    print("SUCCESS with no wrapper folder")
except Exception as e:
    print("Failed with no wrapper, trying 'archive/' wrapper instead...")
    make_zip("archive")
    try:
        torch.load(out_file, map_location='cpu')
        print("SUCCESS with 'archive/' wrapper")
    except Exception as e2:
        print("Both attempts failed:", e2)
