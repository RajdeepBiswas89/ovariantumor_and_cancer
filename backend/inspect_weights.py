import zipfile
import h5py
import os

zip_path = 'C:/Users/RAJDEEP/AppData/Local/Temp/tmpxwkjwyaa.keras'
extract_dir = 'temp_extract'
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extract('model.weights.h5', extract_dir)

weights_path = os.path.join(extract_dir, 'model.weights.h5')

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape}")
    else:
        print(name)

with h5py.File(weights_path, 'r') as f:
    print("Keys in weights file:")
    f.visititems(print_structure)
