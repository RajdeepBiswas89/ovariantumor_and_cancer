#!/usr/bin/env python3
"""
Debug script to examine the bytes format
"""
import pickle
import sys

# Keras 3 compatibility fixes
if 'keras.saving.pickle_utils' not in sys.modules:
    import types
    fake_module = types.ModuleType('keras.saving.pickle_utils')
    fake_module.deserialize_model_from_bytecode = lambda x: x
    sys.modules['keras.saving.pickle_utils'] = fake_module

def examine_bytes():
    """Examine the bytes format"""
    
    print("Examining bytes format...")
    
    try:
        with open('fused_model.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Check ResNet50 classifier bytes
        r50_classifier_bytes = data['resnet50_classifier']
        print(f"ResNet50 classifier bytes length: {len(r50_classifier_bytes)}")
        print(f"First 50 bytes: {r50_classifier_bytes[:50]}")
        print(f"First 50 bytes as hex: {r50_classifier_bytes[:50].hex()}")
        
        # Check if it's a pickle format
        try:
            import io
            test_pickle = pickle.loads(r50_classifier_bytes)
            print(f"Successfully unpickled as: {type(test_pickle)}")
        except Exception as e:
            print(f"Failed to unpickle: {e}")
        
        # Check if it's compressed
        try:
            import gzip
            with gzip.open(io.BytesIO(r50_classifier_bytes), 'rb') as f:
                decompressed = f.read()
            print(f"Successfully decompressed with gzip, length: {len(decompressed)}")
        except Exception as e:
            print(f"Failed to decompress with gzip: {e}")
            
        # Check if it's a zip file
        try:
            import zipfile
            with zipfile.ZipFile(io.BytesIO(r50_classifier_bytes), 'r') as zf:
                print(f"Successfully opened as zip, files: {zf.namelist()}")
        except Exception as e:
            print(f"Failed to open as zip: {e}")
        
    except Exception as e:
        print(f"Error examining bytes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_bytes()