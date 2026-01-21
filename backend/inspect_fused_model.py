#!/usr/bin/env python3
"""
Debug script to inspect the fused_model.pkl contents
"""
import pickle
import sys

# Keras 3 compatibility fixes
if 'keras.saving.pickle_utils' not in sys.modules:
    import types
    fake_module = types.ModuleType('keras.saving.pickle_utils')
    fake_module.deserialize_model_from_bytecode = lambda x: x
    sys.modules['keras.saving.pickle_utils'] = fake_module

def inspect_fused_model():
    """Inspect the contents of fused_model.pkl"""
    
    print("Inspecting fused_model.pkl...")
    
    try:
        with open('fused_model.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded pickle successfully!")
        print(f"Keys in fused_model.pkl: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n{key}:")
            if hasattr(value, '__class__'):
                print(f"  Type: {type(value)}")
                print(f"  Class: {value.__class__}")
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")
            
            # If it's a model, check if it has predict method
            if hasattr(value, 'predict'):
                print(f"  Has predict method: Yes")
            else:
                print(f"  Has predict method: No")
                
            # If it's bytes, show first few bytes
            if isinstance(value, bytes):
                print(f"  First 20 bytes: {value[:20]}")
        
    except Exception as e:
        print(f"Error inspecting fused_model.pkl: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_fused_model()