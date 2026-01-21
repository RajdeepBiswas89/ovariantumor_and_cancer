#!/usr/bin/env python3
"""
Debug script to extract and save individual model files from fused_model.pkl
"""
import pickle
import sys
import zipfile
import io
import os

# Keras 3 compatibility fixes
if 'keras.saving.pickle_utils' not in sys.modules:
    import types
    fake_module = types.ModuleType('keras.saving.pickle_utils')
    fake_module.deserialize_model_from_bytecode = lambda x: x
    sys.modules['keras.saving.pickle_utils'] = fake_module

def extract_models():
    """Extract individual model files from fused_model.pkl"""
    
    print("Extracting models from fused_model.pkl...")
    
    try:
        with open('fused_model.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Extract each model as .keras file
        models_to_extract = [
            ('resnet50_classifier', 'resnet50_classifier.keras'),
            ('resnet18_classifier', 'resnet18_classifier.keras'),
            ('resnet50_extractor', 'resnet50_extractor.keras'),
            ('resnet18_extractor', 'resnet18_extractor.keras')
        ]
        
        for model_key, filename in models_to_extract:
            model_bytes = data[model_key]
            with open(filename, 'wb') as f:
                f.write(model_bytes)
            print(f"Extracted {model_key} to {filename} ({len(model_bytes)} bytes)")
        
        # Extract RFE models
        for rfe_key in ['rfe_r50', 'rfe_r18']:
            if rfe_key in data:
                rfe_filename = f"{rfe_key}.pkl"
                with open(rfe_filename, 'wb') as f:
                    pickle.dump(data[rfe_key], f)
                print(f"Extracted {rfe_key} to {rfe_filename}")
        
        # Extract config
        config_filename = 'fused_model_config.json'
        import json
        with open(config_filename, 'w') as f:
            json.dump({
                'class_names': data.get('class_names', []),
                'img_size': data.get('img_size', 224),
                'num_classes': data.get('num_classes', 4)
            }, f, indent=2)
        print(f"Extracted config to {config_filename}")
        
        print("\nAll models extracted successfully!")
        
    except Exception as e:
        print(f"Error extracting models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_models()