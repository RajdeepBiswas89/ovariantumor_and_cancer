import pickle
import os
import sys
import numpy as np

# Patch for keras.saving.pickle_utils and others
import types
try:
    import keras.saving.pickle_utils
except ImportError:
    fake_pickle_utils = types.ModuleType("keras.saving.pickle_utils")
    def deserialize_model_from_bytecode(bytecode, *args, **kwargs):
        return None 
    fake_pickle_utils.deserialize_model_from_bytecode = deserialize_model_from_bytecode
    sys.modules["keras.saving.pickle_utils"] = fake_pickle_utils

# Patch for tf_keras.engine.functional
try:
    import tf_keras.engine.functional
except ImportError:
    fake_tf_functional = types.ModuleType("tf_keras.engine.functional")
    class DummyFunctional:
        def __setstate__(self, state): self.__dict__.update(state)
        def __call__(self, *args, **kwargs): return None
    fake_tf_functional.Functional = DummyFunctional
    sys.modules["tf_keras.engine.functional"] = fake_tf_functional

# We need the real loader patches
from real_model_loader import load_fused_model

model_path = 'fused_model.pkl'
try:
    print(f"Loading {model_path}...")
    data = load_fused_model(model_path)
    print("\nKeys in pickle:", data.keys())
    
    if 'class_names' in data:
        print("Found class_names:", data['class_names'])
    else:
        print("No 'class_names' key found.")
        
    if 'resnet50_extractor' in data:
        print("resnet50_extractor:", type(data['resnet50_extractor']))
        
    if 'rfe_r50' in data:
        print("rfe_r50:", type(data['rfe_r50']))
        
    if 'resnet50_classifier' in data:
        print("resnet50_classifier:", type(data['resnet50_classifier']))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
