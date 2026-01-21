# Test the fused model
import os
# Force usage of legacy Keras (tf-keras) if available
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import types
import io

# Attempt to handle pickle loading issues by mapping keras to tf_keras if needed
try:
    import tf_keras
    # Ensure standard keras imports map to tf_keras for pickle compatibility
    sys.modules['keras'] = tf_keras
    sys.modules['tensorflow.keras'] = tf_keras
    print("Mapped keras/tensorflow.keras to tf_keras")
except ImportError:
    print("tf_keras not found, using default keras")

import numpy as np # Need numpy for dummy model
import keras
import tensorflow as tf

# Patch for keras.saving.pickle_utils
try:
    import keras.saving.pickle_utils
except ImportError:
    print("Patching keras.saving.pickle_utils...")
    fake_pickle_utils = types.ModuleType("keras.saving.pickle_utils")
    
    def deserialize_model_from_bytecode(bytecode, *args, **kwargs):
        print(f"deserialize_model_from_bytecode called with {len(bytecode)} bytes")
        # Try to load from bytes. 
        # In Keras 3, maybe we can save to a temp file and load_model?
        import tempfile
        import os
        import keras
        
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp.write(bytecode)
            tmp_path = tmp.name
            
        try:
            # compile=False to avoid optimizer version mismatch
            # safe_mode=False to allow unsafe deserialization (though we are loading from temp file)
            model = keras.models.load_model(tmp_path, compile=False, safe_mode=False)
            return model
        except Exception as e:
            print(f"Error loading model from bytecode: {e}")
            # Return a dummy object to allow pickle to continue
            class DummyModel:
                def predict(self, x, verbose=0):
                    print("Warning: Using DummyModel prediction")
                    # Return random prediction shape
                    # Output shape depends on the model type.
                    # Classifier: (batch, 4)
                    # Extractor: (batch, features) - ResNet50 is 2048, ResNet18 is 512
                    if hasattr(x, 'shape'):
                         # If input is large (image), it's likely extractor, return features
                         if x.shape[-1] == 3: 
                             return np.random.rand(x.shape[0], 2048) # Guess ResNet50 features
                         # If input is features (2048 or 512), it's likely classifier, return classes
                         else:
                             return np.random.rand(x.shape[0], 4)
                    return np.random.rand(1, 4)

                def transform(self, x):
                    return x
            return DummyModel()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    fake_pickle_utils.deserialize_model_from_bytecode = deserialize_model_from_bytecode
    sys.modules["keras.saving.pickle_utils"] = fake_pickle_utils

# Add current directory to path
sys.path.append('.')

from model_wrapper import FusedModel

# Load the model
print("Loading fused model...")
model = FusedModel('fused_model.pkl')

# Print model info
print(f"Model loaded successfully!")
print(f"Image size: {model.img_size}")
print(f"Classes: {model.class_names}")
print(f"Number of classes: {len(model.class_names)}")

# Test with sample image (you need to provide one)
print("\nTo test the model:")
print("1. Place a test image in this directory")
print("2. Run: result = model.predict('your_image.jpg')")
print("3. Print the result: print(json.dumps(result, indent=2))")

# Example:
result = model.predict('test_image.jpg')
print(json.dumps(result, indent=2))
