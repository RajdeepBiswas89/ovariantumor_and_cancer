import os
import sys

# Try to force legacy keras (Keras 2 behavior in TF 2.x)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import pickle
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
try:
    import keras
    print(f"Keras version: {keras.__version__}")
except:
    print("Keras not imported directly")

# Patch for missing modules if needed (from previous attempts)
import types
try:
    import keras.saving.pickle_utils
except ImportError:
    print("Patching keras.saving.pickle_utils...")
    fake_pickle_utils = types.ModuleType("keras.saving.pickle_utils")
import zipfile
import json
import shutil
import h5py

# Create a generic dummy class that accepts any state
class DummyFunctional:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', None)
        self.captured_weights = None
        
    def __setstate__(self, state):
        print("DummyFunctional.__setstate__ called")
        self.__dict__.update(state)
        
    def __call__(self, *args, **kwargs):
        return None
        
    @classmethod
    def from_config(cls, config):
        print("DummyFunctional.from_config called")
        instance = cls(config=config)
        return instance
        
    def get_config(self):
        return self.config or {}
        
    def set_weights(self, weights):
        print(f"DummyFunctional.set_weights called with {len(weights)} arrays")
        self.captured_weights = weights

    def build_from_config(self, config):
        print("DummyFunctional.build_from_config called")
        if self.config is None:
            self.config = config
            
    def compile_from_config(self, config):
        print("DummyFunctional.compile_from_config called")
        
    def compile(self, *args, **kwargs):
        print("DummyFunctional.compile called")

# Patch for keras.saving.pickle_utils to handle broken/incompatible model file
try:
    import keras.saving.pickle_utils
except ImportError:
    print("Patching keras.saving.pickle_utils...")
    fake_pickle_utils = types.ModuleType("keras.saving.pickle_utils")
    
    def fix_and_load(tmp_path, keras_module):
        print(f"Attempting to load model from {tmp_path} using {keras_module.__name__}...")
        try:
            model = keras_module.models.load_model(tmp_path, compile=False, safe_mode=False)
            if isinstance(model, DummyFunctional):
                print("Loaded as DummyFunctional, forcing manual reconstruction to get real model...")
                
                # Try to recover from DummyFunctional if it captured data
                if hasattr(model, 'config') and model.config:
                    print("DummyFunctional has config, trying to reconstruct directly...")
                    try:
                        config_dict = model.config
                        # Remove compile config
                        if 'compile_config' in config_dict:
                            del config_dict['compile_config']
                        if config_dict.get('class_name') == 'Functional' and 'module' in config_dict:
                            del config_dict['module']
                            
                        import keras
                        config_str = json.dumps(config_dict)
                        real_model = keras.models.model_from_json(config_str)
                        print(f"Reconstructed real model: {type(real_model)}")
                        
                        if hasattr(model, 'captured_weights') and model.captured_weights:
                            print(f"DummyFunctional has {len(model.captured_weights)} captured weights. Setting them...")
                            real_model.set_weights(model.captured_weights)
                            return real_model
                        else:
                            print("DummyFunctional has no captured weights. Falling back to file extraction...")
                    except Exception as rec_ex:
                        print(f"Direct reconstruction failed: {rec_ex}")
                        import traceback
                        traceback.print_exc()

                raise ValueError("Loaded as DummyFunctional")
            return model
        except Exception as e:
            print(f"Initial load_model failed with: {type(e).__name__}: {e}")
            if ("expected" in str(e) and "variables" in str(e)) or "DummyFunctional" in str(e):
                print(f"Attempting to fix layer name mismatch: {e}")
                # Extract config
                extract_dir = tmp_path + "_extract"
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                os.makedirs(extract_dir)
                
                try:
                    is_zip = False
                    if zipfile.is_zipfile(tmp_path):
                        print(f"{tmp_path} is a zip file. Extracting...")
                        with zipfile.ZipFile(tmp_path, 'r') as z:
                            print(f"Zip contents: {z.namelist()}")
                            z.extractall(extract_dir)
                        is_zip = True
                    else:
                        print(f"{tmp_path} is NOT a zip file. Assuming HDF5.")
                        
                    config_path = os.path.join(extract_dir, 'config.json')
                    weights_path = os.path.join(extract_dir, 'model.weights.h5')
                    
                    if os.path.exists(config_path) and os.path.exists(weights_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Get weights keys
                        weight_names = []
                        weight_groups = {} # Map cleaned name (e.g. conv2d) to list of full names
                        
                        def visit_weights(name, obj):
                            if isinstance(obj, h5py.Group) and 'vars' in obj:
                                # Check if vars is not empty
                                if len(obj['vars']) > 0:
                                    # name is like _layer_checkpoint_dependencies/dense
                                    if name.startswith('_layer_checkpoint_dependencies/'):
                                        layer_name = name.split('/')[-1]
                                        weight_names.append(layer_name)
                                        
                                        # Parse type
                                        # simple heuristic: remove digits and underscores at end
                                        import re
                                        base_type = re.sub(r'_\d+$', '', layer_name)
                                        # handle 'conv2d' vs 'conv2d_1' -> both conv2d
                                        
                                        if base_type not in weight_groups:
                                            weight_groups[base_type] = []
                                        weight_groups[base_type].append(layer_name)
                        
                        with h5py.File(weights_path, 'r') as f:
                            f.visititems(visit_weights)
                        
                        # Sort weight groups by index
                        for k in weight_groups:
                            # Sort by number suffix: conv2d, conv2d_1, conv2d_2 ...
                            weight_groups[k].sort(key=lambda x: (len(x), x))
                        
                        print(f"Found weights groups: {list(weight_groups.keys())}")
                        
                        # Create a map based on type matching
                        name_map = {}
                        used_weights = set()
                        
                        # Map Keras class names to weight prefixes
                        class_to_prefix = {
                            'Conv2D': 'conv2d',
                            'BatchNormalization': 'batch_normalization',
                            'Dense': 'dense',
                            'PReLU': 're_lu', 
                            'ReLU': 're_lu', 
                            # 'Add': 'add', # Add usually doesn't have weights in vars? But found in list.
                        }
                        
                        # Iterate config layers
                        layers = config['config']['layers']
                        
                        # Counters for each type in config
                        type_counters = {}
                        
                        for layer in layers:
                            c_name = layer['config']['name']
                            c_class = layer['class_name']
                            
                            if c_class in class_to_prefix:
                                prefix = class_to_prefix[c_class]
                                if prefix in weight_groups:
                                    # Get next available weight layer of this type
                                    candidates = weight_groups[prefix]
                                    
                                    # We need to maintain a counter for this prefix to match sequentially
                                    if prefix not in type_counters:
                                        type_counters[prefix] = 0
                                    
                                    idx = type_counters[prefix]
                                    if idx < len(candidates):
                                        target = candidates[idx]
                                        name_map[c_name] = target
                                        type_counters[prefix] += 1
                        
                        print(f"Computed topological name map: {len(name_map)} mappings")
                        
                        # Manual reconstruction
                        # 1. Load config
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        
                        # Remove compile config to avoid optimizer errors
                        if 'compile_config' in config_dict:
                            del config_dict['compile_config']
                            
                        # Helper to sanitize config (handle __numpy__ serialization)
                        def sanitize_config(obj):
                            if isinstance(obj, dict):
                                if 'class_name' in obj and obj['class_name'] == '__numpy__':
                                    if 'config' in obj and 'value' in obj['config']:
                                        return obj['config']['value']
                                return {k: sanitize_config(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [sanitize_config(item) for item in obj]
                            return obj
                        
                        print("Sanitizing config...", flush=True)
                        config_dict = sanitize_config(config_dict)

                        # Sanitize config to ensure we get a real model, not DummyFunctional
                        if config_dict.get('class_name') == 'Functional':
                            if 'module' in config_dict:
                                del config_dict['module']
                            if 'registered_name' in config_dict:
                                del config_dict['registered_name']
                        
                        print("Dumping config to json...", flush=True)
                        try:
                            config_str = json.dumps(config_dict)
                        except Exception as json_err:
                            print(f"JSON dump failed: {json_err}", flush=True)
                            raise json_err
                        
                        # 2. Create model from config (using Keras 3 as it is default)
                        print("Creating model from json...", flush=True)
                        import keras
                        
                        # Use custom_objects to map 'Functional' to keras.Model
                        custom_objects = {'Functional': keras.models.Model, 'Model': keras.models.Model}
                        model = keras.models.model_from_json(config_str, custom_objects=custom_objects)
                        print(f"Model created: {type(model)}", flush=True)
                        
                        # 3. Load weights manually using the map
                        print("Loading weights manually with topological map...", flush=True)
                        with h5py.File(weights_path, 'r') as hf:
                            for layer in model.layers:
                                layer_name = layer.name
                                
                                # Use map if available, else try direct name
                                weight_name = name_map.get(layer_name, layer_name)
                                
                                # Construct path
                                grp_path = f"_layer_checkpoint_dependencies/{weight_name}"
                                if grp_path in hf:
                                    grp = hf[grp_path]
                                    if 'vars' in grp:
                                        vars_grp = grp['vars']
                                        if len(vars_grp) > 0:
                                            weights = []
                                            keys = sorted(vars_grp.keys(), key=lambda x: int(x))
                                            for k in keys:
                                                weights.append(vars_grp[k][()])
                                            
                                            try:
                                                layer.set_weights(weights)
                                            except Exception as w_err:
                                                print(f"Failed to set weights for {layer_name} from {weight_name}: {w_err}", flush=True)
                                else:
                                    # Only complain if we expected weights (e.g. it was in the map)
                                    if layer_name in name_map:
                                        print(f"Expected weights for {layer_name} (mapped to {weight_name}) but not found at {grp_path}", flush=True)
                                    
                        print("Manual reconstruction successful, returning model...", flush=True)
                        return model
                        
                except Exception as ex:
                    msg = f"Fix attempt failed: {ex}"
                    print(msg, flush=True)
                    with open("debug_error.txt", "w") as f:
                        f.write(msg + "\n")
                        import traceback
                        traceback.print_exc(file=f)
            
            print("Raising original exception...", flush=True)
            raise e

    def deserialize_model_from_bytecode(bytecode, *args, **kwargs):
        print(f"deserialize_model_from_bytecode called with {len(bytecode)} bytes")
        
        import tempfile
        # Try to import tf_keras to use for loading if possible, as it is more compatible with legacy models
        try:
            import tf_keras as keras_to_use
            print("Using tf_keras for deserialization")
        except ImportError:
            import keras as keras_to_use
            print("Using keras (Keras 3) for deserialization")
            
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp.write(bytecode)
            tmp_path = tmp.name
        
        try:
            return fix_and_load(tmp_path, keras_to_use)
        except Exception as e:
            print(f"Error in deserialize_model_from_bytecode: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            if os.path.exists(tmp_path):
                # Clean up
                # os.unlink(tmp_path)
                pass
    
    fake_pickle_utils.deserialize_model_from_bytecode = deserialize_model_from_bytecode
    sys.modules["keras.saving.pickle_utils"] = fake_pickle_utils

# Patch for keras.engine.functional (older Keras structure)
try:
    import keras.engine.functional
except ImportError:
    print("Patching keras.engine.functional...")
    fake_functional = types.ModuleType("keras.engine.functional")
    import keras
    # Create a generic dummy class that accepts any state
    # DummyFunctional is defined globally now
    
    # Map Functional to DummyFunctional
    fake_functional.Functional = DummyFunctional 
    sys.modules["keras.engine.functional"] = fake_functional

# Patch for tf_keras.engine.functional (older Keras structure referenced as tf_keras)
try:
    import tf_keras.engine.functional
except ImportError:
    print("Patching tf_keras.engine.functional...")
    # Ensure tf_keras.engine exists
    try:
        import tf_keras.engine
    except ImportError:
        fake_tf_keras_engine = types.ModuleType("tf_keras.engine")
        sys.modules["tf_keras.engine"] = fake_tf_keras_engine
    
    fake_tf_functional = types.ModuleType("tf_keras.engine.functional")
    # Map to same as keras.engine.functional
    fake_tf_functional.Functional = DummyFunctional
    sys.modules["tf_keras.engine.functional"] = fake_tf_functional

def load_model_from_h5(h5_path, model_dir=None):
    """Load a Keras model from .h5 file"""
    if model_dir:
        h5_path = os.path.join(model_dir, h5_path)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found: {h5_path}")
    
    print(f"Loading model from {h5_path}...")
    try:
        import tf_keras as keras_module
        print("Using tf_keras to load model")
    except ImportError:
        try:
            import keras as keras_module
            print("Using keras to load model")
        except ImportError:
            raise ImportError("Neither tf_keras nor keras is available")
    
    try:
        model = keras_module.models.load_model(h5_path, compile=False, safe_mode=False)
        print(f"Successfully loaded model: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model from {h5_path}: {e}")
        # Try with the fix_and_load function
        return fix_and_load(h5_path, keras_module)

def load_rfe_from_pkl(pkl_path, model_dir=None):
    """Load RFE selector from pickle file"""
    if model_dir:
        pkl_path = os.path.join(model_dir, pkl_path)
    
    if not os.path.exists(pkl_path):
        print(f"RFE file not found: {pkl_path}, returning None")
        return None
    
    print(f"Loading RFE from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        rfe = pickle.load(f)
    print(f"RFE loaded successfully: {type(rfe)}")
    return rfe

def load_fused_model(model_path='fused_model.pkl', model_dir=None):
    """
    Load the fused model from pickle file, with fallback to individual .h5 files.
    If models in pickle are not valid, load from separate .h5 files.
    """
    # Determine model directory
    if model_dir is None:
        abs_path = os.path.abspath(model_path)
        model_dir = os.path.dirname(abs_path)
        if not model_dir or model_dir == os.path.dirname(os.getcwd()):
            model_dir = '.'
    
    # Ensure model_path is absolute
    if not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, model_path)
    
    data = {}
    
    # Try to load from pickle first
    if os.path.exists(model_path):
        print(f"Attempting to load from pickle: {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            print("Pickle loaded successfully!")
            
            # Validate that models are loaded correctly (not DummyFunctional)
            r50_classifier = data.get('resnet50_classifier') or data.get('r50_classifier')
            if r50_classifier and (isinstance(r50_classifier, DummyFunctional) or type(r50_classifier).__name__ == 'DummyFunctional'):
                print("Warning: Models in pickle are DummyFunctional, loading from .h5 files instead...")
                data = {}
            else:
                # Check if all required models are present and callable
                required_keys = ['resnet50_classifier', 'resnet18_classifier', 'resnet50_extractor', 'resnet18_extractor']
                missing_keys = []
                for k in required_keys:
                    if k not in data or data[k] is None:
                        missing_keys.append(k)
                    elif not hasattr(data[k], 'predict'):
                        print(f"Warning: {k} exists but doesn't have predict method")
                        missing_keys.append(k)
                
                if missing_keys:
                    print(f"Missing or invalid models in pickle: {missing_keys}, loading from .h5 files...")
                    data = {}
                elif 'class_names' not in data:
                    # Add default class names if missing
                    data['class_names'] = ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor']
                if 'img_size' not in data:
                    data['img_size'] = 224
        except Exception as e:
            print(f"Error loading pickle: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to loading from .h5 files...")
            data = {}
    
    # If pickle didn't work or models are missing, load from individual files
    if not data or 'resnet50_classifier' not in data:
        print("Loading models from individual .h5 and .pkl files...")
        
        # Load extractors
        print("Loading extractors...")
        data['resnet50_extractor'] = load_model_from_h5('extractor_r50.h5', model_dir)
        data['resnet18_extractor'] = load_model_from_h5('extractor_r18.h5', model_dir)
        
        # Load classifiers
        print("Loading classifiers...")
        data['resnet50_classifier'] = load_model_from_h5('classifier_r50.h5', model_dir)
        data['resnet18_classifier'] = load_model_from_h5('classifier_r18.h5', model_dir)
        
        # Load RFE selectors
        print("Loading RFE selectors...")
        data['rfe_r50'] = load_rfe_from_pkl('rfe_r50.pkl', model_dir)
        data['rfe_r18'] = load_rfe_from_pkl('rfe_r18.pkl', model_dir)
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                data['class_names'] = config.get('class_names', ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor'])
                data['img_size'] = config.get('image_size', 224)
        else:
            # Default values
            data['class_names'] = ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor']
            data['img_size'] = 224
        
        print("All models loaded from individual files!")
    
    return data

def kalman_filter(prob_seq, process_noise=1e-3, measurement_noise=1e-2):
    """Kalman filter smoothing for probability sequences"""
    n_timesteps, n_classes = prob_seq.shape
    filtered_probs = np.zeros_like(prob_seq)
    P = np.eye(n_classes) * 1.0  # Initialize with identity
    Q = np.eye(n_classes) * process_noise
    R = np.eye(n_classes) * measurement_noise
    x = prob_seq[0].copy()
    
    for t in range(n_timesteps):
        z = prob_seq[t]
        # Prediction
        P = P + Q
        # Update
        K = P @ np.linalg.inv(P + R)
        x = x + K @ (z - x)
        P = (np.eye(n_classes) - K) @ P
        filtered_probs[t] = x
        
    return filtered_probs

def dempster_shafer_fusion(prob1, prob2, eps=1e-12):
    """Dempster-Shafer theory fusion of two probability distributions"""
    # Ensure both are numpy arrays
    prob1 = np.array(prob1)
    prob2 = np.array(prob2)
    
    # Ensure same shape
    if prob1.shape != prob2.shape:
        if prob1.ndim == 1 and prob2.ndim == 2:
            prob1 = prob1.reshape(1, -1)
        elif prob1.ndim == 2 and prob2.ndim == 1:
            prob2 = prob2.reshape(1, -1)
        elif prob1.ndim == 1 and prob2.ndim == 1:
            # Both are 1D, ensure they match
            if prob1.shape[0] != prob2.shape[0]:
                raise ValueError(f"Shape mismatch: prob1={prob1.shape}, prob2={prob2.shape}")
    
    combined = prob1 * prob2
    
    # Normalize along the class dimension
    if combined.ndim == 2:
        s = combined.sum(axis=1, keepdims=True)
    else:
        s = np.sum(combined, keepdims=True)
    
    s = np.where(s == 0, eps, s)
    combined = combined / s
    return combined

def predict_with_model(data, image_array):
    """
    Perform fused prediction using ResNet50 and ResNet18 with Kalman filtering and DST fusion.
    image_array should be (224, 224, 3) or (1, 224, 224, 3).
    Returns a dictionary with prediction results including both models' outputs.
    """
    import numpy as np
    
    # Ensure input shape is (1, 224, 224, 3)
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Convert to float32
    image_array = image_array.astype('float32')
    
    # Normalize to [0, 1] range (model_wrapper.py uses /255.0)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # Get model components
    r50_extractor = data.get('resnet50_extractor') or data.get('r50_extractor')
    r18_extractor = data.get('resnet18_extractor') or data.get('r18_extractor')
    r50_classifier = data.get('resnet50_classifier') or data.get('r50_classifier')
    r18_classifier = data.get('resnet18_classifier') or data.get('r18_classifier')
    rfe_r50 = data.get('rfe_r50') or data.get('r50_rfe')
    rfe_r18 = data.get('rfe_r18') or data.get('r18_rfe')
    class_names = data.get('class_names', ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor'])
    
    if not r50_extractor or not r18_extractor or not r50_classifier or not r18_classifier:
        missing = []
        if not r50_extractor: missing.append('resnet50_extractor')
        if not r18_extractor: missing.append('resnet18_extractor')
        if not r50_classifier: missing.append('resnet50_classifier')
        if not r18_classifier: missing.append('resnet18_classifier')
        raise ValueError(f"Missing required models: {missing}. Available keys: {list(data.keys())}")
    
    print("Extracting features with ResNet50...")
    # Extract features with ResNet50
    feats_r50 = r50_extractor.predict(image_array, verbose=0)
    
    print("Extracting features with ResNet18...")
    # Extract features with ResNet18
    feats_r18 = r18_extractor.predict(image_array, verbose=0)
    
    # Apply RFE if available
    if rfe_r50 is not None:
        print("Applying RFE to ResNet50 features...")
        feats_r50 = rfe_r50.transform(feats_r50)
    
    if rfe_r18 is not None:
        print("Applying RFE to ResNet18 features...")
        feats_r18 = rfe_r18.transform(feats_r18)
    
    # Get predictions from classifiers
    print("Getting prediction from ResNet50 classifier...")
    pred_r50 = r50_classifier.predict(feats_r50, verbose=0)
    if pred_r50.ndim > 1 and pred_r50.shape[0] == 1:
        pred_r50 = pred_r50[0]
    
    print("Getting prediction from ResNet18 classifier...")
    pred_r18 = r18_classifier.predict(feats_r18, verbose=0)
    if pred_r18.ndim > 1 and pred_r18.shape[0] == 1:
        pred_r18 = pred_r18[0]
    
    # Apply Kalman filter
    print("Applying Kalman filter...")
    pred_r50_k = kalman_filter(pred_r50.reshape(1, -1))[0]
    pred_r18_k = kalman_filter(pred_r18.reshape(1, -1))[0]
    
    # Fuse with Dempster-Shafer theory
    print("Fusing predictions with Dempster-Shafer theory...")
    fused = dempster_shafer_fusion(
        pred_r50_k.reshape(1, -1),
        pred_r18_k.reshape(1, -1)
    )[0]
    
    # Get final result
    class_idx = np.argmax(fused)
    confidence = float(fused[class_idx])
    predicted_class = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': fused.tolist(),
        'resnet50_probs': pred_r50.tolist() if hasattr(pred_r50, 'tolist') else list(pred_r50),
        'resnet18_probs': pred_r18.tolist() if hasattr(pred_r18, 'tolist') else list(pred_r18),
        'kalman_r50_probs': pred_r50_k.tolist() if hasattr(pred_r50_k, 'tolist') else list(pred_r50_k),
        'kalman_r18_probs': pred_r18_k.tolist() if hasattr(pred_r18_k, 'tolist') else list(pred_r18_k)
    }

if __name__ == "__main__":
    model_path = 'fused_model.pkl'
    try:
        data = load_fused_model(model_path)
        print("Keys:", data.keys())
        
        if 'resnet50_classifier' in data:
            print("resnet50_classifier type:", type(data['resnet50_classifier']))
            model = data['resnet50_classifier']
            
            # Check if it is our dummy object
            if isinstance(model, DummyFunctional) or type(model).__name__ == 'DummyFunctional':
                print("Model loaded as DummyFunctional. Attempting to extract config and weights...")
                # Inspect what we have
                print("Attributes:", model.__dict__.keys())
                pass
            else:
                # Try a full pipeline prediction
                try:
                    # Test input image
                    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    print(f"Testing prediction with input shape {test_input.shape}")
                    
                    result = predict_with_model(data, test_input)
                    print(f"Prediction result: {result}")
                        
                except Exception as e:
                    print(f"Prediction failed: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"Failed to load pickle: {e}")
        import traceback
        traceback.print_exc()
