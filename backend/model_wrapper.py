import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import sys
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Keras 3 compatibility fixes
if 'keras.saving.pickle_utils' not in sys.modules:
    import types
    fake_module = types.ModuleType('keras.saving.pickle_utils')
    fake_module.deserialize_model_from_bytecode = lambda x: x
    sys.modules['keras.saving.pickle_utils'] = fake_module

class FusedModel:
    """Wrapper for the fused model"""
    
    def __init__(self, model_path):
        # Try to load pickle file, but continue even if it fails
        self.model_data = {}
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                print(f"Loaded pickle file: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load pickle file {model_path}: {e}")
                print("Will try to load models from individual .h5 files...")
                self.model_data = {}
        else:
            print(f"Pickle file {model_path} not found. Loading models from individual .h5 files...")
            self.model_data = {}
        
        # Load models from bytes or existing objects
        import tempfile
        
        def load_model_from_bytes(model_bytes):
            """Load Keras model from bytes using temporary file"""
            # The bytes contain a Keras 3 zip format
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Try different loading strategies
                strategies = [
                    lambda: tf.keras.models.load_model(tmp_path, compile=False, safe_mode=False),
                    lambda: tf.keras.models.load_model(tmp_path, compile=False, safe_mode=True),
                    lambda: tf.keras.models.load_model(tmp_path, compile=True, safe_mode=False),
                    lambda: tf.keras.models.load_model(tmp_path, compile=True, safe_mode=True),
                ]
                
                for i, strategy in enumerate(strategies):
                    try:
                        print(f"Trying loading strategy {i+1}...")
                        model = strategy()
                        print(f"Strategy {i+1} succeeded!")
                        return model
                    except Exception as e:
                        print(f"Strategy {i+1} failed: {str(e)[:100]}...")
                        continue
                
                # If all strategies fail, try loading just the architecture
                print("All strategies failed, trying to load architecture only...")
                try:
                    # Try to load as JSON config
                    import zipfile
                    import json
                    with zipfile.ZipFile(tmp_path, 'r') as zf:
                        config_data = zf.read('config.json')
                        config = json.loads(config_data.decode('utf-8'))
                        model = tf.keras.models.model_from_config(config)
                        print("Loaded architecture from config.json")
                        return model
                except Exception as e:
                    print(f"Architecture loading failed: {e}")
                    raise Exception("All model loading strategies failed")
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Load ResNet50 classifier
        if os.path.exists('classifier_r50.h5'):
            print("Loading ResNet50 classifier from H5 file...")
            self.r50_classifier = tf.keras.models.load_model('classifier_r50.h5', compile=False)
        elif os.path.exists('resnet50_classifier.keras'):
            print("Loading ResNet50 classifier from Keras file...")
            self.r50_classifier = tf.keras.models.load_model('resnet50_classifier.keras', compile=False, safe_mode=False)
        elif 'resnet50_classifier' in self.model_data:
            if isinstance(self.model_data['resnet50_classifier'], bytes):
                print("Loading ResNet50 classifier from bytes...")
                self.r50_classifier = load_model_from_bytes(self.model_data['resnet50_classifier'])
            else:
                self.r50_classifier = self.model_data['resnet50_classifier']
        else:
            raise FileNotFoundError("ResNet50 classifier not found. Please ensure classifier_r50.h5 exists or fused_model.pkl contains the model.")
            
        # Load ResNet18 classifier
        if os.path.exists('classifier_r18.h5'):
            print("Loading ResNet18 classifier from H5 file...")
            self.r18_classifier = tf.keras.models.load_model('classifier_r18.h5', compile=False)
        elif os.path.exists('resnet18_classifier.keras'):
            print("Loading ResNet18 classifier from Keras file...")
            self.r18_classifier = tf.keras.models.load_model('resnet18_classifier.keras', compile=False, safe_mode=False)
        elif 'resnet18_classifier' in self.model_data:
            if isinstance(self.model_data['resnet18_classifier'], bytes):
                print("Loading ResNet18 classifier from bytes...")
                self.r18_classifier = load_model_from_bytes(self.model_data['resnet18_classifier'])
            else:
                self.r18_classifier = self.model_data['resnet18_classifier']
        else:
            raise FileNotFoundError("ResNet18 classifier not found. Please ensure classifier_r18.h5 exists or fused_model.pkl contains the model.")
            
        # Load ResNet50 extractor
        if os.path.exists('extractor_r50.h5'):
            print("Loading ResNet50 extractor from H5 file...")
            self.r50_extractor = tf.keras.models.load_model('extractor_r50.h5', compile=False)
        elif os.path.exists('resnet50_extractor.keras'):
            print("Loading ResNet50 extractor from Keras file...")
            self.r50_extractor = tf.keras.models.load_model('resnet50_extractor.keras', compile=False, safe_mode=False)
        elif 'resnet50_extractor' in self.model_data:
            if isinstance(self.model_data['resnet50_extractor'], bytes):
                print("Loading ResNet50 extractor from bytes...")
                self.r50_extractor = load_model_from_bytes(self.model_data['resnet50_extractor'])
            else:
                self.r50_extractor = self.model_data['resnet50_extractor']
        else:
            raise FileNotFoundError("ResNet50 extractor not found. Please ensure extractor_r50.h5 exists or fused_model.pkl contains the model.")
            
        # Load ResNet18 extractor
        if os.path.exists('extractor_r18.h5'):
            print("Loading ResNet18 extractor from H5 file...")
            self.r18_extractor = tf.keras.models.load_model('extractor_r18.h5', compile=False)
        elif os.path.exists('resnet18_extractor.keras'):
            print("Loading ResNet18 extractor from Keras file...")
            self.r18_extractor = tf.keras.models.load_model('resnet18_extractor.keras', compile=False, safe_mode=False)
        elif 'resnet18_extractor' in self.model_data:
            if isinstance(self.model_data['resnet18_extractor'], bytes):
                print("Loading ResNet18 extractor from bytes...")
                self.r18_extractor = load_model_from_bytes(self.model_data['resnet18_extractor'])
            else:
                self.r18_extractor = self.model_data['resnet18_extractor']
        else:
            raise FileNotFoundError("ResNet18 extractor not found. Please ensure extractor_r18.h5 exists or fused_model.pkl contains the model.")
        
        # Load class names and image size
        self.class_names = self.model_data.get('class_names', None)
        self.img_size = self.model_data.get('img_size', None)
        
        # If not in pickle, try loading from config.json
        if self.class_names is None or self.img_size is None:
            if os.path.exists('config.json'):
                print("Loading configuration from config.json...")
                try:
                    import json
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                    self.class_names = config.get('class_names', ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor'])
                    self.img_size = config.get('image_size', 224)
                    print(f"Loaded config: {len(self.class_names)} classes, image size: {self.img_size}")
                except Exception as e:
                    print(f"Failed to load config.json: {e}")
                    if self.class_names is None:
                        self.class_names = ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor']
                    if self.img_size is None:
                        self.img_size = 224
            else:
                # Default values
                if self.class_names is None:
                    self.class_names = ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor']
                if self.img_size is None:
                    self.img_size = 224
        
        # Optional RFE selectors - try loading from pickle file first, then from .pkl files
        self.rfe_r50 = self.model_data.get('rfe_r50', None)
        self.rfe_r18 = self.model_data.get('rfe_r18', None)
        
        # If RFE not in pickle, try loading from .pkl files
        if self.rfe_r50 is None and os.path.exists('rfe_r50.pkl'):
            print("Loading RFE for ResNet50 from rfe_r50.pkl...")
            try:
                with open('rfe_r50.pkl', 'rb') as f:
                    self.rfe_r50 = pickle.load(f)
                print("RFE for ResNet50 loaded successfully")
            except Exception as e:
                print(f"Failed to load rfe_r50.pkl: {e}")
                self.rfe_r50 = None
        
        if self.rfe_r18 is None and os.path.exists('rfe_r18.pkl'):
            print("Loading RFE for ResNet18 from rfe_r18.pkl...")
            try:
                with open('rfe_r18.pkl', 'rb') as f:
                    self.rfe_r18 = pickle.load(f)
                print("RFE for ResNet18 loaded successfully")
            except Exception as e:
                print(f"Failed to load rfe_r18.pkl: {e}")
                self.rfe_r18 = None
    
    def preprocess(self, image):
        """Preprocess image"""
        # Input: image array (H, W, C) or path or PIL Image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img) / 255.0
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img) / 255.0
        else:
            # Assume it's already a numpy array
            img_array = image / 255.0
            
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
            
        return img_array
    
    def kalman_filter(self, prob_seq, process_noise=1e-3, measurement_noise=1e-2):
        """Kalman filter smoothing"""
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
    
    def dempster_shafer_fusion(self, prob1, prob2, eps=1e-12):
        """Dempster-Shafer fusion"""
        combined = prob1 * prob2
        s = combined.sum(axis=1, keepdims=True)
        s[s == 0] = eps
        combined = combined / s
        return combined
    
    def predict(self, image):
        """Make fused prediction"""
        # Preprocess
        img_array = self.preprocess(image)
        
        # Extract features
        feats_r50 = self.r50_extractor.predict(img_array, verbose=0)
        feats_r18 = self.r18_extractor.predict(img_array, verbose=0)
        
        # Apply RFE if available
        if self.rfe_r50 is not None:
            feats_r50 = self.rfe_r50.transform(feats_r50)
        if self.rfe_r18 is not None:
            feats_r18 = self.rfe_r18.transform(feats_r18)
        
        # Get predictions
        pred_r50 = self.r50_classifier.predict(feats_r50, verbose=0)[0]
        pred_r18 = self.r18_classifier.predict(feats_r18, verbose=0)[0]
        
        # Apply Kalman filter
        pred_r50_k = self.kalman_filter(pred_r50.reshape(1, -1))[0]
        pred_r18_k = self.kalman_filter(pred_r18.reshape(1, -1))[0]
        
        # Fuse with DST
        fused = self.dempster_shafer_fusion(
            pred_r50_k.reshape(1, -1),
            pred_r18_k.reshape(1, -1)
        )[0]
        
        # Get final result
        class_idx = np.argmax(fused)
        confidence = float(fused[class_idx])
        predicted_class = self.class_names[class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': fused.tolist(),
            'resnet50_probs': pred_r50.tolist(),
            'resnet18_probs': pred_r18.tolist(),
            'kalman_r50_probs': pred_r50_k.tolist(),
            'kalman_r18_probs': pred_r18_k.tolist()
        }

# Usage:
# model = FusedModel('fused_model_package/fused_model.pkl')
# result = model.predict('image.jpg')
