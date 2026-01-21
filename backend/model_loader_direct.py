import pickle
import os
import sys
import numpy as np
from typing import Dict, Any, List
import tensorflow as tf
import keras

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

def load_models_from_files():
    """Load models from individual files instead of fused_model.pkl"""
    models = {}
    
    # Load ResNet50 extractor
    try:
        print("Loading ResNet50 extractor...")
        models['resnet50_extractor'] = keras.models.load_model('extractor_r50.h5')
        print("ResNet50 extractor loaded successfully")
    except Exception as e:
        print(f"Error loading ResNet50 extractor: {e}")
        models['resnet50_extractor'] = None
    
    # Load ResNet50 classifier
    try:
        print("Loading ResNet50 classifier...")
        models['resnet50_classifier'] = keras.models.load_model('classifier_r50.h5')
        print("ResNet50 classifier loaded successfully")
    except Exception as e:
        print(f"Error loading ResNet50 classifier: {e}")
        models['resnet50_classifier'] = None
    
    # Load ResNet18 extractor
    try:
        print("Loading ResNet18 extractor...")
        models['resnet18_extractor'] = keras.models.load_model('extractor_r18.h5')
        print("ResNet18 extractor loaded successfully")
    except Exception as e:
        print(f"Error loading ResNet18 extractor: {e}")
        models['resnet18_extractor'] = None
    
    # Load ResNet18 classifier
    try:
        print("Loading ResNet18 classifier...")
        models['resnet18_classifier'] = keras.models.load_model('classifier_r18.h5')
        print("ResNet18 classifier loaded successfully")
    except Exception as e:
        print(f"Error loading ResNet18 classifier: {e}")
        models['resnet18_classifier'] = None
    
    # Load RFE models
    try:
        print("Loading RFE ResNet50...")
        with open('rfe_r50.pkl', 'rb') as f:
            models['rfe_r50'] = pickle.load(f)
        print("RFE ResNet50 loaded successfully")
    except Exception as e:
        print(f"Error loading RFE ResNet50: {e}")
        models['rfe_r50'] = None
    
    try:
        print("Loading RFE ResNet18...")
        with open('rfe_r18.pkl', 'rb') as f:
            models['rfe_r18'] = pickle.load(f)
        print("RFE ResNet18 loaded successfully")
    except Exception as e:
        print(f"Error loading RFE ResNet18: {e}")
        models['rfe_r18'] = None
    
    # Load config
    try:
        print("Loading config...")
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        models['class_names'] = config.get('class_names', ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor'])
        models['img_size'] = config.get('img_size', 224)
        models['num_classes'] = config.get('num_classes', 4)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        models['class_names'] = ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor']
        models['img_size'] = 224
        models['num_classes'] = 4
    
    return models

def predict_with_models(models: Dict[str, Any], image_array: np.ndarray, model_type: str = 'resnet50'):
    """
    Predict using the loaded models
    
    Args:
        models: Dictionary of loaded models
        image_array: Input image array (224, 224, 3) or (1, 224, 224, 3)
        model_type: 'resnet50' or 'resnet18'
    
    Returns:
        Dictionary with prediction results
    """
    print(f"Using {model_type} model for prediction")
    
    # Ensure input shape is (1, 224, 224, 3)
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Convert to float32
    image_array = image_array.astype('float32')
    
    # Preprocess for ResNet
    print("Applying preprocessing...")
    try:
        if model_type == 'resnet50':
            from tf_keras.applications.resnet50 import preprocess_input
        else:
            from tf_keras.applications.resnet50 import preprocess_input  # Use same for ResNet18
        image_array = preprocess_input(image_array)
    except ImportError:
        try:
            from keras.applications.resnet50 import preprocess_input
            image_array = preprocess_input(image_array)
        except ImportError:
            print("Warning: Could not import preprocess_input, using manual caffe preprocessing")
            def preprocess_input(x):
                x = x.copy()
                x = x[..., ::-1] # RGB -> BGR
                mean = [103.939, 116.779, 123.68]
                x[..., 0] -= mean[0]
                x[..., 1] -= mean[1]
                x[..., 2] -= mean[2]
                return x
            image_array = preprocess_input(image_array)
    
    # Get extractor and classifier based on model type
    extractor = models.get(f'{model_type}_extractor')
    classifier = models.get(f'{model_type}_classifier')
    rfe = models.get(f'rfe_{model_type}')
    
    if not extractor:
        raise ValueError(f"Model {model_type}_extractor not found")
    if not classifier:
        raise ValueError(f"Model {model_type}_classifier not found")
    
    # 1. Feature Extraction
    print("Extracting features...")
    features = extractor.predict(image_array, verbose=0)
    print(f"Feature shape: {features.shape}")
    
    # 2. RFE / Feature Selection
    if rfe:
        print("Applying RFE feature selection...")
        features_selected = rfe.transform(features)
        print(f"Selected feature shape: {features_selected.shape}")
    else:
        print("No RFE model found, using all features")
        features_selected = features
    
    # 3. Classification
    print("Running classification...")
    pred = classifier.predict(features_selected, verbose=0)
    print(f"Raw prediction output: {pred}")
    
    # Map to class
    class_names = models.get('class_names', ['class0_notinfected', 'class1_infected', 'class2_ovariancancer', 'class3_ovariantumor'])
    predicted_index = np.argmax(pred[0])
    predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else f"Class {predicted_index}"
    confidence = float(pred[0][predicted_index])
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': pred[0].tolist(),
        'model_type': model_type,
        'feature_shape': features.shape,
        'selected_feature_shape': features_selected.shape
    }

if __name__ == "__main__":
    print("Loading models from individual files...")
    models = load_models_from_files()
    
    # Test with a dummy image
    print("\nTesting with dummy image...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    try:
        # Test ResNet50
        if models.get('resnet50_extractor') and models.get('resnet50_classifier'):
            print("\nTesting ResNet50...")
            result = predict_with_models(models, test_image, 'resnet50')
            print("ResNet50 Result:")
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Probabilities: {result['probabilities']}")
        
        # Test ResNet18
        if models.get('resnet18_extractor') and models.get('resnet18_classifier'):
            print("\nTesting ResNet18...")
            result = predict_with_models(models, test_image, 'resnet18')
            print("ResNet18 Result:")
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Probabilities: {result['probabilities']}")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll models loaded successfully!")