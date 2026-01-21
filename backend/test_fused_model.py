#!/usr/bin/env python3
"""
Test script for the fused model using model_wrapper.py
"""
import numpy as np
from PIL import Image
import os
from model_wrapper import FusedModel

def test_fused_model():
    """Test the fused model with a sample image"""
    
    print("Testing fused model...")
    
    try:
        # Load the fused model
        print("Loading fused model from fused_model.pkl...")
        model = FusedModel('fused_model.pkl')
        print("Fused model loaded successfully!")
        
        # Create a test image (224x224 RGB)
        print("Creating test image...")
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test prediction
        print("Running prediction...")
        result = model.predict(test_image)
        
        print("Prediction successful!")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All probabilities: {result['all_probabilities']}")
        print(f"ResNet50 probs: {result['resnet50_probs']}")
        print(f"ResNet18 probs: {result['resnet18_probs']}")
        print(f"Kalman filtered ResNet50: {result['kalman_r50_probs']}")
        print(f"Kalman filtered ResNet18: {result['kalman_r18_probs']}")
        
        return True
        
    except Exception as e:
        print(f"Error testing fused model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fused_model()
    if success:
        print("\n✅ Fused model test completed successfully!")
    else:
        print("\n❌ Fused model test failed!")