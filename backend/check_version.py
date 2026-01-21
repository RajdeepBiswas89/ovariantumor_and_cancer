import tensorflow as tf
import keras
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
try:
    import tf_keras
    print(f"tf_keras version: {tf_keras.__version__}")
except ImportError:
    print("tf_keras not installed")
