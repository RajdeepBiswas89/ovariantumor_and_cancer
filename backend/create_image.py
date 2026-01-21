from PIL import Image
import numpy as np

img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save('test_image.jpg')
print("Created test_image.jpg")
