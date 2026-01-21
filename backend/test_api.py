import requests
import numpy as np
from PIL import Image
import io
import time

# Create a dummy image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

url = 'http://localhost:8002/predict'
files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

print("Sending request to backend...")
try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
    else:
        print("Error Response:")
        print(response.text)
except Exception as e:
    print(f"Connection Error: {e}")
