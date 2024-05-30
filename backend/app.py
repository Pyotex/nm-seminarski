from flask import Flask, request, jsonify
from model import predict_emotion
from PIL import Image, ExifTags
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import time
import cv2

def rotate_image(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                key = ExifTags.TAGS.get(tag, tag)
                if key == 'Orientation':
                    orientation = value
                    break
            else:
                orientation = None

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

            return image
            print(f"Image rotated and saved to {path}")

    except (AttributeError, KeyError, IndexError):
        pass

    return image

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def upload_image():
    start_time = time.time()
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']
    print(image)

    image = Image.open(BytesIO(image.read()))
    image = rotate_image(image)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces found")
    else:
        (x, y, w, h) = faces[0]
        
        cropped_face = gray_image[y:y+h, x:x+w]
        
        cropped_face_pil = Image.fromarray(cropped_face)
        
        resized_face = cropped_face_pil.resize((48, 48), Image.LANCZOS)
        
        grayscale_face = resized_face.convert('L')


    image_array = np.array(grayscale_face)

    image_array = image_array / 255.0

    image_tensor = torch.tensor(image_array, dtype=torch.float32)

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    elapsed_time = time.time() - start_time
    print(f'Preprocessing finished in {elapsed_time:.2f} seconds')

    result = predict_emotion(image_tensor)

    return jsonify({"message": result}), 200