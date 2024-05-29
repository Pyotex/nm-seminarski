from flask import Flask, request, jsonify
from model import predict_emotion
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import cv2
from PIL import Image, ExifTags

def rotate_image(image):
    try:
        # Get the Exif data
        exif = image._getexif()
        if exif is not None:
            # Find the orientation tag
            for tag, value in exif.items():
                key = ExifTags.TAGS.get(tag, tag)
                if key == 'Orientation':
                    orientation = value
                    break
            else:
                orientation = None

            # Rotate the image if needed
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

            # Save the rotated image
            return image
            print(f"Image rotated and saved to {path}")

    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        pass

    return image

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']
    print(image)

    image = Image.open(BytesIO(image.read()))
    image = rotate_image(image)
    # read metadata and rotate the image if necessary


    image = np.array(image)
    # Convert RGB to BGR (required for OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are found, print a message
    if len(faces) == 0:
        print("No faces found")
    else:
        # Get the coordinates of the first face detected
        (x, y, w, h) = faces[0]
        
        # Crop the face from the image
        cropped_face = gray_image[y:y+h, x:x+w]
        
        # Convert the cropped face to PIL format for resizing
        cropped_face_pil = Image.fromarray(cropped_face)
        
        # Resize the cropped face to 48x48
        resized_face = cropped_face_pil.resize((48, 48), Image.LANCZOS)
        
        # Convert the image to grayscale
        grayscale_face = resized_face.convert('L')


    # Convert the image to a numpy array
    image_array = np.array(grayscale_face)

    # Normalize the array to the range [0, 1]
    image_array = image_array / 255.0

    # Convert the numpy array to a PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float32)

    # Add a batch dimension (1, 1, 48, 48)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    result = predict_emotion(image_tensor)

    return jsonify({"message": result}), 200