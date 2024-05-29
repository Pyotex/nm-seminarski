import cv2
from PIL import Image

# Load the image using OpenCV
image_path = 'face3.jpg'  # replace with your image path
image = cv2.imread(image_path)

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
    
    # Save the resulting image
    grayscale_face.save('cropped_face_48x48.jpg')

    print("Face cropped and saved as cropped_face_48x48.jpg")