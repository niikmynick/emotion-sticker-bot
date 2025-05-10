import cv2
import numpy as np
from mtcnn import MTCNN


detector = MTCNN()


def extract_face(image, required_size=(224, 224)):
    # Convert BGR (cv2.imread) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(image_rgb)
    if not results:
        return None

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)

    face = image_rgb[y : y + h, x : x + w]

    face = cv2.resize(face, required_size)
    return face


def preprocess_face(img, size=(48, 48)) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_resized = cv2.resize(gray, size)
    face_resized = face_resized.astype("float32") / 255.0

    face_resized = np.expand_dims(face_resized, axis=-1)
    return face_resized


if __name__ == '__main__':
    image = cv2.imread('a.jpg')
    face = extract_face(image)
    if face is not None:
        # face = preprocess_face(face)
        cv2.imshow('Extracted Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")
