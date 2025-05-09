import json
import numpy as np
import cv2

from util.model import load_human_model
from util.face import extract_face, preprocess_face

model = load_human_model()

with open('human_class_indices.json', 'r') as f:
    class_indices = json.load(f)

indices_to_emotion = {v: k for k, v in class_indices.items()}


def get_human_emotion(image):
    face = extract_face(image)
    if face is None:
        return None

    face_input = preprocess_face(face)
    face_input = np.expand_dims(face_input, axis=0)

    preds = model.predict(face_input)
    idx = int(np.argmax(preds[0]))
    return indices_to_emotion.get(idx, None)


if __name__ == '__main__':
    image = cv2.imread('a.jpg')
    emotion = get_human_emotion(image)
    if emotion:
        print(f"Detected emotion: {emotion}")
    else:
        print("No face detected or emotion unknown.")