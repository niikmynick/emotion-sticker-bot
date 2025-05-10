import json
import numpy as np
import cv2

from util.model import load_animal_model


model = load_animal_model()

with open('animal_class_indices.json', 'r') as f:
    class_indices = json.load(f)

indices_to_emotion = {v: k for k, v in class_indices.items()}


def get_animal_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img48 = cv2.resize(gray, (48, 48))
    x = img48.astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))

    preds = model.predict(x)
    idx = int(preds.argmax(axis=1)[0])
    return indices_to_emotion[idx]


if __name__ == '__main__':
    image = cv2.imread('a.jpeg')
    emotion = get_animal_emotion(image)
    if emotion:
        print(f"Detected emotion: {emotion}")
    else:
        print("No face detected.")
