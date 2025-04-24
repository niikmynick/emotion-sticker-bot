import cv2
from mtcnn import MTCNN

detector = MTCNN()


def extract_face(image_path: str):
    results = detector.detect_faces(image_path)
    if not results:
        return None
    x, y, w, h = results[0]['box']
    face_detected = image_path[y:y+h, x:x+w]
    return cv2.resize(face_detected, (224, 224))


if __name__ == '__main__':
    image = cv2.imread('a.jpg')
    face = extract_face(image)
    if face is not None:
        cv2.imshow('Extracted Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")
