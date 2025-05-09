from keras.src.saving import load_model

from properties import HUMAN_MODEL_PATH, HUMAN_BEST_MODEL_PATH, ANIMAL_MODEL_PATH, ANIMAL_BEST_MODEL_PATH


def load_human_model():
    return load_model(HUMAN_BEST_MODEL_PATH)

def load_animal_model():
    return load_model(ANIMAL_BEST_MODEL_PATH)


if __name__ == '__main__':
    a = load_human_model()
    print(a)
