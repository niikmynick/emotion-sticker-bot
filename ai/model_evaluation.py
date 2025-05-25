import argparse
import matplotlib.pyplot as plt
from keras.src.saving import load_model
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from ai.model_creator import categorical_focal_loss
from util.model import load_human_model, load_animal_model
from properties import (
    HUMAN_DATASET_PATH,
    ANIMAL_DATASET_PATH,
)

HUMAN_DATASET_PATH = '../' + HUMAN_DATASET_PATH
ANIMAL_DATASET_PATH = '../' + ANIMAL_DATASET_PATH

def build_test_gen(dataset_root: str, batch_size: int = 64):
    """
    Создаёт ImageDataGenerator для каталога  <root>/test/
    (shuffle=False важно — порядок классов будет совпадать с .classes)
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        f"{dataset_root}test/",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def evaluate_one(model_loader, dataset_root, title: str):
    gen = build_test_gen(dataset_root)
    model = model_loader()

    # Реальные и предсказанные метки
    y_true = gen.classes
    y_pred = model.predict(gen, verbose=1).argmax(axis=1)

    # Confusion-matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = list(gen.class_indices.keys())

    # -------- визуализация --------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # -------- текстовая сводка --------
    print(title)
    print(classification_report(y_true, y_pred, target_names=labels))


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="только Human-CNN")
    parser.add_argument("--animal", action="store_true", help="только Animal-CNN")
    args = parser.parse_args()

    run_human = args.human or not (args.human or args.animal)
    run_animal = args.animal or not (args.human or args.animal)

    def local_load_model():
        return load_model('../models/curr_best/human_best_model - Copy.keras', custom_objects={ 'loss': categorical_focal_loss(alpha=0.25, gamma=2.0)})
        # return load_model('./models/human.keras')

    if run_human:
        evaluate_one(
            local_load_model,
            HUMAN_DATASET_PATH,
            title="Human-CNN (FER-2013) — confusion matrix",
        )
    if run_animal:
        evaluate_one(
            load_animal_model,
            ANIMAL_DATASET_PATH,
            title="Animal-CNN (Pets Facial Expression) — confusion matrix",
        )