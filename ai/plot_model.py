import keras

from ai.model_creator import build_model

model = build_model()

dot_img_file = 'model.svg'
keras.utils.plot_model(model, to_file=dot_img_file , dpi=100, show_shapes=True)