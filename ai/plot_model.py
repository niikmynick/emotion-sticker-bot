import keras

from ai.model_creator import build_mini_xception

model = build_mini_xception()

dot_img_file = 'model.svg'
keras.utils.plot_model(model, to_file=dot_img_file , dpi=100, show_shapes=True, rankdir='LR')