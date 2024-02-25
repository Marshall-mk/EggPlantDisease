import tensorflow as tf
import numpy as np
from PIL import Image


class EggPlantPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)
        self.image_size = self.model.input_shape[1]

    # def preprocess(self, image):
    #     image = tf.image.resize(image, (self.image_size, self.image_size))
    #     image = np.expand_dims(np.expand_dims(image, -1), 0)
    #     return tf.cast(image, tf.float32) / 255.0

    def infer(self, image):
        tensor_image = tf.convert_to_tensor(
            np.array(Image.open(image)), dtype=tf.float32
        )[np.newaxis, :]
        # #tensor_image = self.preprocess(tensor_image)
        # shape= tensor_image.shape
        # tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        # print(tensor_image.shape)

        pred = self.model.predict(tensor_image)
        pred = int(np.argmax(pred))
        return {"output": pred}


if __name__ == "__main__":
    image = "../data/test/.."
    predictor = EggPlantPredictor("../models/Model.h5")
    print(predictor.infer(image))
