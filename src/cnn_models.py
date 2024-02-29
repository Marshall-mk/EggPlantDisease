from keras import layers, models
from keras.applications import EfficientNetB0, DenseNet121
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

def dense_net_model(input_shape, classes=7):
    """Finetune DenseNet model"""
    model = DenseNet121(
        input_shape=input_shape, include_top=True, weights='imagenet', classes=classes
    )
    # train only the top layers
    for layer in model.layers[:-5]:
        layer.trainable = False
    return model

def vgg16_model(input_shape, classes=7):
    """Finetune VGG16 model"""
    model = VGG16(
        input_shape=input_shape, include_top=True, weights='imagenet', classes=classes
    )
    # train only the top layers
    for layer in model.layers[:-5]:
        layer.trainable = False
    return model

def efficient_net_model(input_shape, classes=7):
    """Finetune EfficientNet model"""
    model = EfficientNetB0(
        input_shape=input_shape, include_top=True, weights='imagenet', classes=classes
    )
    # train only the top layers
    for layer in model.layers[:-5]:
        layer.trainable = False
    return model


def dense_net_model_FT(input_shape, classes=7):
    """Fully train DenseNet model"""
    inputs = layers.Input(shape=input_shape)
    base_model = DenseNet121(
        input_shape=input_shape, include_top=True, weights=None, classes=classes
    )(inputs)
    return models.Model(inputs, base_model)


def vgg16_model_FT(input_shape, classes=7):
    """Fully train VGG16 model"""
    inputs = layers.Input(shape=input_shape)
    base_model = VGG16(
        input_shape=input_shape, include_top=True, weights=None, classes=classes
    )(inputs)
    return models.Model(inputs, base_model)


def efficient_net_model_FT(input_shape, classes=7):
    """Fully train EfficientNet model"""
    inputs = layers.Input(shape=input_shape)
    base_model = EfficientNetB0(
        input_shape=input_shape, include_top=True, weights=None, classes=classes
    )(inputs)
    return models.Model(inputs, base_model)
