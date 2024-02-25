from keras import layers, models
from keras.applications import EfficientNetB0, DenseNet121
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16


def dense_net_model(input_shape, classes=7):
    """DenseNet model"""
    inputs = layers.Input(shape=input_shape)
    base_model = DenseNet121(input_shape=input_shape, include_top=True, weights=None, classes=len(classes))(inputs)
    # base_model.trainable = False
    model = models.Model(inputs, base_model)
    return model

def vgg16_model(input_shape, classes=7):
    """VGG16 model"""
    inputs = layers.Input(shape=input_shape)
    base_model = VGG16(input_shape=input_shape, include_top=True, weights=None, classes=len(classes))(inputs)
    # base_model.trainable = False
    model = models.Model(inputs, base_model)
    return model

def efficient_net_model(input_shape, classes=7):
    """EfficientNet model"""
    inputs = layers.Input(shape=input_shape)
    base_model = EfficientNetB0(input_shape=input_shape, include_top=True, weights=None, classes=len(classes))(inputs)
    # base_model.trainable = False
    model = models.Model(inputs, base_model)
    return model

def compile_model(model, optimizer, loss, metrics):
    """Compiles model"""
    model.compile(optimizer=optimizer, 
                    loss=loss, 
                    metrics=metrics,
                    weighted_metrics=None,
                    run_eagerly=None,
                    steps_per_execution=None,
                    jit_compile=None)
    return model