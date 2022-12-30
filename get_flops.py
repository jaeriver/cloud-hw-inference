import tensorflow as tf
import keras.backend as K
import os
import time
import shutil
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras

# from keras import backend as K
from tensorflow.keras.preprocessing import image
# from concurrent import futures
from itertools import compress
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

input_tensor = Input(shape=(224, 224, 3))

from tensorflow.keras.applications import (
    xception,
    vgg16,
    vgg19,
    resnet,
    resnet50,
    resnet_v2,
    inception_v3,
    inception_resnet_v2,
    mobilenet,
    densenet,
    nasnet,
    mobilenet_v2,
    efficientnet,
    efficientnet_v2,
    mobilenet_v3,
    MobileNetV3Small,
    MobileNetV3Large,
)

models_detail = {
    'xception':xception.Xception(weights='imagenet'),
    'vgg16':vgg16.VGG16(weights='imagenet'),
    'resnet50':resnet50.ResNet50(weights='imagenet'),
    'resnet101':resnet.ResNet101(weights='imagenet'),
    'resnet152':resnet.ResNet152(weights='imagenet'),
    'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
    'inception_v3':inception_v3.InceptionV3(weights='imagenet'),
    'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
    'densenet121':densenet.DenseNet121(weights='imagenet'),
    'densenet169':densenet.DenseNet169(weights='imagenet'),
    'densenet201':densenet.DenseNet201(weights='imagenet'),
    'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
    'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
    'efficientnetb0':efficientnet.EfficientNetB0(weights='imagenet'),
    'efficientnetb1':efficientnet.EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb2':efficientnet.EfficientNetB2(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb3':efficientnet.EfficientNetB3(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb4':efficientnet.EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb5':efficientnet.EfficientNetB5(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb6':efficientnet.EfficientNetB6(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnetb7':efficientnet.EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2b0':efficientnet_v2.EfficientNetV2B0(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2b1':efficientnet_v2.EfficientNetV2B1(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2b2':efficientnet_v2.EfficientNetV2B2(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2b3':efficientnet_v2.EfficientNetV2B3(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2l':efficientnet_v2.EfficientNetV2L(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2m':efficientnet_v2.EfficientNetV2M(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'efficientnet_v2s':efficientnet_v2.EfficientNetV2S(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'mobilenet_v3small':MobileNetV3Small(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'mobilenet_v3large':MobileNetV3Large(input_tensor=input_tensor, weights='imagenet', include_top=True),
    'vgg19':vgg19.VGG19(weights='imagenet'),
    'nasnetlarge':nasnet.NASNetLarge(input_tensor=input_tensor, weights='imagenet', include_top=True),
}

model_types = [key for key, value in models_detail.items()]


def get_flops():
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# .... Define your model here ....
# You need to have compiled your model before calling this.

for model_type in model_types:
    model = models_detail[model_type]
    print(model_type)
    print(get_flops())
