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
# from tensorflow.keras.applications import (
#     xception,
#     vgg16,
#     vgg19,
#     resnet,
#     resnet50,
#     resnet_v2,
#     inception_v3,
#     inception_resnet_v2,
#     mobilenet,
#     densenet,
#     nasnet,
#     mobilenet_v2,
#     efficientnet,
#     efficientnet_v2,
#     mobilenet_v3,
#     MobileNetV3Small,
#     MobileNetV3Large,
# )
# from keras import backend as K
from tensorflow.keras.preprocessing import image
# from concurrent import futures
from itertools import compress
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

input_tensor = Input(shape=(224, 224, 3))

model_types = [
    # 'xception',
    'vgg16',
    'vgg19',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnet50_v2',
    'resnet101_v2',
    'resnet152_v2',
    'inception_v3',
    'inception_resnet_v2',
    'mobilenet',
    'densenet121',
    'densenet169',
    'densenet201',
    'nasnetmobile',
    'nasnetlarge',
    'mobilenet_v2'
    'efficientnetb0',
    'efficientnetb1',
    'efficientnetb2',
    'efficientnetb3',
    'efficientnetb4',
    'efficientnetb5',
    'efficientnetb6',
    'efficientnetb7',
    'efficientnet_v2b0',
    'efficientnet_v2b1',
    'efficientnet_v2b2',
    'efficientnet_v2b3',
    'efficientnet_v2l',
    'efficientnet_v2m',
    'efficientnet_v2s',
    'mobilenet_v3small',
    'mobilenet_v3large',
]

def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# .... Define your model here ....
# You need to have compiled your model before calling this.

for model_type in model_types:
    mtype = model_type
    model = load_model(model_type + '_saved_model')
    print(model_type)
    print(get_flops())
