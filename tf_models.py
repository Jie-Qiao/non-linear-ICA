from signals import *
import numpy as np
#import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
from keras.layers import Dense, maximum

def abs_activation(x):
    return np.abs(x)

get_custom_objects().update({'abs_activation': Activation(abs_activation)})



def logistic_model(n_sources,n_layers_feature,feature_layer_size,n_layers_psi,psi_layer_size):
    ### feature extractor ###
    x = keras.layers.Input(shape=(n_sources,))
    u = keras.layers.Input(shape=(n_sources,))

    init_bound = np.sqrt(6)/np.sqrt(2*feature_layer_size)
    initializer = keras.initializers.RandomUniform(minval=-init_bound, maxval=init_bound)

    hx = maximum([Dense(feature_layer_size,kernel_initializer=initializer)(x) for _ in range(2)])
    for _ in range(n_layers_feature-1):
        hx = maximum([Dense(feature_layer_size,kernel_initializer=initializer)(hx) for _ in range(2)])
    features = keras.layers.Dense(n_sources, activation=Activation(abs_activation),kernel_initializer='random_uniform')(hx)
    
    ### psi functions ##

    all_psi = []
    init_bound = np.sqrt(6)/np.sqrt(2*psi_layer_size)
    initializer = keras.initializers.RandomUniform(minval=-init_bound, maxval=init_bound)
    for i in range(n_sources):
        hi = keras.layers.Lambda(lambda x: x[:,i])(features)
        hi = keras.layers.Reshape([1])(hi)
        input = keras.layers.Concatenate(axis=1)([hi,u])
        z = keras.layers.Dense(psi_layer_size, activation='relu',kernel_initializer=initializer)(input)
        for _ in range(n_layers_psi-1):
            psi = keras.layers.Dense(psi_layer_size, activation='relu',kernel_initializer=initializer)(z)
        out = keras.layers.Dense(1, activation='relu',kernel_initializer=initializer)(psi)
        all_psi.append(out)
    r = keras.layers.Add()(all_psi)

    probas = Activation('sigmoid')(r)
    print(probas)

    model = keras.Model(inputs=[x,u], outputs=probas)
    feature_extractor = theano.function([model.get_input(train=False)], features.get_output(train=False))
    return model,feature_extractor


