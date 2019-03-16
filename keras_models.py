from signals import *
import numpy as np
#import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
from keras import regularizers
from keras.layers import Dense, maximum
np.random.seed(200)

def abs_activation(x):
    return np.abs(x)

get_custom_objects().update({'abs_activation': Activation(abs_activation)})



def logistic_model(n_sources,n_layers_feature,feature_layer_size,n_layers_psi,psi_layer_size,regularization_coeff=0):
    ### feature extractor ###
    x = keras.layers.Input(shape=(n_sources,))
    u = keras.layers.Input(shape=(n_sources,))

    init_bound = np.sqrt(6)/np.sqrt(2*feature_layer_size)
    initializer = keras.initializers.RandomUniform(minval=-init_bound, maxval=init_bound)

    hx = maximum([Dense(feature_layer_size,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(regularization_coeff))(x) for _ in range(2)])
    #hx = Dense(feature_layer_size,kernel_initializer=initializer,activation='tanh')(x)
    for _ in range(n_layers_feature):
        #hx = Dense(feature_layer_size,kernel_initializer=initializer)(x)
        hx = maximum([Dense(feature_layer_size,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(regularization_coeff))(hx) for _ in range(2)])
    features = keras.layers.Dense(n_sources, kernel_initializer='random_uniform',name='features',kernel_regularizer=regularizers.l2(regularization_coeff))(hx)
    
    ### psi functions ##

    all_psi = []
    init_bound = np.sqrt(6)/np.sqrt(2*psi_layer_size)
    initializer = keras.initializers.RandomUniform(minval=-init_bound, maxval=init_bound)
    for i in range(n_sources):
        hi = keras.layers.Lambda(lambda x: x[:,i])(features)
        hi = keras.layers.Reshape([1])(hi)
        psi = keras.layers.Concatenate(axis=1)([hi,u])
        #z = maximum([Dense(psi_layer_size,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(regularization_coeff))(input) for _ in range(2)])
        for _ in range(n_layers_psi):
            psi = maximum([Dense(psi_layer_size,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(regularization_coeff))(psi) for _ in range(2)])
        out = maximum([Dense(1,kernel_initializer=initializer,kernel_regularizer=regularizers.l2(regularization_coeff))(psi) for _ in range(2)])
        all_psi.append(out)
    r = keras.layers.Add()(all_psi)

    probas = Activation('sigmoid')(r)

    model = keras.Model(inputs=[x,u], outputs=probas)
    return model


