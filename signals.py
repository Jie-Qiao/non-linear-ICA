import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K

def generate_AR_source(n_sources,n_points,AR_coef,step):

    """Generates an array of auto-regressive sources (n_points,n_sources)
    Each row of the table represents the values of all the sources at a specifc timestamp
    """
    s=np.zeros((n_points,n_sources)).astype('float32')
    sec = int(1/step)
    s[:sec,:] = np.random.randn(sec,n_sources)

    for j in range(sec,n_points):
        s[j,:] = np.random.laplace(loc=AR_coef*s[j-sec,:], scale=1.0)

    return s

def write_signal_to_disk(s,step,filename):

    f = open('data/'+filename +'.txt','w')
    timestamp = 0
    for sources_t in s:
        f.write(str(np.round(timestamp,1))+'\t'+'\t'.join([str(i) for i in sources_t])+'\n')
        timestamp += step
    f.close()

def load_signal_from_disk(filename):
    f = open('data/'+filename +'.txt','r')
    step = 0
    X = []
    for line in f.readlines():
        l = line.split('\t')
        X.append([float(i) for i in l[1:]])
        if step == 0:
            step = l[0]
    f.close()
    return np.array(X),step




def plot_signals(s,step,name=''):
    n_points,n_sources = s.shape
    times = np.arange(int(n_points))*step

    for j in range(n_sources):
        plt.plot(times,s[:,j])

    plt.xlabel('Time (s)')
    plt.ylabel('Signals')
    plt.title('Generated '+str(name))
    plt.savefig('plots/'+name+'.png')
    plt.clf()

def mixing_MLP(n_sources,n_layers,layer_size,leaky_slope):

    model = keras.Sequential()
    activation = keras.layers.LeakyReLU(alpha=leaky_slope)

    init_bound = np.sqrt(6)/np.sqrt(2*layer_size)
    initializer = keras.initializers.RandomUniform(minval=-init_bound, maxval=init_bound, seed=None)
    for _ in range(n_layers):
        model.add(keras.layers.Dense(layer_size, activation=activation,kernel_initializer=initializer))
    model.add(keras.layers.Dense(n_sources, activation=activation,kernel_initializer=initializer))
    return model

def mix_sources(s,n_layers,layer_size,leaky_slope):
    n_sources = s.shape[1]
    mixing_model = mixing_MLP(n_sources,n_layers,layer_size,leaky_slope)
    return mixing_model.predict(s)




