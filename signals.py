import numpy as np
import matplotlib.pyplot as plt
import keras
from  munkres import Munkres
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

def showtimedata(X, xlabel="Time", ylabel="Channel", fontsize=14, linewidth=1.5,
                 intervalstd=10, figsize=None):

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [2, 1]
    X = X.copy()
    X = X.reshape([X.shape[0],-1])

    if X.shape[1]==1:
        X = X.reshape([1,-1])

    Nch = X.shape[0]
    Nt = X.shape[1]

    vInterval = X.std(axis=1).max() * intervalstd
    vPos = vInterval * (np.arange(Nch,0,-1) - 1)
    vPos = vPos.reshape([1, -1]).T  # convert to column vector
    X = X + vPos

    # Plot ----------------------------------------------------
    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    for i in range(Nch):
        plt.plot(list(range(Nt)), X[i,:], linewidth=linewidth)

    plt.xlim(0, Nt-1)
    plt.ylim(X.min(),X.max())

    ylabels = [str(num) for num in range(Nch)]
    plt.yticks(vPos,ylabels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.rcParams["font.size"] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x,rowvar=False)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]
        
    print(corr)
    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort



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


def pca(x, num_comp=None, params=None, zerotolerance = 1e-7):
    """Apply PCA whitening to data.
    Args:
        x: data. 2D ndarray [num_comp, num_data]
        num_comp: number of components
        params: (option) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (option)
    Returns:
        x: whitened data
        parms: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    print("PCA...")

    # Dimension
    if num_comp is None:
        num_comp = x.shape[0]
    print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        print("    use learned value")
        data_pca = x - params['mean']
        x = np.dot(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = np.mean(x, 1).reshape([-1, 1])
        x = x - xmean

        # Eigenvalue decomposition
        xcov = np.cov(x)
        d, V = np.linalg.eigh(xcov)  # Ascending order
        # Convert to descending order
        d = d[::-1]
        V = V[:, ::-1]

        zeroeigval = np.sum((d[:num_comp] / d[0]) < zerotolerance)
        if zeroeigval > 0: # Do not allow zero eigenval
            raise ValueError

        # Calculate contribution ratio
        contratio = np.sum(d[:num_comp]) / np.sum(d)
        print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = np.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = np.dot(np.diag(dsqrtinv), V.transpose())  # whitening matrix
        A = np.dot(V, np.diag(dsqrt))  # de-whitening matrix
        x = np.dot(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = np.cov(x)

    return x.T


