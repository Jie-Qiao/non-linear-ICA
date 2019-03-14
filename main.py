import numpy as np
from signals import *
from keras_models import *
from keract import get_activations
import sys
from sklearn.decomposition import FastICA
from keras.models import load_model
np.random.seed(200)
##
train_or_load = sys.argv[1] 

## Sources parameters ##
n_sources = 3
n_points = 10**3
AR_coef=0.7
step=0.5
sec = int(1/step)

## Mixing parameters ##
n_mixing_layers = 3
mixing_layer_size = 2*n_sources
leaky_slope = 0.7

## MLP parameters
n_




if train_or_load == 'train':

    ## Generating sources and mixtures
    sources = generate_AR_source(n_sources,n_points,AR_coef,step)
    mixtures = mix_sources(sources,n_mixing_layers,mixing_layer_size, leaky_slope)
    write_signal_to_disk(sources,step,filename='sources')
    write_signal_to_disk(mixtures,step,filename='mixtures')
    plot_signals(sources,step,name='sources')
    plot_signals(mixtures,step,name='mixtures')

    u = np.copy(mixtures)[:-sec]
    u_shuffled = np.copy(u)
    np.random.shuffle(u_shuffled)
    x = mixtures[sec:]


    ## Format to obtain training data and labels
    x_concat = np.concatenate([x,x])
    u_concat = np.concatenate([u,u_shuffled])
    labels = np.concatenate([np.ones(u.shape[0]),np.zeros(u_shuffled.shape[0])])

    shuffling_order = np.random.permutation(len(labels))

    x_concat = x_concat[shuffling_order,:]
    u_concat = u_concat[shuffling_order,:]
    ## Building and training our logistic model ##
    logistic = logistic_model(n_sources,n_layers_feature=n_mixing_layers,feature_layer_size=mixing_layer_size,n_layers_psi=n_mixing_layers,psi_layer_size=mixing_layer_size)
    logistic.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    logistic.fit(x = [x_concat,u_concat],y=labels, epochs=10000, batch_size= 100)

    logistic.save('models/logistic_model.h5')

elif train_or_load == 'load':
    sources,_ = load_signal_from_disk(filename='sources')
    mixtures,_ = load_signal_from_disk(filename='mixtures')
    u = np.copy(mixtures)[:-sec]
    x = mixtures[sec:]
    logistic = load_model('models/logistic_model.h5')
else:
    print("Please specify whether you want to train a new model or load an already existing one")

## Retrieving extracted features
activations = get_activations(logistic, [x,u])
for layer in activations:
    if 'feature' in layer:
        extracted_features = activations[layer]

## Apply FastICA
transformer = FastICA(n_components=n_sources,random_state=200)
transformed_extracted_features = transformer.fit_transform(extracted_features)
#plot_signals(transformed_extracted_features,step,name='extracted_features')
#transformed_extracted_features = extracted_features
## Compare the extracted features to original sources
print(transformed_extracted_features.shape,sources.shape)
h_u_pos = np.concatenate([transformed_extracted_features,sources[sec:]],axis=1)
h_u_neg = np.concatenate([transformed_extracted_features,-sources[sec:]],axis=1)
R_pos = np.corrcoef(h_u_pos,rowvar=False)[n_sources:,:n_sources]
R_neg = np.corrcoef(h_u_neg,rowvar=False)[n_sources:,:n_sources]

write_signal_to_disk(np.round(np.corrcoef(h_u_pos,rowvar=False),2),0,'correlation')

max_cor = np.maximum(np.amax(R_pos,axis=0),np.amax(R_neg,axis=0))
matching_pos = np.argmax(R_pos,axis=0)
matching_neg = np.argmax(R_neg,axis=0)

final_matching = [matching_pos[i] if R_pos[matching_pos[i],i]==max_cor[i] else matching_neg[i] for i in range(len(max_cor))]

i=0
for matching_source,cor in zip(final_matching,max_cor):
    i += 1
    print("Extracted feature "+str(i)+" correponds to source "+str(matching_source+1)+" with corr coef: "+str(cor))
