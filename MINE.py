import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.feature_selection import mutual_info_regression

# H=20
# n_epochs = 200
# data_size = 20000
def MINE(x_in, y_in, H=20):

    # shuffle and concatenate
    y_shuffle = tf.random_shuffle(y_in)
    x_conc = tf.concat([x_in, x_in], axis=0)
    y_conc = tf.concat([y_in, y_shuffle], axis=0)

    # propagate the forward pass
    layerx = layers.linear(x_conc, H)
    layery = layers.linear(y_conc, H)
    layer2 = tf.nn.relu(layerx + layery)
    output = layers.linear(layer2, 1)

    # split in T_xy and T_x_y predictions
    N_samples = tf.shape(x_in)[0]
    T_xy = output[:N_samples]
    T_x_y = output[N_samples:]

    # compute the negative loss (maximise loss == minimise -loss)
    neg_loss = -(tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y))))
    opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(neg_loss)

    return neg_loss, opt

def run_mine(extracted_features, n_sources=4, save=True, name="MINE"):
    MIs = np.zeros(( n_sources, n_sources ))
    for i in range(n_sources):
        for j in range(n_sources):
            if i != j:
                MIs[i,j] = run_individual_mine(extracted_features[:,0], extracted_features[:,1])


    if save:
        print(MIs)
        fig, ax = plt.subplots() # im = ax.imshow(harvest)
        ax.imshow(MIs)

        ax.set_xticks(np.arange(n_sources))
        ax.set_yticks(np.arange(n_sources))
        # ... and label them with the respective list entries
        ax.set_xticklabels(np.arange(n_sources))
        ax.set_yticklabels(np.arange(n_sources))

        for i in range(n_sources):
            for j in range(n_sources):
                ax.text(j, i, "{:.3f}".format(MIs[i, j]), ha="center",
                        va="center", color="b")

        ax.set_title("Estimated Mutual Information")
        fig.tight_layout()

        # plt.xlabel('Feature')
        # plt.ylabel('Feature')
        # plt.title('Estimated Mutual Information')
        plt.savefig('plots/'+name+'.png')
        plt.clf()
    return MIs

def run_individual_mine(x_sample, y_sample, n_epochs=200, data_size=20000, H=20, save=False):
    if x_sample.ndim == 1:
        x_sample = np.expand_dims(x_sample, 1)
    if y_sample.ndim == 1:
        y_sample = np.expand_dims(y_sample, 1)
    # prepare the placeholders for inputs
    x_in = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 1], name='y_in')

    # make the loss and optimisation graphs
    neg_loss, opt = MINE(x_in, y_in)

    # start the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # train
    MIs = []
    for epoch in range(n_epochs):
        # perform the training step
        feed_dict = {x_in:x_sample, y_in:y_sample}
        _, neg_l = sess.run([opt, neg_loss], feed_dict=feed_dict)

        # save the loss
        MIs.append(-neg_l)

    if save:
        fig, ax = plt.subplots()
        ax.plot(range(len(MIs)), MIs, label='MINE estimate')
        # ax.plot([0, len(MIs)], [mi,mi], label='True Mutual Information')
        ax.set_xlabel('training steps')
        ax.legend(loc='best')
        fig.savefig('plots/MINE.png')
    return np.array(MIs).max()
