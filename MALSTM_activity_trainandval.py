from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import StratifiedShuffleSplit
from os import listdir
from numpy import genfromtxt
from itertools import chain
import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, n_classes = 9):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(range(n_classes)))
    plt.xticks(tick_marks, range(n_classes), rotation=45)
    plt.yticks(tick_marks, range(n_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def CM(cm, acc_score):
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig2 = plt.figure(figsize = (8,8))
    plot_confusion_matrix(cm_normalized, title="Matriz de confusión normalizada", n_classes = 9)
    plt.savefig("Confusion_matrix.png")
    plt.close(fig2)

def not_in_sequence(i,j,SIS):
    for index in range(len(SIS)-1):
        if((SIS[index] == i and SIS[index+1] == j) or (SIS[index] == j and SIS[index+1] == i)):
            return False
    return True

def signal_image(signals):
    SI = [signals[:,0]] #Signal Image, collection of sequences
    SIS = [0] #Signal index string, keeps record of the order of added sequences to SI
    Ns = signals.shape[1] #Total number of different sequences
    Nsis = 1 #Number of added sequences
    i = 0
    j = 1
    while(i != j):
        if(j > Ns-1):
            j = 0
        elif(not_in_sequence(i,j,SIS)):
            SI.append(signals[:,j])
            SIS.append(j)
            Nsis = Nsis + 1
            i = j
            j = i + 1
        else:
            j = j + 1
    return SI[0:len(SI)-1], SIS, Nsis, Ns

def signal_images(input):
    Signal_images = []
    for i in range(input.shape[0]):
        Signal_images.append(signal_image(input[i,:,:])[0])
    return np.array(Signal_images, dtype = np.float32)

#Selects K items from a set at random
#can be used to construct smaller batches.
def selectKitems(X, y, k):
    selected_items = np.zeros((k,X.shape[1], X.shape[2]))
    selected_labels = np.zeros((k,y.shape[1]))
    for i in range(k):
        selected_items[i] = X[i]
        selected_labels[i] = y[i]
    for i in range(k,len(X)):
        rand = random.randint(0,i+1)
        if(rand < k):
            selected_items[rand] = X[i]
            selected_labels[rand] = y[i]
    return selected_items, selected_labels

def get_file_names(path):
    folder = path + 'S0%d'
    filenames = []
    for x in range(1,10):
        current_folder = folder % x
        onlyfiles = [current_folder + "/" + f for f in listdir(current_folder)]
        filenames.append(sorted(onlyfiles))
    return filenames

def conversion(label):
    if(label == 'bike'):
        return 0
    elif(label == 'climbing'):
        return 1
    elif(label == 'descending'):
        return 2
    elif(label == 'gymbike'):
        return 3
    elif(label == 'jumping'):
        return 4
    elif(label == 'running'):
        return 5
    elif(label == 'standing'):
        return 6
    elif(label == 'treadmill'):
        return 7
    elif(label == 'walking'):
        return 8

def group_0_conversion(label):
    if(label == 'bike'):
        return 0
    elif(label == 'descending'):
        return 1
    elif(label == 'gymbike'):
        return 2

def group_1_conversion(label):
    if(label == 'jumping'):
        return 0
    elif(label == 'treadmill'):
        return 1
    elif(label == 'climbing'):
        return 2

def group_2_conversion(label):
    if(label == 'running'):
        return 0
    elif(label == 'walking'):
        return 1
    elif(label == 'standing'):
        return 2

def group_conversion(label):
    if(label == 'gymbike' or label == 'bike' or label == 'descending'):
        return 0
    if(label == 'climbing' or label == 'jumping'  or label == 'treadmill'):
        return 1
    if(label == 'running' or label == 'walking' or label == 'standing'):
        return 2

def one_hot_encoding(label, classes):
    r = np.zeros(classes)
    r[label] = 1
    return r

def conversion_one_hot(labels, classes):
    r = np.zeros((len(labels),classes))
    for i in range(len(labels)):
        r[i] = one_hot_encoding(labels[i], classes)
    return r

def make_dataset(full_path, seed):
    full_file_names = get_file_names(full_path)
    X = []
    X_flattened = []
    X_flattened_6 = []
    y_names = []
    y_numbers = []
    for folder_number in range(9):
        for filename in range(len(full_file_names[folder_number])):
            path = full_file_names[folder_number][filename]
            sample_data = genfromtxt(path, delimiter=',')
            X.append(sample_data)
            l = [x[0:6] for x in sample_data]
            X_flattened_6.append(list(chain.from_iterable(l)))
            X_flattened.append(list(chain.from_iterable(sample_data)))
            label = path.split('/')
            label = label[len(label)-1].split('.')[0]
            label = label[0:len(label)-1]
            y_names.append(label)
            y_numbers.append(conversion(label))

    X_to_use = X_flattened
    y_numbers = conversion_one_hot(y_numbers)
    sss = StratifiedShuffleSplit(n_splits=1,test_size = 0.4, random_state = seed)
    sss.get_n_splits(X_to_use,y_numbers)
    for train_index, test_index in sss.split(X_to_use,y_numbers):
        train_X, test_X = np.array(X_to_use, dtype = np.float32)[train_index], np.array(X_to_use, dtype = np.float32)[test_index]
        train_y, test_y = np.array(y_numbers, dtype = np.float32)[train_index], np.array(y_numbers, dtype= np.float32)[test_index]

    ssss = StratifiedShuffleSplit(n_splits=1,test_size = 0.5, random_state = seed)
    ssss.get_n_splits(test_X, test_y)
    for test_index, validate_index in ssss.split(test_X,test_y):
        test_X, validate_X = test_X[test_index], test_X[validate_index]
        test_y, validate_y = test_y[test_index], test_y[validate_index]

    print("train size: ", len(train_X))
    print("validate size: ", len(validate_X))
    print("test_ size: ", len(test_X))
    return train_X, validate_X, test_X, train_y, validate_y, test_y

def split(X, y, ratio, seed):
    ssss = StratifiedShuffleSplit(n_splits=1, test_size = ratio, random_state = seed)
    ssss.get_n_splits(X, y)
    for train_index, test_index in ssss.split(X,y):
        train_X, test_X = np.array(X, dtype = np.float32)[train_index], np.array(X, dtype = np.float32)[test_index]
        train_y, test_y = np.array(y, dtype = np.float32)[train_index], np.array(y, dtype= np.float32)[test_index]
    return train_X, test_X, train_y, test_y

#Creates dataset from files.
#For curiosity's sake also separates them in groups, as defined above.
def make_dataset_groups(full_path, seed):
    full_file_names = get_file_names(full_path)
    X = []
    X_flattened = []
    X_flattened_6 = []
    y_names = []
    y_numbers = []
    y_groups = []

    X_group_0 = []
    y_group_0 = []
    X_group_1 = []
    y_group_1 = []
    X_group_2 = []
    y_group_2 = []
    for folder_number in range(9):
        for filename in range(len(full_file_names[folder_number])):
            path = full_file_names[folder_number][filename]
            sample_data = genfromtxt(path, delimiter=',')
            X.append(sample_data)
            l = [x[0:6] for x in sample_data]
            X_flattened_6.append(list(chain.from_iterable(l)))
            X_flattened.append(list(chain.from_iterable(sample_data)))
            label = path.split('/')
            label = label[len(label)-1].split('.')[0]
            label = label[0:len(label)-1]
            y_names.append(label)
            y_numbers.append(conversion(label))
            y_groups.append(group_conversion(label))
            if(label == 'gymbike' or label == 'bike' or label == 'descending'):
                X_group_0.append(list(chain.from_iterable(sample_data)))
                y_group_0.append(group_0_conversion(label))
            elif(label == 'climbing' or label == 'jumping'  or label == 'treadmill'):
                X_group_1.append(list(chain.from_iterable(sample_data)))
                y_group_1.append(group_1_conversion(label))
            elif(label == 'running' or label == 'walking' or label == 'standing'):
                X_group_2.append(list(chain.from_iterable(sample_data)))
                y_group_2.append(group_2_conversion(label))

    X_to_use = X_flattened
    y_numbers = conversion_one_hot(y_numbers, 9)
    y_groups = conversion_one_hot(y_groups, 3)
    y_group_0 = conversion_one_hot(y_group_0, 3)
    y_group_1 = conversion_one_hot(y_group_1, 3)
    y_group_2 = conversion_one_hot(y_group_2, 3)
    sss = StratifiedShuffleSplit(n_splits=1,test_size = 0.4, random_state = seed)
    sss.get_n_splits(X_to_use,y_numbers)
    for train_index, test_index in sss.split(X_to_use,y_numbers):
        train_X, test_X = np.array(X_to_use, dtype = np.float32)[train_index], np.array(X_to_use, dtype = np.float32)[test_index]
        train_y, test_y = np.array(y_numbers, dtype = np.float32)[train_index], np.array(y_numbers, dtype= np.float32)[test_index]
        train_y_groups, test_y_groups = np.array(y_groups, dtype = np.float32)[train_index], np.array(y_groups, dtype= np.float32)[test_index]

    sss = StratifiedShuffleSplit(n_splits=1,test_size = 0.5, random_state = seed)
    sss.get_n_splits(test_X,test_y)
    for train_index, test_index in sss.split(test_X, test_y):
        validate_X, test_X = np.array(test_X, dtype = np.float32)[train_index], np.array(test_X, dtype = np.float32)[test_index]
        validate_y, test_y = np.array(test_y, dtype = np.float32)[train_index], np.array(test_y, dtype= np.float32)[test_index]
        validate_y_groups, test_y_groups = np.array(test_y_groups, dtype = np.float32)[train_index], np.array(test_y_groups, dtype= np.float32)[test_index]

    train_X_0, test_X_0, train_y_0, test_y_0 = split(X_group_0, y_group_0, 0.4, seed)
    train_X_1, test_X_1, train_y_1, test_y_1 = split(X_group_1, y_group_1, 0.4, seed)
    train_X_2, test_X_2, train_y_2, test_y_2 = split(X_group_2, y_group_2, 0.4, seed)

    validate_X_0, test_X_0, validate_y_0, test_y_0 = split(test_X_0, test_y_0, 0.5, seed)
    validate_X_1, test_X_1, validate_y_1, test_y_1 = split(test_X_1, test_y_1, 0.5, seed)
    validate_X_2, test_X_2, validate_y_2, test_y_2 = split(test_X_2, test_y_2, 0.5, seed)

    return train_X_0, validate_X_0, test_X_0, train_y_0, validate_y_0, test_y_0, train_X_1, validate_X_1, test_X_1, train_y_1, validate_y_1, test_y_1,train_X_2, validate_X_2, test_X_2, train_y_2, validate_y_2, test_y_2,train_X, validate_X, test_X, train_y, validate_y, test_y, train_y_groups, validate_y_groups, test_y_groups


# Training Parameters
learning_rate = 0.0001
training_steps = 10000
display_step = 1

# Network Parameters
timesteps = 500 # timesteps
num_hidden = 635 # hidden layer num of features in LSTM
is_training = True # Bool for Batch normalization

full_path = '/home/mrlz/Desktop/proj_int/Smartphone_Dataset/'
train_X_0, validate_X_0, test_X_0, train_y_0, validate_y_0, test_y_0, train_X_1, validate_X_1, test_X_1, train_y_1, validate_y_1, test_y_1,train_X_2, validate_X_2, test_X_2, train_y_2, validate_y_2, test_y_2,train_X, validate_X, test_X, train_y, validate_y, test_y, train_y_groups, validate_y_groups, test_y_groups = make_dataset_groups(full_path, 1727465)
#Standard Scaling the samples on a per channel basis yields faster
#convergence times for the network.
scaler = preprocessing.StandardScaler()

train_X = train_X
test_X = test_X
validate_X = validate_X

train_X = train_X.reshape((train_X.shape[0], 500, 9))
test_X = test_X.reshape((test_X.shape[0], 500, 9))
validate_X = validate_X.reshape((validate_X.shape[0], 500, 9))

#Here we can choose starting and ending channel, currently using them
#all leads to better results
start_index = 0
end_index = 9
train_X = train_X[:,:,start_index:end_index].reshape((train_X.shape[0], 500, end_index - start_index))
test_X = test_X[:,:,start_index:end_index].reshape((test_X.shape[0], 500, end_index - start_index))
validate_X = validate_X[:,:,start_index:end_index].reshape((validate_X.shape[0], 500, end_index - start_index))

print(train_X.shape)

train_y = train_y
validate_y = validate_y
test_y = test_y

batch_size = 229
num_classes = train_y.shape[1]

num_input = train_X.shape[2]
train_X = train_X.reshape((len(train_X), timesteps, num_input))
validate_X = validate_X.reshape((len(validate_X), timesteps, num_input))
test_X = test_X.reshape((len(test_X), timesteps, num_input))

scalers = []
for i in range(train_X.shape[2]):
    scalers.append(preprocessing.StandardScaler())
    train_X[:, :, i] = scalers[i].fit_transform(train_X[:, :, i])
    test_X[:, :, i] = scalers[i].transform(test_X[:, :, i])
    validate_X[:, :, i] = scalers[i].transform(validate_X[:, :, i])

#We compute the signal images, upon which fft will be applied at
#train time. The reason it is not pre-applied is because
#we will add noise to the samples to compensate for the small dataset.
train_X = signal_images(train_X)
validate_X = signal_images(validate_X)
test_X = signal_images(test_X)

#We use the full non-test set to train
train_X = np.concatenate((train_X, validate_X), axis = 0)
train_y = np.concatenate((train_y, validate_y), axis = 0)
#We train with all the simples each step
batch_size = train_X.shape[0]

num_input = train_X.shape[2]

# We define the placeholders to assemble the architecture
X = tf.placeholder("float", [None, train_X.shape[1], train_X.shape[2]])
Y = tf.placeholder("float", [None, num_classes])
noise_on = tf.placeholder(tf.bool)
is_training = tf.placeholder(tf.bool)

# Define weights for LSTM output layer
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases, noise_on, is_training):
    noise = tf.random_normal(shape = tf.shape(x), mean = 0.0, stddev = 1.0, dtype = tf.float32)
    x = tf.cond(tf.equal(noise_on, tf.constant(True)), lambda: tf.add(x,noise), lambda: x )
    x = tf.spectral.rfft2d(x)
    x = tf.abs(x)

    #Conditionally add noise
    # noise = tf.random_normal(shape = tf.shape(x), mean = 0.0, stddev = 1.0, dtype = tf.float32)
    # x = tf.cond(tf.equal(noise_on, tf.constant(True)), lambda: tf.add(x,noise), lambda: x )

    #Reshape to feed convolutional layer, (batch, Action image stacks, FFT channels, 1 channel)
    x = tf.reshape(x, shape = [-1, 36, 251, 1])
    conv1 = tf.layers.conv2d(x, 5, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling())

    #First squeeze and excite block
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    print(conv1.get_shape().as_list())
    filters = conv1.get_shape().as_list()[-1]
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(conv1)
    print(squeeze.get_shape().as_list())
    squeeze = tf.layers.dense(squeeze, filters//16, activation = tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling())
    squeeze = tf.layers.dense(squeeze, filters, activation = tf.nn.sigmoid, kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv1_pool = tf.multiply(conv1, tf.reshape(squeeze, shape = [-1, 1, 1, filters]))

    #Second convolutional layer
    conv2 = tf.layers.conv2d(conv1_pool, 10, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling())
    # conv2_pool = tf.layers.average_pooling2d(conv2, 2, 2)

    #Second squeeze and excite block
    conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    filters = conv2.get_shape().as_list()[-1]
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(conv2)
    squeeze = tf.layers.dense(squeeze, filters//16, activation = tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling())
    squeeze = tf.layers.dense(squeeze, filters, activation = tf.nn.sigmoid, kernel_initializer = tf.contrib.layers.xavier_initializer())
    conv2_pool = tf.multiply(conv2, tf.reshape(squeeze, shape = [-1, 1, 1, filters]))

    stride = 3
    #Define LSTM cell with dropout (0.2 chance of dropping) and Attention (window size = 10)
    AttentionLSTM = tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden),output_keep_prob=0.8), 10)
    #Initialize LSTM inner state (C_i)
    initial_state_1 = AttentionLSTM.zero_state(batch_size, tf.float32)

    #Reshape and unstack signal in FFT channel sequence (approx to time sequence)
    x_lstm = tf.reshape(x, shape = [-1, 36, 251])
    x_lstm = tf.unstack(x_lstm, 251, 2)
    outputs_2, states_2 = rnn.static_rnn(AttentionLSTM, x_lstm, dtype=tf.float32)

    #Flatten output of convolutional path
    conv2_pool = tf.contrib.layers.flatten(conv2_pool)
    #Compute LSTM output
    x_lstm = tf.matmul(outputs_2[-1], weights['out']) + biases['out']
    #Combine LSTM output with convolutional path output
    x_out = tf.concat([x_lstm,conv2_pool], 1)
    #Combine results in dense layer with 1 neuron per class
    return tf.layers.dense(x_out, num_classes, kernel_initializer = tf.contrib.layers.xavier_initializer())


#Output layer of network
logits = RNN(X, weights, biases, noise_on, is_training)
#Softmax of output layer
prediction = tf.nn.softmax(logits)
#Outputs the class probability (by softmax) of most likely label
#which is used for unsupervised learning
best_two = tf.nn.top_k(prediction, k = 1)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
#Define optimizer to be Adam with 0.0001 learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

max_grad_norm = 5.0
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ##
with tf.control_dependencies(update_ops):#
    tvars = tf.trainable_variables()
    #Compute the gradients directly to apply gradient clipping
    grads_and_vars = optimizer.compute_gradients(loss_op, tvars)
    grads, _ = tf.clip_by_global_norm([x[0] for x in grads_and_vars], max_grad_norm)
    gradients = zip(grads, tvars)
    train_op = optimizer.apply_gradients(gradients)

# Evaluate model (with test logits, for dropout to be disabled)
prediction_labels = tf.argmax(prediction, 1)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


activate_noise = False
with tf.Session() as sess:
    sess.run(init)
    X_plot = []
    y_plot_train = []
    y_plot_validate = []
    y_plot_test = []
    for step in range(1, training_steps+1):
        if(step == 4):
            #Initially we feed noiseless batches because networks
            #tend to be very susceptible to the weights they learn
            #at the beginning of their training
            print("Noise activated")
            activate_noise = True
        sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: activate_noise})
        if step % display_step == 0 or step == 1:
            X_plot.append(step)
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: False})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            y_plot_train.append(acc)
            val_acc = sess.run(accuracy, feed_dict={X: validate_X, Y: validate_y, is_training: True, noise_on: False})
            print(" Validation Accuracy:", val_acc)
            y_plot_validate.append(val_acc)
            test_acc = sess.run(accuracy, feed_dict={X: test_X, Y: test_y, is_training: True, noise_on: False})
            y_plot_test.append(test_acc)
            if(loss <= 0.002):
                print("Desired error reached")
                #When validation set was being used to choose model parameters
                #an unsupervised learning stage could be used to improve results.

                # print("Unsupervised learning")
                # while(len(validate_X) > 0):
                #     print("Validation set remaining items:", len(validate_X))
                #     best = sess.run([best_two], feed_dict={X: validate_X, Y: validate_y, is_training: False, noise_on: False})
                #     best_index = np.argmax(best[0][0])
                #     if(best[0][0][best_index] >= 0.999):
                #         print(" Selected!", best[0][0][best_index])
                #         train_X = np.concatenate((train_X, [validate_X[best_index]]), axis = 0)
                #         train_y = np.concatenate((train_y, [one_hot_encoding(best[0][1][best_index][0],num_classes)]), axis = 0)
                #         batch_size = batch_size + 1
                #         # for iters in range(3):
                #         # sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: activate_noise})
                #     # elif(loss > 0.001):
                #         # loss = sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: False, noise_on: activate_noise})
                #     validate_X = np.delete(validate_X, best_index, 0)
                #     validate_y = np.delete(validate_y, best_index, 0)
                #     # loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_X, Y: train_y, is_training: False, noise_on: False})
                #     # print("Step " + str(step) + ", Minibatch Loss= " + \
                #     #       "{:.4f}".format(loss) + ", Training Accuracy= " + \
                #     #       "{:.3f}".format(acc))
                #
                # unsup = 0
                # while(loss > 0.0015 or unsup < 10):
                #     print(" Unsupervised training at loss ", loss)
                #     sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: activate_noise})
                #     loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_X, Y: train_y, is_training: False, noise_on: False})
                #     unsup = unsup + 1
                # best_index = np.argmax(best['values'])
                # print(best_index)
                # print(best['indices'][best_index])
                # print
                # print(prediction[best[0]])
                # print(validate_y[best[0]])
                # print(prediction[best[1]])
                # print(validate_y[best[1]])
                break

    #Little accuracy plot
    fig, ax = plt.subplots(figsize=(40,20))
    ax.plot(X_plot, y_plot_train)
    ax.plot(X_plot, y_plot_validate)
    ax.plot(X_plot, y_plot_test)
    ax.legend(['train', 'validate', 'test'], loc = 'lower right')
    ax.set(xlabel='Epochs', ylabel='Accuracy',
           title='Accuracy vs Epochs')
    ax.grid()

    fig.savefig("MALSTM_activity_fullset.png")
    plt.show()
    print("Optimization Finished!")

    print("bn, no noise")
    print("Training Accuracy:", \
        sess.run(accuracy, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: False}))

    print("Validation Accuracy:", \
        sess.run(accuracy, feed_dict={X: validate_X, Y: validate_y, is_training: True, noise_on: False}))

    accu = sess.run(accuracy, feed_dict={X: test_X, Y: test_y, is_training: True, noise_on: False})
    print("Testing Accuracy:", accu)


    #Print confusion matrix for test set
    labelspred = sess.run(prediction_labels, feed_dict={X: test_X, Y: test_y, is_training: True, noise_on: False})
    # print(labelspred)
    # print(np.argmax(test_y, axis = 1))
    conf = confusion_matrix(np.ravel(np.argmax(test_y, axis = 1)), np.ravel(labelspred))
    print(conf)
    CM(conf, accu)

    #For further curiosity's sake, unsupervised learning over
    #testing set to see how it modifies results.
    # test_X_original = test_X
    # test_y_original = test_y
    # while(len(test_X) > 0):
    #     print("test set remaining items:", len(test_X))
    #     best = sess.run([best_two], feed_dict={X: test_X, Y: test_y, is_training: True, noise_on: False})
    #     best_index = np.argmax(best[0][0])
    #     if(best[0][0][best_index] >= 0.999):
    #         print(" Selected!", best[0][0][best_index])
    #         train_X = np.concatenate((train_X, [test_X[best_index]]), axis = 0)
    #         train_y = np.concatenate((train_y, [one_hot_encoding(best[0][1][best_index][0],num_classes)]), axis = 0)
    #         batch_size = batch_size + 1
    #         sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: True})
    #     elif(loss > 0.015):
    #         loss = sess.run(train_op, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: True})
    #     test_X = np.delete(test_X, best_index, 0)
    #     test_y = np.delete(test_y, best_index, 0)
    #     loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: False})
    #     print("Step " + str(step) + ", Minibatch Loss= " + \
    #           "{:.4f}".format(loss) + ", Training Accuracy= " + \
    #           "{:.3f}".format(acc))
    #
    #
    # print("Training Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: train_X, Y: train_y, is_training: True, noise_on: False}))
    #
    # print("Validation Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: validate_X_original, Y: validate_y_original, is_training: True, noise_on: False}))
    #
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: test_X_original, Y: test_y_original, is_training: True, noise_on: False}))
