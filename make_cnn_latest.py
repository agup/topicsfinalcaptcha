import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, AveragePooling2D
import sys
import numpy as np
import os
#import imageio
from keras.applications.resnet50 import ResNet50
from keras import Model
import string
from keras.layers import Dense, Activation
import tensorflow as T
from keras import backend as K
from boto3 import client
conn = client('s3')
#import imageio
import matplotlib.image as mpimg
import boto3
import io
from tqdm import tqdm
from io import BytesIO
from keras.optimizers import Adam
import tensorflow as tf

images = []
names = []

for fi in tqdm(os.listdir('data')):
    if fi[-3:] == 'jpg':
        names.append('data/' + fi)
        #images.append(mpimg.imread('data/' + fi))



#print(images)


lab_fil = 'data/labels.txt'

fi = open(lab_fil, 'r')

labels = []
for li in fi:
    lab = li.strip()
    labels.append(lab)


#print(labels)

inds = np.arange(0, len(labels))

#print(inds)
alphabet = list(string.ascii_uppercase) + list(string.digits)

alph_dict = {}
j = 0
for letter in alphabet:
    alph_dict[letter] = j
    j += 1
def batch_generator(images, labels, batch_size):
    i = 0
    X = []
    y = []
    np.random.shuffle(inds)
    while True:
        if i > len(inds) -1 :
            i = 0
            np.random.shuffle(inds)
        im = mpimg.imread(names[inds[i]])
        im = im[:, 100:250, :]
        #X.append(images[inds[i]])
        X.append(im)
        y_lab = labels[inds[i]]
        y_lab_vec = np.zeros((36,))
        y_lab_vec[alph_dict[y_lab[1]]] = 1
        #y_lab_vec[alph_dict[y_lab[1]] + 36] = 1
        #y_lab_vec[alph_dict[y_lab[2]] + 36 + 36] = 1
        #y_lab_vec[alph_dict[y_lab[3]] + 36 + 36 + 36] = 1
        y.append(y_lab_vec)
        i +=1
        if i % batch_size == 0:
            yield X, y
            X = []
            y = []



b_size = 32
gen = batch_generator(images, labels, b_size)
X, y = next(gen)

print(X)
print('y is ', len(y))

print('first entry of y is ', y[0])




def inference(x, drop_rate):
    with tf.variable_scope('hidden1'):
        conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden1 = dropout

    with tf.variable_scope('hidden2'):
        conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden2 = dropout

    with tf.variable_scope('hidden3'):
        conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden3 = dropout

    with tf.variable_scope('hidden4'):
        conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden4 = dropout

    with tf.variable_scope('hidden5'):
        conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden5 = dropout

    with tf.variable_scope('hidden6'):
        conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden6 = dropout

    with tf.variable_scope('hidden7'):
        conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden7 = dropout

    with tf.variable_scope('hidden8'):
        conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        hidden8 = dropout
        print('hidden 8', hidden8)
        flatten = tf.reshape(hidden8, [-1, 130*192])

    with tf.variable_scope('hidden9'):
        dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
        hidden9 = dense

    with tf.variable_scope('hidden10'):
        dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
        hidden10 = dense

    with tf.variable_scope('digit_length'):
        dense = tf.layers.dense(hidden10, units=7)
        length = dense

    with tf.variable_scope('digit1'):
        dense = tf.layers.dense(hidden10, units=36)
        digit1 = dense
    '''
    with tf.variable_scope('digit2'):
        dense = tf.layers.dense(hidden10, units=11)
        digit2 = dense

    with tf.variable_scope('digit3'):
        dense = tf.layers.dense(hidden10, units=11)
        digit3 = dense

    with tf.variable_scope('digit4'):
        dense = tf.layers.dense(hidden10, units=11)
        digit4 = dense

    with tf.variable_scope('digit5'):
        dense = tf.layers.dense(hidden10, units=11)
        digit5 = dense
     length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
     return length_logits, digits_logits
    '''
    return digit1 
    
def loss( digits_logits,  digits_labels):
    #length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
    digit1_cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=digits_labels, logits=digits_logits))
    #digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
    #digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
    #digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
    #digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
    loss = digit1_cross_entropy #length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    return loss



imagepl = tf.placeholder(tf.float32, [None, 200, 150, 3])
labelspl = tf.placeholder(tf.int32, [None, 36])


dl  = inference(imagepl, 0.2)


los = loss(dl, labelspl)


#acc, acc_op = tf.metrics.accuracy( tf.argmax( labelspl, 0) , tf.argmax( dl, 0))
arg_label = tf.argmax( labelspl, 1)
print('arg_laebl', arg_label)
sub_val = tf.subtract(arg_label , tf.argmax( dl, 1))
acc_val = tf.count_nonzero(sub_val)

opt = tf.train.GradientDescentOptimizer(0.001)

train_op  = opt.minimize(los)
accs = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.local_variables_initializer().run(session = sess)
saver = tf.train.Saver()
saver.restore(sess, '/home/ubuntu/latest_cnn3')
for i in range(0, 10000000):
    print('iter', i)
    X, y = next(gen)
    X = np.reshape(X, (b_size, 200, 150, 3))
    y = np.reshape(y, ( b_size, 36 ))
    feed_dict = {imagepl : X, labelspl :y }
    a, lo, t, s = sess.run([acc_val,  los, train_op, sub_val], feed_dict = feed_dict)
    if i %100 == 0:
        np.save('acccu.npy', accs)
        saver.save(sess, '/home/ubuntu/cnn_sec_dig')
    print(lo, '   ', 1.0 - (1.0*a/b_size))
    accs.append(a)
    #print(s)
    sys.stdout.flush()


'''


opt = Adam(lr = 0.0005)
model.compile(optimizer = opt, loss = 'categorical_crossentropy' , metrics = ['accuracy'])
losses = []
for i in range(0, 100000000):
    print(i)
    X, y = next(gen)
    #print(np.shape(X))
    X = np.reshape(X, (200, 3, 400, 200))
    #print(np.shape(y))
    y = np.reshape(y, ( 200, 36 ))
    history = model.fit(X, y, batch_size = 200)
    losses.append(history.history['loss'])
    if i % 5 == 0:
        np.save('losses_cnn.txt', losses)
    if i % 50 == 0:
        model.save('/home/ubuntu/mod_cnn.h5')
'''
