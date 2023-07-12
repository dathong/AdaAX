from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
# from utils import index_of
# from utils import convert_data_x
# from utils import convert_data_x_todigit
from config import Config
import pickle

tf.compat.v1.disable_eager_execution()

num_epochs = 100
total_series_no = 20000
truncated_backprop_length = 10
state_size = Config.stateSize

echo_step = 3
batch_size = 1
num_layers = 1
num_batches = total_series_no//batch_size//truncated_backprop_length
alphabets = pickle.load(open("alphabet","rb"))
y_lbl = ['1','0']
num_classes = len(y_lbl)


def readData():
    x = open("x.csv","r")
    y = open("y.csv","r")
    x_res,y_res = [],[]
    for r in x:
        r_elems = r.strip().split(",")
        x_res.append([str(e) for e in r_elems])
    for r in y:
        y_res.append(r.strip())
    x.close()
    y.close()
    return x_res,y_res


x,y = readData()
# alphabets = list(set([w for r in x for w in r]))
alphabets = pickle.load(open("alphabet","rb"))
print("alphabets = ",alphabets)
print("len alphabet: ",len(alphabets))
x,y = np.array(x), np.array(y)
# x,y = np.array(x), np.array(y)
print(x,x.shape)
print(y,y.shape)
# sys.exit()

batchX_placeholder = tf.placeholder(tf.float32, [None,None,len(alphabets)])
y_lbl_placeholder = tf.placeholder(tf.int64, [None, num_classes])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# c_tile = tf.constant(x)
W_tile = tf.tile(tf.expand_dims(W2,axis=0), [tf.shape(batchX_placeholder)[0],1,1])
# Unpack columns

# Forward passes
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm for _ in range(num_layers)])
init_state = cell.zero_state(tf.shape(batchX_placeholder)[0], tf.float32)

states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=init_state)


# logits_series = tf.matmul(tf.squeeze(states_series,axis=0),W2)
logits_series = tf.matmul(states_series,W_tile)
prediction_series = tf.nn.softmax(logits_series,axis=2)

# logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

output_logits = logits_series[:,-1,:]
y_pred = prediction_series[:,-1,:]

# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#             for logits, labels in zip(logits_series,labels_series)]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_lbl_placeholder, logits=output_logits), name='loss')

# total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

def convert_data_x_todigit(x,alphabets):
    res = []
    for ip in x:
        res1 = []
        for e in ip:
            for i,d in enumerate(e):
                if str(d) == '1':
                    res1.append(alphabets[i])
        res.append(res1)
    return res

def count_acc(y_pred, y,th=0.5):
    count,total = 0,len(y)

    for i in range(len(y_pred)):
            if y_pred[i] >= th:
                if y[i] == 1:
                    count+=1
            else:
                if y[i] == 0:
                    count+=1
    return count/total


def convert_data_x(x, alphabets):
    d = {a: i for i, a in enumerate(alphabets)}
    res = []
    for ip in x:
        res1 = []
        for e in ip:
            if e == "\n":
                continue
            v = [0] * len(d)
            # print("e = ",e)
            # print("d = ",d)
            v[d[e]] = 1
            res1.append(v)
        res.append(res1)
    return res


def convert_data_y(y, alphabets):
    d = {a: i for i, a in enumerate(alphabets)}
    res = []
    for r in y:
        for e in r:
            if e == "\n":
                continue
            v = [0] * len(d)
            v[d[e]] = 1
            res.append(v)
            break
    return res

saver = tf.compat.v1.train.Saver()



with tf.compat.v1.Session() as sess:
    # sess.run(tf.compat.v1.initialize_all_variables())
    saver.restore(sess, './my_test_model')

    # tvars = tf.trainable_variables()
    # tvars_vals = sess.run(tvars)
    #
    # for var, val in zip(tvars, tvars_vals):
    #     print("[db] var names: ",var.name, val)  # Prints the name of the variable alongside its value.

    res = []
    state_seq = []
    word_seq = []
    logit_seq = []
    lbls_seq = []
    reset_bool = []
    ind_seq = []

    loss_list = []
    # x, y = generate_data1(total_series_no = 5000)
    x0 = convert_data_x(x, alphabets)
    y0 = convert_data_y(y, y_lbl)

    # print('x = ', x)
    # print('x0 = ', x0)

    for epoch_idx in range(1):
        y_true_lbl, y_pred_lbl = [], []
        # x, y = generate_data1(total_series_no = 5000)
        # x0 = convert_data_x(x, alphabets)
        # x1, y1 = shuffle(x0, y)
        x1,y1 = x0,y0
        # np.insert(x1, 0, [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], axis=0)
        # np.insert(y1, 0, [1,0], axis=0)
        all_acc = 0

        print("New data, epoch", epoch_idx)



        for i in range(len(x1)):

            x,y = x1[i],y1[i]
            # print("x = ",len(x))
            # print("y = ",y)
            _W2, _states_series, _current_state, _y_pred, _logits_series,_prediction_series = sess.run(
                [ W2, states_series, current_state, y_pred, logits_series,prediction_series],
                feed_dict={
                    batchX_placeholder: [x],
                    y_lbl_placeholder: [y],


                })

            # print("_W2 = ",_W2)
            reset_bool.append(1)
            state_seq.extend(_states_series[0])
            ind_seq.extend([0 for i in range(len(_states_series[0]) - 1)])
            ind_seq.append(1)
            co_batchX = convert_data_x_todigit([x],alphabets)
            word_seq.extend(co_batchX[0])
            logit_seq.extend(_logits_series[0])
            lbls_seq.extend(_y_pred)
            reset_bool.extend([0 for i in range(len(x) - 1)])

            # print("[db] _y_pred,y = ", _y_pred, y)
            y_pred_lbl.append(_y_pred[0][0])
            y_true_lbl.append(y[0])

        yPredFile = open("yPred.csv","w")
        for y_pred in y_pred_lbl:
            yPredFile.write(str(int(round(y_pred))) + "\n")
        yPredFile.close()
        acc = count_acc(y_pred_lbl, y_true_lbl)
        print("[db] acc = ",acc)

    import pandas as pd
    print('printing...')
    df = pd.DataFrame()
    df['states'] =  state_seq
    df['words'] = word_seq
    df['logit_seq'] = logit_seq
    df['reset_bool'] = reset_bool
    df['ind_seq'] = ind_seq
    # df['state_list_sm'] = list(state_list_sm)
    df.to_csv('long_states_df.csv', index=False, header=True)
    print('Done')

# plt.ioff()
# plt.show()