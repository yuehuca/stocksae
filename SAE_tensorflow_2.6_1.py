# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import pandas as pd
import collections

#os.chdir("D:\\workspace\stock20180609")
#os.chdir("D:\\workspace\series")

period = 1
task = 2.6

training_scale = 0.4583
samples_x = np.loadtxt("aa_" + str(period) + ".txt")
samples_y = np.loadtxt("bb_" + str(period) + "_0.txt")
testreturns = np.loadtxt("dd_" + str(period) + ".txt")

samples_y = samples_y.reshape(len(samples_y),1)
min_max_scaler = preprocessing.MinMaxScaler()
samples_x_minmax = min_max_scaler.fit_transform(samples_x)
samples_y_minmax = min_max_scaler.fit_transform(samples_y)
sampleend_x = int(round(training_scale*len(samples_x_minmax),0))
sampleend_y = int(round(training_scale*len(samples_y_minmax),0))
train_x = samples_x_minmax[0:(sampleend_x)]
train_y = samples_y_minmax[0:(sampleend_y)]
test_x = samples_x_minmax[(sampleend_x):len(samples_x_minmax)]
test_y = samples_y_minmax[(sampleend_y):len(samples_y_minmax)]

starter_learning_rate = 0.9999999999999
decay_steps_ae = 500
decay_rate_ae = 0.80
decay_steps = 500
decay_rate = 0.80
training_epochs_ae = 5000   #Pretraining epochs5000
training_epochs = 100000    #Training epochs100000
batch_size = len(train_x)        #Full batch training
display_step = 500        #Display steps
ZeroMaskedFraction = 0.5  #Zeromask fraction, only for pretraining
randstate = 0  #Setting all the rand state
fit_y_test_top10 = []

def sae_network(decay_steps_ae,decay_rate_ae,decay_steps,decay_rate,
                training_epochs_ae,training_epochs,ZeroMaskedFraction,
                randstate):
    #Setting up zeromask
    random.seed(randstate)
    global train_x
    zeromask = random.uniform(0,1,size=(train_x.shape[0],train_x.shape[1]))
    zeromask = np.where(zeromask < ZeroMaskedFraction, zeromask, 1)
    zeromask = np.where(zeromask >= ZeroMaskedFraction, zeromask, 0)
    train_x = samples_x_minmax[0:(sampleend_x)]
    train_x = train_x * zeromask
    
    n_hidden_1 = 7
    n_hidden_2 = 30
    n_hidden_3 = 30
    n_hidden_4 = 30
    n_hidden_5 = 30
    n_hidden_6 = 30
    n_hidden_7 = 30
    n_hidden_8 = 30
    n_input = train_x.shape[1]
    n_output = 1
    
    ################################################################################
    
    #Defining each autoencoder
    
    x_ae = tf.placeholder("float", [None,n_input])
    y_ae = tf.placeholder("float", [None,n_hidden_1])
    
    weights_ae = [
         tf.Variable(tf.random_normal([n_input, n_hidden_1],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_8, n_output],seed=randstate),trainable=True)
    ]
    
    biases_ae = [
         tf.Variable(tf.random_normal([n_hidden_1],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_2],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_3],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_4],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_5],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_6],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_7],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_8],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_output],seed=randstate),trainable=True)
    ]
    
    weights_dae = [
         tf.Variable(tf.random_normal([n_hidden_1,n_input],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_6, n_hidden_5],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_7, n_hidden_6],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_8, n_hidden_7],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_output, n_hidden_8],seed=randstate),trainable=True)
    ]
    
    biases_dae = [
         tf.Variable(tf.random_normal([n_input],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_1],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_2],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_3],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_4],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_5],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_6],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_7],seed=randstate),trainable=True),
         tf.Variable(tf.random_normal([n_hidden_8],seed=randstate),trainable=True)
    ]
    
    weights_aelist = []  #Recording the pretraining weights
    biases_aelist = []  #Recording the pretraining biases
    
    ################################################################################
    
    #Pretraining process
    
    for loop_ae in range(0,len(weights_ae)):
        def encoder_ae(X):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights_ae[loop_ae]),
                                       biases_ae[loop_ae]))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_dae[loop_ae]),
                                       biases_dae[loop_ae]))
            return layer_1, layer_2
    
        y_pred_ae = encoder_ae(x_ae)[1]  #The output need to compare with the target
        y_pred_next = encoder_ae(x_ae)[0]  #The output need to transfer to the next autoencoder
        y_true_ae = y_ae
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps_ae, decay_rate_ae, staircase=True)
        loss_ae = tf.reduce_mean(tf.square(y_pred_ae - y_true_ae))
        optimizer_ae = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_ae)
        #optimizer_ae = tf.train.AdagradOptimizer(learning_rate).minimize(loss_ae)
        init = tf.global_variables_initializer();
        
        with tf.Session() as sess:
            sess.run(init)
            #writer_ae = tf.summary.FileWriter("D://workspace//stock20180609",sess.graph)
            total_batch = int(len(train_x)/batch_size)
            #Pretraining loops
            for epoch in range(training_epochs_ae):
                for i in range(total_batch):
                    _, loss_value_ae = sess.run([optimizer_ae, loss_ae], feed_dict={x_ae: train_x, y_ae: train_x})
                #Pretraining results
                if epoch % display_step == 0:
                    print("Pretraining epoch:", '%04d' % (epoch+1),
                          "loss=", "{:.9f}".format(loss_value_ae))
            train_x = sess.run(y_pred_next, feed_dict={x_ae: train_x})  #Replacing training samples for next autoencoders
            x_ae = tf.placeholder("float", [None,train_x.shape[1]])  #Reshaping/defining input tensor
            y_ae = tf.placeholder("float", [None,train_x.shape[1]])  #Reshaping/defining output tensor
            print("Completing pretraining steps: " + str(loop_ae+1))
            record_weights_ae = sess.run(weights_ae)
            record_biases_ae = sess.run(biases_ae)
            weights_aelist.append(record_weights_ae[loop_ae])  #Recording pretrained weights
            biases_aelist.append(record_biases_ae[loop_ae])  #Recording pretrained biases
            
        del(y_pred_ae)
        del(y_pred_next)
    print("Completing pretraining")
        
    ################################################################################
    
    #Continuing bp training from here to the end
    #Defining bp network
    
    x = tf.placeholder("float", [None,n_input])
    y = tf.placeholder("float", [None,n_output])
    train_x = samples_x_minmax[0:(sampleend_x)]  #Fixing the training samples
    
    #Applying the trained weights and biases to the bp network
    
    weights = [
         tf.Variable(weights_aelist[0],trainable=True),
         tf.Variable(weights_aelist[1],trainable=True),
         tf.Variable(weights_aelist[2],trainable=True),
         tf.Variable(weights_aelist[3],trainable=True),
         tf.Variable(weights_aelist[4],trainable=True),
         tf.Variable(weights_aelist[5],trainable=True),
         tf.Variable(weights_aelist[6],trainable=True),
         tf.Variable(weights_aelist[7],trainable=True),
         tf.Variable(weights_aelist[8],trainable=True),
    ]
    
    biases = [
         tf.Variable(biases_aelist[0],trainable=True),
         tf.Variable(biases_aelist[1],trainable=True),
         tf.Variable(biases_aelist[2],trainable=True),
         tf.Variable(biases_aelist[3],trainable=True),
         tf.Variable(biases_aelist[4],trainable=True),
         tf.Variable(biases_aelist[5],trainable=True),
         tf.Variable(biases_aelist[6],trainable=True),
         tf.Variable(biases_aelist[7],trainable=True),
         tf.Variable(biases_aelist[8],trainable=True),
    ]
    
    def encoder(X): 
        #sigmoid activation function，layer = x*weights['encoder_h1']+biases['encoder_b1']
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[0]),
                                       biases[0]))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights[1]),
                                       biases[1]))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights[2]),
                                       biases[2]))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights[3]),
                                       biases[3]))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights[4]),
                                       biases[4]))
        layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights[5]),
                                       biases[5]))
        layer_7 = tf.nn.sigmoid(tf.add(tf.matmul(layer_6, weights[6]),
                                       biases[6]))
        layer_8 = tf.nn.sigmoid(tf.add(tf.matmul(layer_7, weights[7]),
                                       biases[7]))
        layer_9 = tf.nn.sigmoid(tf.add(tf.matmul(layer_8, weights[8]),
                                       biases[8]))
        return layer_9
    
    y_pred = encoder(x)
    y_true = y
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer();
    
    #saver = tf.train.Saver()  #Defining the saver
    
    ################################################################################
    
    #Training bp network
    
    with tf.Session() as sess:
        sess.run(init)
        #writer = tf.summary.FileWriter("D://workspace//stock20180609",sess.graph)
        #saver = tf.train.import_meta_graph('./tensorflow_SAE_test.meta')  #Omitting these two lines if it is first training
        #saver.restore(sess, tf.train.latest_checkpoint("./"))  #Restoring the weights
        total_batch = int(len(train_x)/batch_size)
        #Training loops
        for epoch in range(training_epochs):
            for i in range(total_batch):
                _, loss_value = sess.run([optimizer, loss], feed_dict={x: train_x, y: train_y})
        #Training results
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "loss=", "{:.9f}".format(loss_value))
        print("Optimization Finished!")
    
        output_train = sess.run(
            y_pred, feed_dict={x: train_x})
        output_test = sess.run(
            y_pred, feed_dict={x: test_x})
        record_weights = sess.run(weights)
        record_biases = sess.run(biases)
        #saver.save(sess,"./tensorflow_SAE_2015_2.0")  #Saving the weights
    
    fit_y_train = min_max_scaler.inverse_transform(output_train)
    fit_y_test = min_max_scaler.inverse_transform(output_test)
    fit_y_train_list = list(fit_y_train)
    fit_y_test_list = list(fit_y_test)
    fit_y_total_list = fit_y_train_list + fit_y_test_list
    fit_y_total = np.array(fit_y_total_list)
    
    len_train = list(range(0,len(fit_y_train)))
    len_test = list(range(len(fit_y_train),len(fit_y_total)))
    len_total = list(range(0,len(fit_y_total)))
    
    #plt.plot(len_total,samples_y,'*',color='#008000')
    #plt.plot(len_train,fit_y_train,'.',color='#0000FF')
    #plt.plot(len_test,fit_y_test,'.',color='#FF0000')
    #plt.show()
    
    #Recording the top10 largest outputs, change to (1,11)
    #Here argsort all the stocks
    for toptest in range(1,(len(fit_y_test)+1)):
        fit_y_test_top = np.argsort(fit_y_test,axis=0)[-toptest]
        fit_y_test_top10.append(fit_y_test_top) 

    
    #savepath = '2015' + str(randstate) + '.csv'
    #fit_y_total_df = pd.DataFrame(fit_y_total)
    #fit_y_total_df.to_csv(savepath)
    #print(fit_y_total)
    
    #tf.reset_default_graph()  #Omitting this line if it is first training
    return fit_y_test_top10

top10averagereturn_total = []
for training_epochs in range(100000,110000,100000):
    fit_y_test_top10 = []
    for randstate in range(0,50,1):
        sae_network(decay_steps_ae,decay_rate_ae,decay_steps,decay_rate,
            training_epochs_ae,training_epochs,ZeroMaskedFraction,
            randstate)


    #Transfering all the top10 outputs to integer list
    fit_y_test_top10_list = []
    for toptest2list in range(0,len(fit_y_test_top10)):
        fit_y_test_top10_list.append(int(fit_y_test_top10[toptest2list]))
    
    #Finding 10 stocks which appear more times among top10 stocks in all the trainings
    top10counter = collections.Counter(fit_y_test_top10_list)
    top10alllist_index = []
    top10alllist_count = []
    for top10element in top10counter:
        top10alllist_count.append(top10counter[top10element])
        top10alllist_index.append(top10element)
    
    #Sorting the counting results and recording indexes of frequent appear list
    top10alllist_count_top10 = []
    for toptest in range(1,11):
        top10alllist_count_top = np.argsort(top10alllist_count,axis=0)[-toptest]
        top10alllist_count_top10.append(top10alllist_count_top) 
    
    #Transfering the frequent appear list indexes to total test samples indexes
    top10all_index = []
    for top10index in top10alllist_count_top10:
        top10all_index.append(top10alllist_index[top10index])
    
    #Finding related returns and calculating average return
    top10returns = testreturns[[top10all_index]]
    top10averagereturn = np.mean(top10returns)
    
    #Saving the results
    savepath = str(period) + '_' + str(task) + '_' + str(training_epochs) + '.csv'
    savepath_all = 'all_' + str(period) + '_' + str(task) + '_' + str(training_epochs) + '.csv'
    top10all_index.append('Average')
    top10returns = list(top10returns)
    top10returns.append(top10averagereturn)
    top10all_index_df = pd.DataFrame(top10all_index)
    top10returns_df = pd.DataFrame(top10returns)
    results_df = pd.concat( [top10all_index_df, top10returns_df], axis=1 )
    results_df.to_csv(savepath)

    fit_y_test_top10_df = pd.DataFrame(fit_y_test_top10)
    fit_y_test_top10_df.to_csv(savepath_all)

    top10averagereturn_total.append(top10averagereturn)
    
top10averagereturn_total_average = np.mean(top10averagereturn_total)
print(top10averagereturn_total_average)

