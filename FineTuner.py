# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:27:37 2018

@author: Mateusz
"""
import numpy as np
from matplotlib import pyplot as  plt
from inspect import getsourcefile
from random import shuffle
import os
from math import sqrt
import math
import pickle as pkl 
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from AutoEnc import *
from sys import stdout
#from sklearn.utils.extmath import softmax
import sys

class FineTuner():
    def __init__(self, mu, gamma, low, high, acc_achv, activation_name, hidden, batch_size,\
                 l1, l2, n_iter, save = True, spl_perc = 1, erl_ctr = 2, drop_p = 1):
        '''class representing multi layer neural network
        attributes:
            -path - path where file is stored
            -mu - learning rate
            -weights - weights of a single neuron
            -activation - activation function one of the following linear, sigmoid, tanh
            -eta - adaptive learning rate coefficient
            -n_iter - number of iterations to be done 
            -beta - parameter for sigmoid function only 
            -batch_size - number trainings samples in single mini batch
            -n_hidden - number of neurons in hidden layer
            -eps - cost which indicate stopping train function
            -erl_stop - if true then early stopping technique is applied '''
        self.path= os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.epoch = 0
        self.mu = mu
        self.gamma = gamma
        self.low, self.high = low, high 
        self.acc_achv = acc_achv
        self.beta = 1
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.hidden = hidden
        self.read_sets(spl_perc)
        self.activation_name = activation_name
        self.set_activation_function(activation_name)
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.batch_loss = []
        self.spl_perc = spl_perc
        self.drop_p = drop_p
        self.save = save
        self.erl_ctr = erl_ctr
        self.stop = False
        self.n_iter = n_iter
        self.name = '_'.join(list(map(str, ['mu', self.mu, 'gamma', self.gamma, 'low',self.activation_name,'activation',\
                                            'low', self.low, 'high', self.high, 'acc_achv',\
                                            self.acc_achv, 'batch_sz', self.batch_size, 'hidd', self.hidden])))
      
    def set_activation_function(self, activation):
        '''function for setting activation function '''
        self.activation = {'relu':self.relu, 'sigmoid':self.sigmoid, 'softmax':self.softmax }[activation]
        self.der_activation = {'relu':self.relu_der,'sigmoid':self.sigmoid_der, 'softmax':self.softmax_der }[activation]

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-self.beta*X))

    def sigmoid_der(self, X):
        return self.beta*self.sigmoid(X)*(1-self.sigmoid(X))

    def relu(self, X):
        return np.where(X>0, X, 0)
    
    def relu_der(self, X):
        return np.where(X>0, 1, 0)

    def softmax(self, X):
        e_x = np.exp(X - np.max(X))
        return e_x / (e_x).sum(axis=0) # only difference
        
    def softmax_der(self, X):
        return self.softmax(X)*(1-self.softmax(X))
    
    def read_weights(self, type):
        '''function for reading weights from file'''
        wght_anc = 'weights_'+str(self.hidden)+'.npy'
        wght_fnt = 'weights_'+str(self.hidden)+'_finetuned.npy'
        wght_hid_fnt = 'weights_'+str(self.hidden)+'hidden_finetuned.npy'
        
        if type == 'train':
            if os.path.isfile(wght_anc):
                print ('Assigning weights trained for autoencoder. Hidden weights randomly assigned')
                self.weights = np.load(wght_anc)
                self.weights_hidden = np.random.uniform(low = self.low, high = self.high, size = (10,self.hidden+1))
            else:
                print ('File with', self.hidden, 'hidden units not found. Training autoencoder with', self.hidden, 'neurons...')
                anc = AutoEnc(mu = self.mu, gamma = self.gamma, low = self.low, high = self.high, activation_name = 'sigmoid',\
                      n_iter = 10, hidden = self.hidden, batch_size = self.batch_size, l1 = self.l1, l2 = self.l2, spl_perc = self.spl_perc)
                anc.train()
                self.weights = np.load(wght_anc)
                self.weights_hidden = np.random.uniform(low = self.low, high = self.high, size = (10,self.hidden+1))
        elif type == 'test':
            if os.path.isfile(wght_fnt) and os.path.isfile(wght_hid_fnt):
                print ('Assigning finetuned weights')
                self.weights = np.load(wght_fnt)
                self.weights_hidden = np.load(wght_hid_fnt)
            else:
                print ('Finetuned weights not found.')
            
        self.wght_bef = {'weights':self.weights, 'weights_hidden':self.weights_hidden}
            
    def read_sets(self, spl_perc):
        '''method for reading test, validation and training sets'''
        with open('mnist.pkl', 'rb') as file:
            self.train_set, self.valid_set, self.test_set = pkl.load(file, encoding='iso-8859-1')
        
        self.train_set, self.train_set_hot = self.cut_set(self.train_set, spl_perc)
        self.valid_set, self.valid_set_hot = self.cut_set(self.valid_set, spl_perc)
        self.test_set, self.test_set_hot = self.cut_set(self.test_set, spl_perc)
        self.n_features = self.train_set.shape[1]
        self.p = self.train_set.shape[0] #total number of training samples
        
    def cut_set(self, set_cut, spl_perc):
        '''cut set to spl_perc percents and return new set'''
        rnd_set = [np.random.choice(set_cut[0].shape[0], int(set_cut[0].shape[0]*spl_perc), replace=False)]
        ret_set = set_cut[0][rnd_set] #choose random numbers from set passed as argument
        res_set_label = set_cut[1][rnd_set] #choose labels which corresponds to randomly chosen numbers
        hot_label = np.eye(10)[res_set_label]
        return ret_set, hot_label.T  #return cutted sets along with labels
        
    def add_bias_row(self, X):
        '''adding single row of bias to matrix X'''
        bias = np.ones((1, X.shape[1]))
        return np.vstack([bias, X])
        
    def add_bias_col(self, X):
        '''adding single column of bias to matrix X'''
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])
        
    def feedworward(self, inp_data):
        '''feedworward learning algorithm
                1. Start with input data and add bias column to it
                2. Make a dot product of transposed input data and weights
                3. Apply activation function to results of dot product
                4. Make a dot product of hidden weights and result of activation function
                5. Apply activation function and result of dot product from step 4.
            output:
                z_2, a_2, z_3, y_p '''
        x_data = self.add_bias_col(inp_data)
        z_2 = self.weights.dot(x_data.T)
        a_2 = self.activation(z_2)
        a_2 = self.add_bias_row(a_2)
        z_3 = self.weights_hidden.dot(a_2)
        y_p = self.softmax(z_3)
        return x_data, z_2, a_2, z_3, y_p
        
    def backpropagation(self, y_p, z_2, a_2, k):
        '''backpropagation learning algorithm
                1. Start with calucalting error between expected results and y calculated in feedforward algoritm
                2. Add bias row to the output of weights times x_data
                3. (SGD) Make a dot product of error at the last layer and transposed weights
                4. Multiply result(3.) by derivative of activation function 
            output:
                sigma2, sigma3 '''
        sigma3 = y_p
        sigma3[k] = sigma3[k] - 1
        z_2 = self.add_bias_row(z_2) #add bias to row of the last layer
        sigma2 = self.weights_hidden.T.dot(sigma3)*self.der_activation(z_2) #do backpropagation
        sigma2 = sigma2[1:, :] #leave bias
        return sigma2, sigma3
    
    def get_gradient(self, x_data, a_2, sigma2, sigma3, weights_t, weights_hidden_t):
        '''gradient calculation for backpropagation algorithm
                1. Calculate gradient for output layer 
                2. Calculate gradient for hidden layer 
                3. '''
        grad1 = sigma2.dot(x_data) #calculate gradient for hidden layer
        grad2 = sigma3.dot(a_2.T) #calculate gradient for output layer
        wght_drop = self.weights*self.drop
        delta_w1 = self.mu*grad1 + self.mu*self.l1/self.p*np.sign(wght_drop)+self.mu*self.l2/self.p*wght_drop   #mulitply gradients by learning rate
        delta_w2 =  self.mu*grad2 + self.mu*self.l1/self.p*np.sign(self.weights_hidden)+self.mu*self.l2/self.p*self.weights_hidden      #mulitply gradients by learning rate
        self.weights -= (delta_w1 + (self.gamma*delta_w1) ) #subtract delta w_1 from weights at time t 
        self.weights_hidden -= (delta_w2 + (self.gamma*delta_w2) ) #subtract delta w_1 from hidden weights at time t 
        return delta_w1, delta_w2 
    
    def dropout(self):
        '''dropout function which prevent from overfitting'''
        self.drop = np.random.binomial(1, self.drop_p, size = self.weights.shape)/self.drop_p
        self.weights *= self.drop
    
    def regul_l1(self):
        'regularization l2'''
        self.train_loss[-1] += self.l1/(self.p)*(np.sum(abs(self.weights)) + np.sum(abs(self.weights_hidden)))
        
    def regul_l2(self):
        '''regularization l2'''
        self.train_loss[-1] += self.l2/(2*self.p)*(np.sum(self.weights**2) + np.sum(self.weights_hidden**2))
    
    def train(self):
        '''main train function for training multilayer neural network'''
        print ('Finetuner training')
        self.read_weights('train')
        p = self.train_set.shape[0] #total number of training samples
        self.batches = int(p/self.batch_size)
        delta_weights = np.zeros(self.weights.shape) #weights at time t
        delta_weights_hidden = np.zeros(self.weights_hidden.shape) #hidden weights at time t
        acc_achv = False
        
        while (not acc_achv and not self.stop):
            idx_split = np.array_split(range(p),int(self.batches)) #split training array on smaller batches
            idx = np.random.permutation(self.train_set.shape[0]) #shuffling numbers in each epoch to prevent from adapting too much to pattern
            X_data = self.train_set[idx] #choose X_data after shuffling
            y_data = self.train_set_hot[:,idx]
            
            ctr = 0
            
            while ctr < len(idx_split) and not self.stop:
                self.dropout()
                idx = idx_split[ctr]
                x_data = X_data[idx] #training checking
                y_true = y_data[:,idx]
                x_data, z_2, a_2, z_3, y_p = self.feedworward(x_data)
                k = np.where(y_true!=0)
                y_p_fnd = np.where(y_p[k] ==0, np.finfo(np.float32).eps, y_p[k])
                btc_loss = -1/len(idx_split)*np.sum(np.log(y_p[k]))
                self.batch_loss.append(btc_loss)
                wht_bef = np.copy(self.weights)
                hid_wht_bef = np.copy(self.weights_hidden)
                sigma2, sigma3 = self.backpropagation(y_p, z_2, a_2, k)
                delta_weights, delta_weights_hidden = self.get_gradient(x_data, a_2, sigma2, sigma3, delta_weights, delta_weights_hidden)
                self.hinton()
                ctr+=1
            
            x_data, z_2, a_2, z_3, y_p = self.feedworward(self.train_set) #validation checking
            k = np.where(self.train_set_hot!=0)
            y_p_fnd = np.where(y_p[k] ==0, np.finfo(np.float32).eps, y_p[k])
            trn_loss = -1/len(self.train_set)*np.sum(np.log(y_p[k]))
            self.train_loss.append(trn_loss)
            self.regul_l2()
            self.regul_l1()
            x_data, z_2, a_2, z_3, y_p = self.feedworward(self.valid_set) #validation checking
            k = np.where(self.valid_set_hot!=0)
            y_p_fnd = np.where(y_p[k] ==0, np.finfo(np.float32).eps, y_p[k])
            val_loss = -1/len(self.valid_set)*np.sum(np.log(y_p[k]))
            val_loss_min = np.min(self.valid_loss) if len(self.valid_loss) > 1 else 10**6
            self.valid_loss.append(val_loss)
            val_acc = self.get_acc('valid')
            self.valid_acc.append(val_acc)
            trn_acc = self.get_acc('train')
            self.train_acc.append(trn_acc)
            if self.valid_loss[-1] < val_loss_min:
                print ('Best weights swapping')
                self.wght_bef['weights'] = wht_bef
                self.wght_bef['hidden_weights'] = hid_wht_bef
            print ('accuracy: train %.3f valid %.3f at epoch %d' %(trn_acc, val_acc, self.epoch) )
            print ('loss: train %.3f valid %.3f at epoch %d' %(self.train_loss[-1], val_loss, self.epoch) )
            self.early_stopping()
            if self.valid_acc[-1] > self.acc_achv:
                acc_achv = True
                print ('Accuracy %.2f %% achieved in epoch %d' %(self.acc_achv*100, self.epoch+1))
            self.epoch += 1
            if self.epoch > self.n_iter:
                self.stop = True
                print ('Accuracy not achieved, stopping at epoch ' ,self.epoch)
        if self.save:
            wght_fnt = 'weights_'+str(self.hidden)+'_finetuned.npy'
            wght_hid_fnt = 'weights_'+str(self.hidden)+'hidden_finetuned.npy'
            np.save(wght_fnt, self.weights)
            np.save(wght_hid_fnt, self.weights_hidden)
            
    def early_stopping(self):
        if len(self.valid_loss) > self.erl_ctr:
            last = self.valid_loss[-1]
            ctr = 0
            for i in range(1,self.erl_ctr+1):
#                diff = last - self.valid_loss[-(i+1)]
                if last >= self.valid_loss[-(i+1)]:
                    ctr +=1
                    last = self.valid_loss[-(i+1)]
            if ctr == self.erl_ctr:
                self.stop = True
                self.weights = self.wght_bef['weights']
                self.weights_hidden = self.wght_bef['weights_hidden']
                print ('Early stopping because: ', self.valid_loss[-self.erl_ctr:])
                

    def hinton(self):
        ro_i = 1.05
        ro_d = 0.7
        k_w = 1.04
        if len(self.train_loss) > 1:
            if math.sqrt(self.train_loss[-1]) > k_w *math.sqrt(self.train_loss[-2]):
                self.mu *= ro_d
            elif (math.sqrt(self.train_loss[-1]) <= k_w *math.sqrt(self.train_loss[-2]) ):
                self.mu *= ro_i

    
    def test(self,type = 'test'):
        '''method for testing data along with forward propagation algorithm'''
        if type == 'test':
            test_set = self.test_set
        elif type =='valid':
            test_set = self.valid_set
        elif type =='train':
            test_set = self.train_set
        
        x_data, z_2, a_2, z_3, y_p = self.feedworward(test_set)
        pred = np.argmax(y_p,0)
        return pred
            
    def get_acc(self, type ='test'):
        '''method for checking accuracy of data, either testing, validation or training'''
        y_p = self.test(type)
        if type == 'test':
            y_true = np.argmax(self.test_set_hot,axis=0)
        elif type == 'valid':
            y_true = np.argmax(self.valid_set_hot,axis=0)
        elif type == 'train':
            y_true = np.argmax(self.train_set_hot,axis=0)
        return np.sum(y_p== y_true)/y_p.shape[0]
    
    def plot_single(self, y_p, true_lbl):
        fig, ax = plt.subplots()
        img = y_p.reshape(28, 28)
        ax.imshow(img, cmap='Greys', interpolation='nearest')
        ax.set_title('real: %d' %  true_lbl)
            
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    def plot_batch_loss(self):
        fig, ax = plt.subplots()
        ax.plot(range(1,len(self.batch_loss)+1), self.batch_loss, label = 'Zbiór paczek')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Strata')
        ax.set_title('Wykres straty dla autoenkodera przy %d iteracjach' % self.n_iter)
        ax.legend()
        fig.savefig('best_model\\Autoencoder\\batchloss_'+self.name+'.eps', format='eps',dpi=1000)
        plt.close()
        
    def plot_examples(self, y_p):
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        numbers = rnd.sample(range(y_p.shape[0]),25)
        ctr = 0
        
        for i in numbers:
            curr = self.test_set[i]
            img = curr.reshape(28, 28)
            ax[ctr].imshow(img, cmap='Greys', interpolation='nearest')
            ax[ctr].set_title('pred %d real: %d' %  (y_p[i], np.argmax(self.test_set_hot[:,i])))
            ctr +=1
            
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
    
    def plot_acc(self,path):
        fig, ax = plt.subplots()
        ax.plot(self.valid_acc, label = 'Zbiór walidacyjny')
        ax.plot(self.train_acc, label = 'Zbiór treningowy')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Dokładnosc')
        ax.set_title('Wykres dokładnosci dla zadanej dokladnosci %.2f %%' %self.acc_achv)
        ax.legend()
        fig.savefig(path + '\\Finetuner\\acc_'+self.name+'.eps', format='eps',dpi=1000)
        
      
    def plot_loss(self, path):
        fig, ax = plt.subplots()
        ax.plot(self.valid_loss, label = 'Zbiór walidacyjny')
        ax.plot(self.train_loss, label = 'Zbiór Treningowy')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Strata')
        ax.set_title('Wykres straty dla zadanej dokladnosci %.2f %%' %self.acc_achv)
        ax.legend()
        fig.savefig(path + '\\Finetuner\\loss_'+self.name+'.eps', format='eps',dpi=1000)
        
    def get_pixels_int(self):
        '''calculate pixel intensity'''
        scaler = MinMaxScaler()
        trans = scaler.fit_transform(self.weights)
        intensity = np.zeros(self.weights.shape)
        
        for i in range(trans.shape[0]):
            neuron_sum = 0
            for j in range(trans.shape[1]):
                neuron_sum += (trans[i][j])**2
                neuron_wght = trans[i][j]
                pix_int = neuron_wght/sqrt(neuron_sum)
                intensity[i,j] = pix_int
        
        return intensity
    
    def plot_pix_int(self):
        '''plot pixel intensity'''
        intensity = self.get_pixels_int()
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(10):
            curr = intensity[i,1:]
            img = curr.reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('Nrn %d int' % (i+1))
            
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        fig.savefig('ex6\\pix_int_'+self.name+ '.eps', format = 'eps', dpi = 1000)

if __name__ == '__main__':
    mu = 0.001
    gamma = 0.1
    low, high = -0.1,0.1
    activation_name = 'relu'
    hidden = 10
    batch_size = 40
    acc_achv = 0.9
    l1 = 0.0001
    l2 = 0.0005
    spl_perc = 0.02
    erl_ctr = 10
    drop_p = 1
    n_iter = 1000
    fnt = FineTuner(mu = mu, acc_achv = acc_achv, gamma = gamma, low = low, high = high, activation_name = activation_name,
                    hidden = hidden, batch_size = batch_size, l1 = l1, l2 = l2, \
                    erl_ctr = erl_ctr, spl_perc = spl_perc, drop_p = drop_p, n_iter  =n_iter )
    fnt.train()
    fnt.plot_acc('ex6\\')
    fnt.plot_loss('ex6\\')
    y_p = fnt.test('test')
    fnt.plot_examples(y_p)
    fnt.plot_pix_int()
    
    