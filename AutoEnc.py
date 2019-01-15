# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:27:37 2018

@author: Mateusz
"""
import numpy as np
from matplotlib import pyplot as  plt
from inspect import getsourcefile
import random as rnd
import os
from math import sqrt
import math
import pickle as pkl 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from FineTuner import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

class AutoEnc():
    def __init__(self, mu, gamma, low, high, activation_name, n_iter, hidden, batch_size, l1,\
                 l2, save = True, erl_ctr = 10, spl_perc = 1, drop_p = 0):
        '''class representing multi layer neural network
        attributes:
            -path - path where file is stored
            -mu - learning rate
            -activation - activation function one of the following linear, sigmoid, tanh
            -n_iter - number of iterations to be done 
            -beta - parameter for sigmoid function only 
            -batch_size  - number samples in minibatch
            -n_hidden - number of neurons in hidden layer
            -eps - cost which indicate stopping train function
            -l1 - regularization parameter l1
            -l2 - regularization parameter l2
            -p_units_dis - percent of disabled units
            -erl_stop - if true then early stopping technique is applied '''
        self.path= os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.epoch = 0
        self.mu = mu
        self.gamma = gamma
        self.low, self.high = low, high 
        self.n_iter = n_iter
        self.beta = 1
        self.batch_size = batch_size
        self.hidden = hidden
        self.l1 = l1
        self.l2 = l2
        self.drop_p = drop_p
        self.n_features = 784
        self.read_sets(spl_perc)
        self.activation_name = activation_name
        self.set_activation_function(activation_name)
        self.batch_loss = []
        self.train_loss = []
        self.valid_loss = []
        self.erl_ctr = erl_ctr
        self.save = save
        self.name = '_'.join(list(map(str, ['mu', self.mu, 'gamma', self.gamma, 'low',self.activation_name,'activation',\
                                    'low', self.low, 'high', self.high, 'n_iter',\
                                    self.n_iter, 'batch_sz', self.batch_size, 'hidd', self.hidden])))
        
    def read_weights(self):
        '''function for reading weights from file'''
        wght = 'weights_'+str(self.hidden)+'.npy'
        wght_hid = 'weights_'+str(self.hidden)+'hidden.npy'
        
        if os.path.isfile(wght) and os.path.isfile(wght_hid):
            print ('Reading weights from file', wght, 'and', wght_hid)
            self.weights = np.load(wght)
            self.weights_hidden = np.load(wght_hid)
        else:
            print ('Assigning random weights')
            self.set_weights()
        
            
    def read_sets(self, spl_perc):
        '''method for reading test, validation and training sets'''
        with open('E:\\Sieci neuronowe\\mnist.pkl', 'rb') as file:
            self.train_set, self.valid_set, self.test_set = pkl.load(file, encoding='iso-8859-1')
        
        self.train_set, self.train_set_label = self.cut_set(self.train_set, spl_perc)
        self.valid_set, self.valid_set_label = self.cut_set(self.valid_set, spl_perc)
        self.test_set, self.test_set_label = self.cut_set(self.test_set, spl_perc)
        self.n_features = self.train_set.shape[1]
        self.p = self.train_set.shape[0] #total number of training samples
        
    def cut_set(self, set_cut, spl_perc):
        '''cut set to spl_perc percents and return new set'''
        ret_set = np.array([np.zeros(784)])
        res_set_label = np.array([])
        for i in range(10): #for each number in training set choose spl_perc number of training samples
            numb_ind = np.where(set_cut[1] == i)[0]
            numb_len = numb_ind.shape[0]
            rnd_set = [np.random.choice(numb_ind, int(numb_len*spl_perc), replace=False)]
            ret_set = np.append(ret_set, set_cut[0][rnd_set], axis = 0) #choose random indices and return them 
            res_set_label = np.append(res_set_label, set_cut[1][rnd_set] )
        
        return ret_set[1:,:], res_set_label.T #return cutted sets along with labels
        
    def add_bias_row(self, X):
        '''adding single row of bias to matrix X'''
        bias = np.ones((1, X.shape[1])) 
        return np.vstack([bias, X])
        
    def add_bias_col(self, X):
        '''adding single column of bias to matrix X'''
        bias = np.ones((X.shape[0], 1)) 
        return np.hstack([bias, X])
        
    def set_weights(self):
        '''set random weights for input and hidden layers '''
        self.weights = np.random.uniform(low = self.low, high = self.high, size = (self.hidden,self.n_features+1))
        self.weights_hidden = np.random.uniform(low = self.low, high = self.high, size = (self.n_features+1,self.hidden+1))
        
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
        a_2 = self.activation(z_2) #apply activation function 
        a_2 = self.add_bias_row(a_2)
        z_3 = self.weights_hidden.dot(a_2) 
        y_p = self.activation(z_3)
        return x_data, z_2, a_2, z_3, y_p
        
    def backpropagation(self, y_p, x_data, z_2, a_2):
        '''backpropagation learning algorithm
                1. Start with calucalting error between expected results and y calculated in feedforward algoritm
                2. Add bias row to the output of weights times x_data
                3. (SGD) Make a dot product of error at the last layer and transposed weights
                4. Multiply result(3.) by derivative of activation function 
            output:
                sigma2, sigma3 '''
        sigma3 = ( y_p - x_data) #calculate error at the results of input layer
        z_2 = self.add_bias_row(z_2) #add bias to row of the last layer
        sigma2 = self.weights_hidden.T.dot(sigma3)*self.der_activation(z_2) #do backpropagation
        sigma2 = sigma2[1:, :] #leave bias
        return sigma2, sigma3
    
    def update_weights(self, x_data, a_2, sigma2, sigma3, weights_t, weights_hidden_t):
        '''gradient calculation for backpropagation algorithm
                1. Calculate gradient for output layer 
                2. Calculate gradient for hidden layer 
                3. '''
        grad1 = sigma2.dot(x_data) #calculate gradient for hidden layer
        grad2 = sigma3.dot(a_2.T) #calculate gradient for output layer
        wght_drop = self.weights*self.drop
        delta_w1 = 2*self.mu*grad1 + self.mu*self.l1/self.p*np.sign(wght_drop)+self.mu*self.l2/self.p*wght_drop   #mulitply gradients by learning rate
        delta_w2 =  2*self.mu*grad2 + self.mu*self.l1/self.p*np.sign(self.weights_hidden)+self.mu*self.l2/self.p*self.weights_hidden      #mulitply gradients by learning rate
        self.weights -= (delta_w1 + (self.gamma*delta_w1) ) #subtract delta w_1 from weights at time t 
        self.weights_hidden -= (delta_w2 + (self.gamma*delta_w2) ) #subtract delta w_1 from hidden weights at time t 
        return delta_w1, delta_w2 
    
    def regul_l1(self):
        'regularization l2'''
        self.batch_loss[-1] += self.l1/(self.p)*(np.sum(abs(self.weights)) + np.sum(abs(self.weights_hidden)))
        
    def regul_l2(self):
        '''regularization l2'''
        self.batch_loss[-1] += self.l2/(2*self.p)*(np.sum(self.weights**2) + np.sum(self.weights_hidden**2))
            
    
    def train(self):
        '''main train function for training multilayer neural network'''
        print ('Auteencoder training')
        self.set_weights() # randomly set weights
        self.wght_bef = {'weights':self.weights, 'weights_hidden':self.weights_hidden}
        self.batches = self.p/self.batch_size
        delta_weights = np.zeros(self.weights.shape) #weights at time t
        delta_weights_hidden = np.zeros(self.weights_hidden.shape) #hidden weights at time t
        self.stop = False
        self.stop_ctr = 0
        while (self.epoch < self.n_iter and not self.stop):
            idx = np.random.permutation(self.train_set.shape[0]) #shuffling numbers in each epoch to prevent from adapting too much to pattern
            X_data = self.train_set[idx] #choose X_data after shuffling
            idx_split = np.array_split(range(self.p),self.batches) #split training array on smaller batches
            ctr = 0
            
            while ctr < len(idx_split) and not self.stop:
                self.dropout()
                idx = idx_split[ctr]
                x_data = X_data[idx]
                x_data, z_2, a_2, z_3, y_p = self.feedworward(x_data)
                bch_loss = round(mse(np.transpose(x_data), y_p),3)
                self.batch_loss.append(bch_loss)
                self.regul_l2()
                self.regul_l1()
                
                wht_bef = np.copy(self.weights)
                hid_wht_bef = np.copy(self.weights_hidden)
                sigma2, sigma3 = self.backpropagation(y_p, np.transpose(x_data), z_2, a_2)
                delta_weights, delta_weights_hidden = self.update_weights(x_data,\
                                                                          a_2, sigma2, sigma3,\
                                                                          delta_weights, delta_weights_hidden)
                self.hinton()
                ctr +=1
            self.train_loss.append(np.mean(self.batch_loss[self.epoch*ctr:self.epoch*(1+ctr)]))
            valid_data, z_2, a_2, z_3, y_p = self.feedworward(self.valid_set)
            self.val_loss_min = np.min(self.valid_loss) if len(self.valid_loss) > 1 else 10**6
            self.valid_loss.append(mse(np.transpose(valid_data), y_p))
            if self.valid_loss[-1] < self.val_loss_min:
                print ('Best weights swapping')
                self.wght_bef['weights'] = wht_bef
                self.wght_bef['hidden_weights'] = hid_wht_bef
            self.early_stopping()
            print ('train loss %f at epoch %d. valid loss %f at epoch %d \n' %(self.train_loss[-1], self.epoch,self.valid_loss[-1],  self.epoch) )    
            self.epoch += 1
        
        if self.save:
            np.save('weights_'+str(self.hidden)+'.npy', self.weights)
            np.save('weights_'+str(self.hidden)+'hidden.npy', self.weights_hidden)
    
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
    
    def dropout(self):
        '''dropout function which prevent from overfitting'''
        self.drop = np.random.binomial(1, self.drop_p, size = self.weights.shape)/self.drop_p
        self.weights *= self.drop
    
    def test(self, type = 'test'):
        '''method for testing data along with forward propagation algorithm'''
        if not hasattr(self,'weights') and not hasattr(self,'weights_hidden'):
            self.read_weights()
        if type == 'test':
            test_set = self.test_set
        elif type =='valid':
            test_set = self.valid_set
        elif type =='train':
            test_set = self.train_set
        x_data, z_2, a_2, z_3, y_p = self.feedworward(test_set)
        return y_p

    def get_acc(self, type ='test'):
        '''method for checking accuracy of data, either testing, validation or training'''
        y_p = self.test(type)
        if type == 'test':
            y_true = self.add_bias_col(self.test_set).T
        elif type == 'valid':
            y_true = self.add_bias_col(self.valid_set).T
        elif type == 'train':
            y_true = self.add_bias_col(self.train_set).T
        
        return np.sum(y_p- y_true)/y_p.shape[0]

    def set_activation_function(self, activation):
        '''function for setting activation function '''
        self.activation = {'relu':self.relu, 'sigmoid':self.sigmoid,'softmax':self.softmax}[activation]
        self.der_activation = {'relu':self.relu_der, 'sigmoid':self.sigmoid_der,'softmax_der':self.softmax_der}[activation]

    def relu(self, X):
        return np.where(X>0, X, 0)
    
    def relu_der(self, X):
        return np.where(X>0, 1, 0)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-self.beta*X))

    def sigmoid_der(self, X):
        return self.beta*self.sigmoid(X)*(1-self.sigmoid(X))
    
    def softmax(self, X):
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum(axis=0) # only difference
    
    def softmax_der(self, X):
        return self.softmax(X)*(1-self.softmax(X))
    
    def plot_single(self, y_p, true_lbl):
        fig, ax = plt.subplots()
        img = y_p.reshape(28, 28)
        ax.imshow(img, cmap='Greys', interpolation='nearest')
        ax.set_title('real: %d' %  true_lbl)
            
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        
    def plot_examples(self, y_p, path):
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        numbers = rnd.sample(range(y_p.shape[1]),25)
        ctr = 0
        
        for i in numbers:
            curr = y_p[1:,i]
            img = curr.reshape(28, 28)
            ax[ctr].imshow(img, cmap='Greys', interpolation='nearest')
            ax[ctr].set_title('real: %d' %  self.test_set_label[i])
            ctr +=1
            
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        fig.savefig(path+'Autoenc\\examples'+self.name+'.eps', format='eps',dpi=1000)
        plt.close()
    
    def plot_batch_loss(self,path):
        fig, ax = plt.subplots()
        ax.plot(range(1,len(self.batch_loss)+1), self.batch_loss, label = 'Zbiór paczek')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Strata')
        ax.set_title('Wykres straty dla autoenkodera przy %d iteracjach' % self.n_iter)
        ax.legend()
        fig.savefig(path+'Autoenc\\batchloss_'+self.name+'.eps', format='eps',dpi=1000)
        plt.close()
    
    def plot_loss(self,path):
        fig, ax = plt.subplots()
        ax.plot(range(1,self.epoch+1), self.valid_loss, label = 'Zbiór walidacyjny')
        ax.plot(range(1,self.epoch+1), self.train_loss, label = 'Zbiór treningowy')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Strata')
        ax.set_title('Wykres straty dla autoenkodera przy %d iteracjach' % self.n_iter)
        ax.legend()
        fig.savefig(path+'Autoenc\\loss_'+self.name+'.eps', format='eps',dpi=1000)
        plt.close()
        
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
    mu = 0.0001
    gamma = 0.5
    low, high = -0.05,0.05
    activation_name = 'sigmoid'
    n_iter = 200
    hidden = 10
    batch_size = 100
    l1 = 0.00001
    l2 = 0.00005
    spl_perc = 0.02
    erl_ctr = 10
    drop_p = 1
    anc = AutoEnc(mu = mu, gamma = gamma, low = low, high = high, activation_name = activation_name,\
                  n_iter = n_iter, hidden = hidden, batch_size = batch_size, l1 = l1, l2 = l2,\
                  erl_ctr = erl_ctr, spl_perc = spl_perc, drop_p = drop_p)
    anc.train()
    y_test = anc.test()
    anc.plot_examples(y_test,'ex6\\')
    anc.plot_pix_int()
    