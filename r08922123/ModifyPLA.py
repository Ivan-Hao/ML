import sys
import numpy as np
import random
import matplotlib.pyplot as plt

#Please initialize your algorithm with w = 0 and take sign(0) as -1. As a friendly reminder, remember to add x0 = 1 as always!
def Modify_PLA(data, test, epochs = 1126):
    error_on_test = []
    sign = lambda x : -1 if x <= 0 else 1

    for i in range(epochs):
        w_try = np.array([1,0,0,0,0],dtype='float32')
        update_count = 0       
        while update_count < 100 :
            for i in np.random.permutation(data.shape[0]): #random pick a vector i
                if sign(w_try.dot(data[i][0:-1])) == data[i][-1]:
                    continue
                else:
                    w_try = w_try + data[i][-1]*data[i][0:-1]
                    update_count +=1
                    if update_count == 100:
                        break
        #verify the performance of M_pla using the test set.
        e = np.matmul(test[:,0:5], w_try)
        f = np.where(np.where(e>0,1,-1) == test[:,-1],0,1)
        error_count = np.sum(f)
        error_on_test.append(error_count/test.shape[0])
    return error_on_test

def print_hist(count_list):
    #Plot a histogram to show the error rate versus frequency.
    plt.hist(count_list,bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('The error rate')
    plt.ylabel('Frequency')
    plt.title('ModifyPLA homework')
    plt.show()

if __name__ == '__main__':

    #numpy.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    data_raw_train = np.loadtxt(fname = sys.argv[1], dtype= 'float32')
    data_raw_test = np.loadtxt(fname = sys.argv[2], dtype= 'float32')
    data_train = np.concatenate((np.ones((data_raw_train.shape[0],1)), data_raw_train), axis =1) 
    data_test = np.concatenate((np.ones((data_raw_test.shape[0],1)), data_raw_test), axis =1)     
    e_result = Modify_PLA(data_train, data_test)
    print('Average error rate : ',np.average(e_result))
    print_hist(e_result)