import numpy as np 
import matplotlib.pyplot as plt
import sys

train_data = np.loadtxt('hw3_train.dat')
test_data = np.loadtxt('hw3_test.dat')

sigmoid = lambda x : 1/(1 + np.exp(-x))
crossentropy = lambda x,y,w : np.log(1+np.exp(-y*np.dot(x,w))) 

learning_rate = float(sys.argv[1])

epochs = 2000
batch_size = 1000
sample_size = train_data.shape[0]

w_initial = np.random.rand(20)-0.5
S_w_initial = np.copy(w_initial)
fix_w_initial = np.copy(w_initial)

GD_train_accuracy = []
GD_train_error = []
GD_train_loss = []

S_train_accuracy = []
S_train_error = []
S_train_loss = []

GD_test_accuracy = [] 
S_test_accuracy = []
f_test_accuracy = []
GD_test_error = [] 
S_test_error = []
f_test_error =[]


fix_SGD=[]
fix_error_SGD=[]
f_loss=[]

for i in range(epochs):
    '''
    for j in range(sample_size//batch_size):
        
        front = j*batch_size
        end = (j+1)*batch_size
        
        gradient = sigmoid(np.dot(train_data[front:end,:-1],w_initial)*(-train_data[front:end,-1]))
        gradient = gradient * (-train_data[front:end,-1]) 
        gradient = np.tile(np.reshape(gradient,(batch_size,1)),(1,20)) * train_data[front:end,:-1]
        w_initial -= learning_rate*np.average(gradient,axis=0)
    '''
    f_SGD = i%sample_size

    gradient = sigmoid(np.dot(train_data[:,:-1],w_initial)*(-train_data[:,-1]))
    gradient = gradient * (-train_data[:,-1]) 
    gradient = np.tile(np.reshape(gradient,(batch_size,1)),(1,20)) * train_data[:,:-1]
    w_initial -= learning_rate*np.average(gradient,axis=0)

    SGD = np.random.randint(sample_size)
    S_gradient = sigmoid(np.dot(train_data[SGD:SGD+1,:-1],S_w_initial)*(-train_data[SGD:SGD+1,-1]))
    S_gradient = S_gradient * (-train_data[SGD:SGD+1,-1]) * train_data[SGD:SGD+1,:-1]
    S_w_initial -= learning_rate*np.average(S_gradient,axis=0)

    f_gradient = sigmoid(np.dot(train_data[f_SGD:f_SGD+1,:-1],fix_w_initial)*(-train_data[f_SGD:f_SGD+1,-1]))
    f_gradient = f_gradient * (-train_data[f_SGD:f_SGD+1,-1])  * train_data[f_SGD:f_SGD+1,:-1]
    fix_w_initial -= learning_rate*np.average(f_gradient,axis=0)

    #fSGD
    f_y_predict = np.where(sigmoid(np.dot(train_data[:,:-1],fix_w_initial)) >0.5, 1 ,-1)
    f_precision = np.average(f_y_predict == train_data[:,-1])
    f_error = 1-f_precision
        

    #GD
    y_predict = np.where(sigmoid(np.dot(train_data[:,:-1],w_initial)) >0.5, 1 ,-1)
    precision = np.average(y_predict == train_data[:,-1])
    error = 1-precision

    #SGD
    S_y_predict = np.where(sigmoid(np.dot(train_data[:,:-1],S_w_initial)) >0.5, 1 ,-1)
    S_precision = np.average(S_y_predict == train_data[:,-1])
    S_error = 1-S_precision

    #fSGD
    fix_SGD.append(f_precision)
    fix_error_SGD.append(f_error)
    fix_loss = np.average(crossentropy(train_data[:,:-1],train_data[:,-1],fix_w_initial))
    f_loss.append(fix_loss) 


    #GD
    GD_loss = np.average(crossentropy(train_data[:,:-1],train_data[:,-1],w_initial))      
    GD_train_loss.append(GD_loss)
    GD_train_accuracy.append(precision)
    GD_train_error.append(error)

    #SGD
    S_loss = np.average(crossentropy(train_data[:,:-1],train_data[:,-1],S_w_initial))      
    S_train_loss.append(S_loss)
    S_train_accuracy.append(S_precision)
    S_train_error.append(S_error)


    #fSGD
    f_out_precision = np.where(sigmoid(np.dot(test_data[:,:-1],fix_w_initial)) >0.5, 1 ,-1)
    f_out_acc = np.average(f_out_precision == test_data[:,-1])
    f_out_error = 1-f_out_acc
    f_test_accuracy.append(f_out_acc)
    f_test_error.append(f_out_error)


    #GD test
    out_precision = np.where(sigmoid(np.dot(test_data[:,:-1],w_initial)) >0.5, 1 ,-1)
    out_acc = np.average(out_precision == test_data[:,-1])
    out_error = 1-out_acc
    GD_test_accuracy.append(out_acc)
    GD_test_error.append(out_error)    

    #SGD test
    S_out_precision = np.where(sigmoid(np.dot(test_data[:,:-1],S_w_initial)) >0.5, 1 ,-1)
    S_out_acc = np.average(S_out_precision == test_data[:,-1])
    S_out_error = 1-S_out_acc  
    S_test_accuracy.append(S_out_acc)
    S_test_error.append(S_out_error)


# GD and loss
plt.figure()
plt.plot(GD_train_accuracy,'g-',label="GD accuracy" )
plt.plot(GD_train_loss,'b-', label = "GD loss")
plt.title("Training precision and error rate")
plt.xlabel("T")
plt.ylabel("value")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#SGD and loss
plt.figure()
plt.plot(S_train_accuracy,'g-',label="SGD accuracy" )
plt.plot(S_train_loss,'b-', label = "SGD loss")
plt.title("Training precision and error rate")
plt.xlabel("T")
plt.ylabel("value")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#fSGD and loss
plt.figure()
plt.plot(fix_SGD,'g-',label="fSGD accuracy" )
plt.plot(fix_error_SGD,'b-', label = "fSGD loss")
plt.title("Training precision and error rate")
plt.xlabel("T")
plt.ylabel("value")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#Ein GD vs SGD vs fSGD
plt.figure()
plt.plot(S_train_error,'g-', label = "SGD")
plt.plot(GD_train_error,'b-', label = "GD")
plt.plot(fix_error_SGD,'r-', label="fSGD")
plt.title("Ein GD and SGD and fSGD")
plt.xlabel("T")
plt.ylabel("Ein")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#acc GD and SGD and fSGD
plt.figure()
plt.plot(GD_train_accuracy,'g-',label="GD accuracy" )
plt.plot(S_train_accuracy,'b-', label = "SGD accuracy")
plt.plot(fix_SGD,'r-', label = "fSGD accuracy")
plt.title("GD and SGD and fSGD accuracy")
plt.xlabel("T")
plt.ylabel("accuracy")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#test GD and SGD and fSGD
plt.figure()
plt.plot(GD_test_accuracy,'g-',label="test GD accuracy" )
plt.plot(S_test_accuracy,'b-', label = "test SGD accuracy")
plt.plot(f_test_accuracy,'r-', label = "test fSGD accuracy")
plt.title("Test GD and SGD and fSGD accuracy")
plt.xlabel("T")
plt.ylabel("accuracy")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()

#test GD and SGD and fSGD
plt.figure()
plt.plot(GD_test_error,'g-',label="test GD error" )
plt.plot(S_test_error,'b-', label = "test SGD error")
plt.plot(f_test_error,'r-', label = "test fSGD error")
plt.title("Eout")
plt.xlabel("T")
plt.ylabel("Eout")
plt.legend(loc=0)
plt.tick_params(axis='both', color='red')
plt.show()