import numpy as np
import random
import matplotlib.pyplot as plt

#Please initialize your algorithm with w = 0 and take sign(0) as -1. As a friendly reminder, remember to add x0 = 1 as always!
def PLA(data, epochs = 1126):
    update_list = []
    sign = lambda x : -1 if x <= 0 else 1
    for i in range(epochs):
        w_initial = np.zeros(5)
        update_count = 0
        while 1 :
            no_update = True
            for i in random.sample(range(len(data)),len(data)): #random pick a vector i
                if sign(w_initial.dot(data[i][0:-1])) == data[i][-1]:
                    continue
                else:
                    w_initial += data[i][-1]*data[i][0:-1]
                    update_count += 1   
                    no_update = False
            if no_update == True:
                update_list.append(update_count)        
                break
        
    return (w_initial,update_list)

def print_hist(count_list):
    #Plot a histogram to show the number of updates versus the frequency of the number.
    plt.hist(count_list,bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('The number of updates')
    plt.ylabel('The frequency of the number')
    plt.title('PLA homework')
    plt.show()

if __name__ == '__main__':
    train_data = []
    with open('hw1_6_train.dat.txt') as data_obj:
        for vector in data_obj :
            train_data.append(list(map(float,[1]+vector.rstrip().replace('\t',' ').split())))
    
    training_data = np.array(train_data)
    (w_result,count_result) = PLA(training_data)
    print('Average updates count = ', sum(count_result)/len(count_result))
    print_hist(count_result)
    