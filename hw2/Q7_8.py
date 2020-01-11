import numpy as  np
import sys
import matplotlib.pyplot as plt

def plot_hist(plot_list,data_size,experiment_times):
    #Plot a histogram to show the error rate versus frequency.
    plt.hist(plot_list,bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('The difference between error_in and error_out')
    plt.ylabel('Frequency')
    plt.title('The decisoin stump')
    plt.annotate('data size: '+ str(data_size) + '\n' +'experiment times: '+ str(experiment_times), xy=(0.05,0.9),xycoords='axes fraction')
    plt.show()


def decision_stump(data, label):
    theta = np.sort(data)

    X_tile = np.tile(data, (theta.size, 1)) #length * length matrix
    theta_T = np.reshape(theta,(theta.size,1))
 
    T_tile = np.tile(theta_T,(1,theta.size)) #length * length matrix 
    label_prediction = np.where((X_tile-T_tile)>0, 1,-1) #label prediction

    E_tile = label_prediction != label #find error label
    error = np.sum(E_tile, axis=1) #compute error times
    error_min = np.min(error)
    error_max = np.max(error)

    if error_min <= theta.size - error_max: # find the cut point
        re_theta =  theta[np.argmin(error)]
        re_error = error_min/theta.size
        re_s = 1
    else:
        re_theta = theta[np.argmax(error)]
        re_error = (theta.size-error_max)/theta.size
        re_s = -1
    return re_s,re_theta,re_error    

def artifitial_data(data_size):
    x = np.random.uniform(-1, 1, data_size)
    y = np.where(x>0,1,-1)
    noise = np.random.uniform(0, 1, data_size)
    y[noise <= 0.2] *= -1
    return x, y



if __name__ == '__main__':
    data_size = int(sys.argv[1])
    experiment_times = int(sys.argv[2])
    total_in = 0 
    total_out = 0
    error = lambda _s, _theta: 0.5+0.3*_s*(np.fabs(_theta)-1)
    plot_list = []
    for i in range(experiment_times):
        data, label = artifitial_data(data_size)
        s, theta, error_in = decision_stump(data, label)
        error_out = error(s,theta)
        plot_list.append(error_in-error_out)
        total_in += error_in
        total_out += error_out

    print(' average error_in: ', total_in/experiment_times)
    print(' average error_out: ', total_out/experiment_times)
    plot_hist(plot_list,data_size,experiment_times)