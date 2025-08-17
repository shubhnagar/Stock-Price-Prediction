from matplotlib import pyplot as plt
import numpy as np
import random
import os
random.seed(45)

num_coins = 100
def toss(num_trials):
    
    '''
    num_trials: number of trials to be performed.
    
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial

    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''
    global num_coins
    results = []
    
    ## Write your code here
    for i in range(num_trials):
        h_count = 0
        for j in range(num_coins):
            toss = random.choice([0, 1])
            if (toss == 1):
                h_count += 1
        results.append(h_count)
        
    return np.array(results)
    

def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    fig, axs = plt.subplots(figsize =(10, 7), tight_layout=True)
    
    ## Write your code here

    axs.hist(trial, bins = 100)

    axs.set_xlabel('Number of Heads')
    axs.set_ylabel('Number of Trials')
    axs.set_title('Head Count Over Multiple Trials')

    cwd = os.getcwd()
    print(cwd)
    file_loc = os.path.join(cwd, 'histograms', f'histogram_{num_trials}.png')
    plt.savefig(file_loc)



if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000,100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)
