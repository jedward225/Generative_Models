import numpy as np
import random
import os
from utils import set_seed, visualize_data

def mean_generate(x_range = [-5, 35], y_range = [-12, 32]):
    x = np.arange(x_range[0], x_range[1], 5)
    y = np.arange(y_range[0], y_range[1], 10)
    points = [np.array((i, j)) for i in x for j in y]
    return points

def data_generate(num = 15000):
    means = mean_generate()
    for type in ['train', 'val', 'test']:
        data = []
        for i in range(num):
            mean = random.choice(means)
            data.append(np.random.normal(loc=mean, scale=1))
        
        # visualization
        os.makedirs('./data', exist_ok=True)
        visualize_data(data, f'./data/{type}.png')
        
        # save the data
        data = np.array(data)
        np.save(f'./data/{type}.npy', data)
    
if __name__ == "__main__":
    set_seed(2024)
    data_generate()
