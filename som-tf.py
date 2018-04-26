#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

class SOM():
    def __init__(self, m, n, dim, num_iterations, eta = 0.5, sigma = None):
        """
        m x n : The dimension of 2D lattice in which neurons are arranged
        dim : Dimension of input training data
        num_iterations :  Total number of training iterations
        eta : learning rate
        sigma : The radius of neighourhood function
        """
        self._m = m
        self._n = n
        self._neighbourhood = []
        self._topography = []
        self._num_iterations = int(num_iterations)
        self._learned = False
        
        if sigma is None:
            sigma = max(m,n)/2.0 # Constant radius
        else:
            sigma = float(sigma)
            
        self._graph = tf.Graph()
        
        # Build Computation Graph of SOM
        with self._graph.as_default():
            #Weight matrix and the topography of neurons
            self._W = tf.Variable(tf.random_normal([m*n, dim], seed = 5))
            self._topography = tf.constant(np.array(list(self._neuron_location(m,n))))
            
            # Placeholder for training data
            self._X = tf.placeholder('float', [dim])
            
            # Placeholder to keep track of number of iterations
            self._iter = tf.placeholder('float')
            
            # Finding the Winner and its location            
            d = tf.sqrt(tf.reduce_sum(
                    tf.pow(self._W - tf.stack(
                            [self._X for i in range(m*n)]), 2),1))
            
            self.WTU_idx = tf.argmin(d,0)
            slice_start = tf.pad(tf.reshape(self.WTU_idx, [1]), np.array([[0,1]]))
            self.WTU_loc = tf.reshape(tf.slice(self._topography, slice_start, tf.constant(np.array([1,2]))),[2])
            #self.bd = slice_start
            self.bd2 = self.WTU_loc
            
            # Change learning rate and radius as a function of iterations
            learning_rate = 1 - self._iter/self._num_iterations
            _eta_new = eta * learning_rate
            _sigma_new = sigma * learning_rate
            
            # Calculating Neighbourhood function
            distance_square = tf.reduce_sum(tf.pow(tf.subtract(self._topography, tf.stack([self.WTU_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(distance_square, 'float32'), tf.pow(_sigma_new, 2))))
            
            # multiply learning rate with neighbourhood func
            eta_into_Gamma = tf.multiply(_eta_new, neighbourhood_func)
            
            # Shape it so that it can be multiplied to calculate dW
            weight_multiplier = tf.stack([tf.tile(tf.slice(eta_into_Gamma, np.array([i]), np.array([1])), [dim]) for i in range(m * n)])
            delta_W = tf.multiply(weight_multiplier, tf.subtract(tf.stack([self._X for i in range(m*n)]), self._W))
            new_W = self._W + delta_W
            self._training = tf.assign(self._W, new_W)
            
            # Initialize all variables
            init = tf.global_variables_initializer()
            self._sess = tf.Session()
            self._sess.run(init)
            
    def fit(self, X):
        """
        Function to carry out training
        """
        for i in range(self._num_iterations): 
            for x in X:
                self._sess.run(self._training, feed_dict = {self._X:x, self._iter: i})
                #idx = self._sess.run([self.bd, self.bd2], feed_dict = {self._X:x})
                #print(i)
                #print(idx[0])
                #print(idx[1])
                
            # Store a centroid grid for easy retreival
            centroid_grid = [[] for i in range(self._m)]
            self._Wts = list(self._sess.run(self._W))
            self._locations = list(self._sess.run(self._topography))
            for j, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._Wts[j])
            self._centroid_grid = centroid_grid
            
            self._learned = True
            
            if i%10 == 0:
                print('Iteration ', i)
                #print(centroid_grid)
                #plt.imshow(centroid_grid)
            
    def winner(self, x):
        idx = self._sess.run([self.WTU_idx, self.WTU_loc], feed_dict = {self._X:x})
        return idx
        
    def _neuron_location(self, m, n):
        """
        Function to generate the 2d lattice of neurons
        """
        for i in range(m):
            for j in range(n):
                yield np.array([i,j])
                
    def get_centroids(self):
        """
        Function to return a list of 'm lists, with each inner list containing
        """
        if not self._learned:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
    
    def map_vects(self, X):
        """
        Function to map each input vector to the relevant neuron in the lattice
        """
        if not self._learned:
            raise ValueError("SOM not trained yet")
        to_return = []
        print(X)
        for vect in X:
            min_index = min([i for i in range(len(self._Wts))],
                             key=lambda x: np.linalg.norm(vect - self._Wts[x]))
            print(min_index)
            to_return.append(self._locations[min_index])
            
        return to_return
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#"""
      
from matplotlib import pyplot as plt
 
#Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

n_dim = 3
som = SOM(m = 30, n = 30, dim = n_dim, num_iterations = 20, sigma = 10.0)
som.fit(colors)


# Gete output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(colors)

#Plot
plt.imshow(image_grid)
plt.title('Color Grid SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center', )# bbox=dict(f))
                

#"""


# Reading input data from file



import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('datos_de_entrenamiento.csv', encoding='utf-8')
#data_df = df.loc[:, 'Temperatura':'Conductividad  Ms/m']
data_df = df.loc[:, 'Temperatura':'OD']
normalized_data = normalize(data_df).values
names = df['ID sitio']
scores = df['Calidad']
n_dim = len(data_df.columns)

# Starting self organizing map or winner takes all units
som = SOM(m=30, n=30, dim = n_dim, num_iterations=20, sigma = 10.0)
som.fit(normalized_data)

# Gete output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(normalized_data)
grid_values = np.array(image_grid)

foo = grid_values.reshape((900,n_dim))
foo_std = StandardScaler().fit_transform(foo)
pca = PCA(n_components=3)
x = pca.fit_transform(foo)
bar = x.reshape((30,30,3))

plt.imshow(bar)
"""
grid_map = []
for row in grid_values:
    print(row)
    pca = PCA(n_components=3)
    x = pca.fit_transform(row)
    grid_map.append(x)
"""

    
#image = grid_values[:,:,0:3]
#Plot
#plt.imshow(image)
plt.title('Color Grid SOM')
for i, m in enumerate(mapped):
    #plt.text(m[1], m[0], names[i], ha='center', va='center', )# bbox=dict(f))
    plt.text(m[1], m[0], scores[i])
    

t = data_df[0:4].columns

img = grid_values[:,:,0]
plt.title(t[0])
plt.imshow(img)
img = grid_values[:,:,1]
plt.title(t[1])
plt.imshow(img)
img = grid_values[:,:,2]
plt.title(t[2])
plt.imshow(img)
img = grid_values[:,:,3]
plt.title(t[3])
plt.imshow(img)

"""
data  = normalize(df[['R', 'G', 'B']]).values
name  = df['Color-Name'].values
n_dim = len(df.columns) - 1

#Data for training
colors = data
color_names = name

#For plotting the images
som = WTU(m = 30, n = 30, dim = n_dim, num_iterations = 400, sigma = 10.0)
som.fit(colors)


# Gete output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(colors)

#Plot
plt.imshow(image_grid)
plt.title('Color Grid SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center', )# bbox=dict(f))
    
"""