import torch
import numpy as np
from sklearn.manifold import SpectralEmbedding, TSNE

%matplotlib notebook 
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

# Laplacian Eigenmaps

# New Cell
test_data=torch.load('../../data/mnist/test_data.pt')
test_label=torch.load('../../data/mnist/test_label.pt')
test_data = test_data.reshape(10000,784)
test_data = np.array(test_data)
embedding = SpectralEmbedding(n_components=3)
test_data_new = embedding.fit_transform(test_data)
# New Cell
fig = pylab.figure(1)
ax = Axes3D(fig)
size_vertex_plot = 1
ax.scatter(test_data_new[:,0], test_data_new[:,1], test_data_new[:,2], s=size_vertex_plot*np.ones(10000), c=test_label)
ax.view_init(24,-105)
ax.axis('off')
plt.title('3D Visualization of MNIST with LapEigMaps') 
pyplot.show()
# New Cell
test_data=torch.load('../../data/cifar/test_data.pt')
test_label=torch.load('../../data/cifar/test_label.pt')
test_data = test_data.reshape(10000,3072)
test_data = np.array(test_data)
embedding = SpectralEmbedding(n_components=3)
test_data_new = embedding.fit_transform(test_data)
# New Cell
fig = pylab.figure(1)
ax = Axes3D(fig)
size_vertex_plot = 1
ax.scatter(test_data_new[:,0], test_data_new[:,1], test_data_new[:,2], s=size_vertex_plot*np.ones(10000), c=test_label)
ax.view_init(24,-105)
ax.axis('off')
plt.title('3D Visualization of CIFAR-10 with LapEigMaps') 
pyplot.show()

# TSNE

# New Cell
test_data=torch.load('../../data/mnist/test_data.pt')
test_label=torch.load('../../data/mnist/test_label.pt')
test_data = test_data.reshape(10000,784)
test_data = np.array(test_data)
embedding = TSNE(n_components=3)
test_data_new = embedding.fit_transform(test_data)
# New Cell
fig = pylab.figure(11)
ax = Axes3D(fig)
size_vertex_plot = 10
ax.scatter(test_data_new[:,0], test_data_new[:,1], test_data_new[:,2], s=size_vertex_plot*np.ones(10000), c=test_label)
plt.title('3D Visualization of MNIST with t-SNE') 
pyplot.show()
# New Cell
test_data=torch.load('../../data/cifar/test_data.pt')
test_label=torch.load('../../data/cifar/test_label.pt')
test_data = test_data.reshape(10000,3072)
test_data = np.array(test_data)
embedding = TSNE(n_components=3)
test_data_new = embedding.fit_transform(test_data)
# New Cell
fig = pylab.figure(11)
ax = Axes3D(fig)
size_vertex_plot = 10
ax.scatter(test_data_new[:,0], test_data_new[:,1], test_data_new[:,2], s=size_vertex_plot*np.ones(10000), c=test_label)
plt.title('3D Visualization of CIFAR-10 with t-SNE') 
pyplot.show()
