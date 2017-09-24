import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, exists
from mnist import MNIST

def unpickle(file):
    """
    unpickles cifar batches from the encoded files. Code from
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpack_cifar(direc):
    """
    the data comes in batches. So this function concatenates the data from the batches
    :param direc: directory where the batches are located
    :return:
    """
    assert exists(direc), "directory does not exist"
    X,y = [], []
    for filename in os.listdir(direc):
        if filename[:5] == 'data_':
            data = unpickle(join(direc, filename))
            X.append(data[b'data'].reshape((10000,3,32,32)))
            y += data[b'labels']
    assert X, "No data was found in '%s'. Are you sure the CIFAR10 data is there?"%direc

    X = np.concatenate(X, 0)
    X = np.transpose(X, (0,2,3,1)).astype(np.float32)
    return X,y

def plot_data(X):
    """
    Generic function to plot the images in a grid
    of num_plot x num_plot
    :param X:
    :return:
    """
    plt.figure()
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for i in range(num_plot):
        for j in range(num_plot):
            idx = np.random.randint(0, X.shape[0])
            ax[i,j].imshow(X[idx])
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)

def unpack_mnist(direc):
    """
    Unpack the MNIST data and put them in numpy arrays
    :param direc:
    :return:
    """
    assert exists(direc), "directory does not exist"
    try:
        mndata = MNIST(direc)
        images, labels = mndata.load_training()
    except FileNotFoundError as e:
        print('Make sure that you have downloaded the data and put in %s\n Also make sure that the spelling is correct. \
              the MNIST data comes in t10k-images.idx3-ubyte or t10k-images-idx3-ubyte. We expect the latter'%(direc))
        raise FileNotFoundError(e)
    X_mnist = np.array(images).reshape(60000, 28, 28)
    y_mnist = np.array(labels)

    X_mnist = X_mnist.astype(np.float32)/np.max(X_mnist)
    return X_mnist, y_mnist

def split_data(X, ratio=0.60):
    num_samples = X.shape[0]
    ind_split = int(ratio*num_samples)
    permutation = np.random.permutation(num_samples)
    return X[permutation[:ind_split]], X[permutation[ind_split:]]

class Datagen():
    """
    Object to sample the data that we can segment. The sample function combines data
    from MNIST and CIFAR and overlaps them
    """
    def __init__(self, direc_mnist, direc_cifar):
        ## Unpack the data
        X_cifar,y_cifar = unpack_cifar(direc_cifar)
        X_mnist, y_mnist = unpack_mnist(direc_mnist)

        self.data = {'mnist':{'train':None, 'val':None}, 'cifar':{'train':None, 'val':None}}

        self.data['mnist']['train'], self.data['mnist']['test'] = split_data(X_mnist)
        self.data['cifar']['train'], self.data['cifar']['test'] = split_data(X_cifar)

    def sample(self, batch_size, norm = True, dataset = 'train'):
        """
        Samples a batch of data. It randomly inserts the MNIST images into cifar images
        :param batch_size:
        :param norm: indicate wether to normalize the data or not
        :return:
        """
        assert dataset in ['train', 'test']
        idx_cifar = np.random.choice(self.data['cifar'][dataset].shape[0], batch_size)
        idx_mnist = np.random.choice(self.data['mnist'][dataset].shape[0], batch_size)
        im_cifar = self.data['cifar'][dataset][idx_cifar]
        im_mnist = self.data['mnist'][dataset][idx_mnist][:, ::2, ::2]
        size_mnist = 14

        mnist_mask =  np.greater(im_mnist, 0.3, dtype = np.float32)
        im_mnist *= mnist_mask

        width_start = np.random.randint(0,32-size_mnist,size=(batch_size))
        height_start = np.random.randint(0,32-size_mnist,size=(batch_size))
        color_range = 200

        mnist_batch = np.repeat(np.expand_dims(im_mnist*color_range,3),3,3)

        segm_maps = np.zeros((batch_size, 32, 32))

        for i in range(batch_size):
            im_cifar[i, width_start[i]:width_start[i]+size_mnist, height_start[i]:height_start[i]+size_mnist] += mnist_batch[i]
            segm_maps[i, width_start[i]:width_start[i]+size_mnist, height_start[i]:height_start[i]+size_mnist] += mnist_mask[i]
        im_cifar = np.clip(im_cifar, 0, 255)

        if norm:
            im_cifar = (im_cifar-130.)/70.
        return im_cifar, segm_maps


if __name__ == "__main__":
    dg = Datagen('data/mnist', 'data/cifar')
    data, segm_maps = dg.sample(32)
    plot_data(data)
