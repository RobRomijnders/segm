# Simple Semantic Segmentation
This repository implements the minimal code to do semantic segmentation. 

In semantic segmentation, each pixel of an input image must be assigned to an output class. For example, in segmenting tumors in CT-scans, each pixel must be assigned to class "healthy tissue" and "pathological tissue". Other examples include street scene segmentation, where each pixel belings to "background", "pedestrian", "car, "building" and so on. 

Most applications of semantic segmentation work within a larger pipeline. Therefore, semantic segmentation might seem a complicated algorithm. However, in this repository we use about 300-400 lines of code to show the core of the segmentation.

# Model
Each pixel must be assigned a label. Therefore, the output size is of same order as the input size. Say the input is __Height x Width x Channels__, then the output is __Height x Width x Classes__. Usually, other problems in machine learning require only a small output. For example, classification requires only __num_classes__ output and regression requires a single number.
To deal with these large outputs, two solutions are

 * Divide the problem into __Height*Width__ small problems. Train some algorithm to classify a patch into a single label and run this algorithm over the entire input.
 * Use a neural network to connect each pixel in the input with each pixel in the output. This repository implements this approach.

# Data
We pick a small dataset in order to show the concept. In fact, we create our own dataset. Most public datasets involve images of thousands of pixels. They require much more computation power than the average person has on his/her laptop. 

Our data combines the digits from MNIST with the background of CIFAR10. This resembles the usual foreground-background segmentation task. Below are some examples from the data.

The datagenerator randomly overlays a CIFAR image and an MNIST digit. The images and the offsets are chosen from uniform distributions.

# Convolutional neural network
As we deal with images, we model with a convolutional neural network (CNN). A CNN maps with filters between layers of neurons. This is a natural choice for images. Most concepts in the natural consist of hierarchical patterns. Moreover, with CNN's we can scale to arbitrary height and widths. You can read more about that in [this paper](https://arxiv.org/abs/1411.4038).

# Code
The code consist of three scripts

  * _model.py_ describes the model using Tensorflow library
  * *load_data.py* describes the generation and sampling of the data
  * *main.py* describes the main code: instantiating the model and data generator and trains the model.

# Further reading

  * [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
  * [Chapter 9: Convolutional Neural Networks; Deep learning book](http://www.deeplearningbook.org/)
 
Note that you have to download the data yourself:
  * [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
  * [MNIST](http://yann.lecun.com/exdb/mnist/)
