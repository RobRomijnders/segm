from model import Model
from load_data import Datagen, plot_data
import tensorflow as tf
from util import plot_segm_map, calc_iou
import numpy as np

batch_size = 64
dropout = 0.7

dg = Datagen('data/mnist', 'data/cifar')
data, segm_map = dg.sample(batch_size)
model = Model(batch_size, dropout)

num_iter = 500

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for iter in range(num_iter):
    data_batch, segm_map_batch = dg.sample(batch_size)
    train_loss, _ = sess.run([model.total_loss, model.train_step], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})

    if iter%50 == 0:
        data_batch, segm_map_batch = dg.sample(batch_size, dataset='test')
        test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
        print('iter %5i/%5i loss is %5.3f and mIOU %5.3f'%(iter, num_iter, test_loss, calc_iou(segm_map_batch, segm_map_pred)))

#Final run
data_batch, segm_map_batch = dg.sample(batch_size, dataset='test')
test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
plot_segm_map(data_batch, segm_map_batch, segm_map_pred)
