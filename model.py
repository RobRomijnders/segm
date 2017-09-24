import tensorflow as tf

class Model():
    def __init__(self, batch_size, dropout):
        self.image = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], 'input_images')
        self.segm_map = tf.placeholder(tf.float32, [batch_size, 32, 32], 'output_segm_map')
        # Make dropout also a placeholder !!!

        num_filters = [10, 12, 8]

        W1 = tf.get_variable('weight1', [5, 5, 3, num_filters[0]])
        b1 = tf.get_variable('bias1',[num_filters[0],])
        a1 = tf.nn.conv2d(self.image, W1,[1,1,1,1], "SAME")+ b1
        h1 = tf.nn.relu(tf.nn.dropout(a1, dropout))

        W2 = tf.get_variable('weight2', [5, 5, num_filters[0], num_filters[1]])
        b2 = tf.get_variable('bias2',[num_filters[1],])
        a2 = tf.nn.conv2d(h1, W2,[1,1,1,1], "SAME")+ b2
        h2 = tf.nn.relu(tf.nn.dropout(a2, dropout))

        W3 = tf.get_variable('weight3', [5, 5, num_filters[1], num_filters[2]])
        b3 = tf.get_variable('bias3',[num_filters[2],])
        a3 = tf.nn.conv2d(h2, W3,[1,1,1,1], "SAME")+ b3
        h3 = tf.nn.relu(tf.nn.dropout(a3, dropout))

        W4 = tf.get_variable('weight4', [5, 5, num_filters[2], 1])
        b4 = tf.get_variable('bias4',[1,])
        a4 = tf.squeeze(tf.nn.conv2d(h3, W4,[1,1,1,1], "SAME")+ b4, 3)
        self.h4 = tf.nn.sigmoid(a4)

        cost_per_pixel = self.segm_map*tf.log(self.h4+1E-13) + (1-self.segm_map)*tf.log(1-self.h4+1E-13)  #add +1E-13 to prevent log(0)
        self.total_loss = -1*tf.reduce_mean(cost_per_pixel)

        ## Optimizer
        self.train_step = tf.train.AdamOptimizer(0.005).minimize(self.total_loss)