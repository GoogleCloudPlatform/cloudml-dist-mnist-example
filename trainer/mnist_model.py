# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import tensorflow as tf


def get_model(x, keep_prob=None, training=False):
  num_filters1 = 32
  num_filters2 = 64

  with tf.name_scope('cnn'):
    with tf.name_scope('convolution1'):
      x_image = tf.reshape(x, [-1,28,28,1])
      W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filters1],
                                                stddev=0.1))
      h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                             strides=[1,1,1,1], padding='SAME')
      b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
      h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
      h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1],
                               strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('convolution2'):
      W_conv2 = tf.Variable(
                  tf.truncated_normal([5,5,num_filters1,num_filters2],
                                      stddev=0.1))
      h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                             strides=[1,1,1,1], padding='SAME')
      b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
      h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
      h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1],
                               strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('fully-connected'):
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])
      num_units1 = 7*7*num_filters2
      num_units2 = 1024
      w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
      b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
      hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

    with tf.name_scope('output'):
      if training:
        hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
      else:
        hidden2_drop = hidden2
      w0 = tf.Variable(tf.zeros([num_units2, 10]))
      b0 = tf.Variable(tf.zeros([10]))
      p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

  tf.summary.histogram('conv_filters1', W_conv1)
  tf.summary.histogram('conv_filters2', W_conv2)

  return p


def get_trainer(p, t, global_step):
  with tf.name_scope('optimizer'):
    loss = -tf.reduce_sum(t * tf.log(p), name='loss')
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)

  with tf.name_scope('evaluator'):
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32), name='accuracy')

  return train_step, loss, accuracy
