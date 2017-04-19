# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([784])
  image = tf.cast(image, tf.float32) * (1. / 255)
  label = tf.cast(features['label'], tf.int32)

  return image, label


def input_fn(filename, batch_size=100, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs)

  image, label = read_and_decode(filename_queue)
  images, labels = tf.train.batch(
      [image, label], batch_size=batch_size,
      capacity=1000 + 3 * batch_size)

  return {'image': images}, labels


def get_input_fn(filename, num_epochs=None, batch_size=100):
  return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
  # Input Layer
  input_layer = tf.reshape(features['image'], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN))

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001, optimizer="Adam")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(mode=mode, loss=loss, train_op=train_op,
                                 predictions=predictions)


def build_estimator(model_dir):
  return learn.Estimator(
           model_fn=_cnn_model_fn,
           model_dir=model_dir,
           config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def get_eval_metrics():
  return {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                       prediction_key="classes")
  }


def serving_input_fn():
  feature_placeholders = {'image': tf.placeholder(tf.float32, [None, 784])}
  features = {
    key: tensor
    for key, tensor in feature_placeholders.items()
  }    
  return learn.utils.input_fn_utils.InputFnOps(
    features,
    None,
    feature_placeholders
  )
