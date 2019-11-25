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

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def read_tfrecord(serialized_example):
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

  return {'inputs': image}, label


def input_fn(filename, batch_size=100):
  dataset = tf.data.TFRecordDataset([filename])
  dataset = dataset.map(read_tfrecord)
  dataset = dataset.repeat().batch(batch_size) 
  return dataset


def get_input_fn(filename, batch_size=100):
  return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
  # Input Layer
  input_layer = tf.reshape(features['inputs'], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == Modes.TRAIN))

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  # Define operations
  if mode in (Modes.INFER, Modes.EVAL):
    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

  if mode in (Modes.TRAIN, Modes.EVAL):
    global_step = tf.contrib.framework.get_or_create_global_step()
    label_indices = tf.cast(labels, tf.int32)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(label_indices, depth=10), logits=logits)
    tf.compat.v1.summary.scalar('OptimizeLoss', loss)

  if mode == Modes.INFER:
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs)

  if mode == Modes.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  if mode == Modes.EVAL:
    eval_metric_ops = {
        'accuracy': tf.compat.v1.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def build_estimator(model_dir):
  return tf.estimator.Estimator(
      model_fn=_cnn_model_fn,
      model_dir=model_dir,
      config=tf.estimator.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
  inputs = {'inputs': tf.compat.v1.placeholder(tf.float32, [None, 784])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_and_evaluate(output_dir,
                       data_dir,
                       train_batch_size=100,
                       eval_batch_size=100,
                       train_steps=10000,
                       eval_steps=100,
                       **experiment_args):
    estimator = build_estimator(output_dir)
    train_spec=tf.estimator.TrainSpec(
            input_fn=get_input_fn(
                filename=os.path.join(data_dir, 'train.tfrecords'),
                batch_size=train_batch_size),
            max_steps=train_steps)
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec=tf.estimator.EvalSpec(
            input_fn=get_input_fn(
                filename=os.path.join(data_dir, 'test.tfrecords'),
                batch_size=eval_batch_size),
            steps=eval_steps,
            exporters=exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec,
                                    **experiment_args)
