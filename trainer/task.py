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
import logging
import os
import time

import numpy as np
import tensorflow as tf

import mnist_model
import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100,
                     'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('checkpoint', 100, 'Interval steps to save checkpoint.')
flags.DEFINE_string('log_dir', '/tmp/logs',
                    'Directory to store checkpoints and summary logs')
flags.DEFINE_string('model_dir', '/tmp/model',
                    'Directory to store trained model')
flags.DEFINE_string('data_dir', '/tmp/data',
                    'Directory to store training data')
flags.DEFINE_boolean('local_data', False,
                     'If ture, don\'t fetch training data from the web')


# Global flags
BATCH_SIZE = FLAGS.batch_size
MODEL_DIR = FLAGS.model_dir
LOG_DIR = FLAGS.log_dir
DATA_DIR = FLAGS.data_dir
LOCAL_DATA = FLAGS.local_data
MAX_STEPS = FLAGS.max_steps
CHECKPOINT = FLAGS.checkpoint


def export_model(last_checkpoint):
  # Create a session with a new graph.
  with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.float32, [None, 784])
    p = mnist_model.get_model(x, training=False)

    # Define key elements
    input_key = tf.placeholder(tf.int64, [None,])
    output_key = tf.identity(input_key)

    # Define API inputs/outputs object
    inputs = {'key': input_key.name, 'image': x.name}
    outputs = {'key': output_key.name, 'scores': p.name}
    tf.add_to_collection('inputs', json.dumps(inputs))
    tf.add_to_collection('outputs', json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Restore the latest checkpoint and save the model
    saver = tf.train.Saver()
    saver.restore(sess, last_checkpoint)
    saver.export_meta_graph(os.path.join(MODEL_DIR, 'export.meta'))
    saver.save(sess, os.path.join(MODEL_DIR, 'export'),
               write_meta_graph=False)


def run_training():
  # Get cluster and node information
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_info = env.get('cluster')
  cluster_spec = tf.train.ClusterSpec(cluster_info)
  task_info = env.get('task')
  job_name, task_index = task_info['type'], task_info['index']

  device_fn = tf.train.replica_device_setter(
    cluster=cluster_spec,
    worker_device='/job:%s/task:%d' % (job_name, task_index))

  logging.info('Start job:%s, index:%d', job_name, task_index)

  # Create a server object
  server = tf.train.Server(cluster_spec,
                           job_name=job_name, task_index=task_index)

  # Start a parameter server node
  if job_name == 'ps':
    server.join()

  # Start a master/worker node
  if job_name == 'master' or job_name == 'worker': # Worker node
    is_chief = (job_name == 'master')

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):

        # Prepare training data
        mnist_data = mnist.read_data_sets(DATA_DIR, one_hot=True,
                                          local_only=LOCAL_DATA)

        # Create placeholders
        x = tf.placeholder(tf.float32, [None, 784])
        t = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32, [])
        global_step = tf.Variable(0, trainable=False)

        # Add test loss and test accuracy to summary
        test_loss = tf.placeholder(tf.float32, [])
        test_accuracy = tf.placeholder(tf.float32, [])
        tf.summary.scalar('Test_loss', test_loss)
        tf.summary.scalar('Test_accuracy', test_accuracy)

        # Define a model
        p = mnist_model.get_model(x, keep_prob, training=True)
        train_step, loss, accuracy = mnist_model.get_trainer(p, t, global_step)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()

        # Create a supervisor
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=LOG_DIR,
                                 init_op=init_op, saver=saver, summary_op=None,
                                 global_step=global_step, save_model_secs=0)

        # Create a session and start a training loop
        with sv.managed_session(server.target) as sess:
          reports, step = 0, 0
          start_time = time.time()
          while not sv.should_stop() and step < MAX_STEPS:
             images, labels = mnist_data.train.next_batch(BATCH_SIZE)
             feed_dict = {x:images, t:labels, keep_prob:0.5}
             _, loss_val, step = sess.run([train_step, loss, global_step],
                                          feed_dict=feed_dict)
             if step > CHECKPOINT * reports:
               reports += 1
               logging.info('Step: %d, Train loss: %f', step, loss_val)
               if is_chief:
                 # Save checkpoint
                 sv.saver.save(sess, sv.save_path, global_step=step)

                 # Evaluate the test loss and test accuracy
                 loss_vals, acc_vals = [], []
                 for _ in range(len(mnist_data.test.labels) // BATCH_SIZE):
                   images, labels = mnist_data.test.next_batch(BATCH_SIZE)
                   feed_dict = {x:images, t:labels, keep_prob:1.0}
                   loss_val, acc_val = sess.run([loss, accuracy],
                                                feed_dict=feed_dict)
                   loss_vals.append(loss_val)
                   acc_vals.append(acc_val)
                 loss_val, acc_val = np.sum(loss_vals), np.mean(acc_vals)

                 # Save summary
                 feed_dict = {test_loss:loss_val, test_accuracy:acc_val}
                 sv.summary_computed(sess,
                   sess.run(summary, feed_dict=feed_dict), step)
                 sv.summary_writer.flush()

                 logging.info('Time elapsed: %d', (time.time() - start_time))
                 logging.info('Step: %d, Test loss: %f, Test accuracy: %f',
                              step, loss_val, acc_val)

          # Export the final model
          if is_chief:
            sv.saver.save(sess, sv.save_path, global_step=sess.run(global_step))
            export_model(tf.train.latest_checkpoint(LOG_DIR))

        sv.stop()


def main(_):
  run_training()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
