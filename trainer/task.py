# Copyright 2017 Google Inc. All Rights Reserved.
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


import argparse
import os

import model

from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)


def generate_experiment_fn(data_dir,
                           train_batch_size=100,
                           eval_batch_size=100,
                           train_steps=10000,
                           eval_steps=100,
                           **experiment_args):

  def _experiment_fn(output_dir):
    return Experiment(
        model.build_estimator(output_dir),
        train_input_fn=model.get_input_fn(
            filename=os.path.join(data_dir, 'train.tfrecords'),
            batch_size=train_batch_size),
        eval_input_fn=model.get_input_fn(
            filename=os.path.join(data_dir, 'test.tfrecords'),
            batch_size=eval_batch_size),
        export_strategies=[saved_model_export_utils.make_export_strategy(
            model.serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1)],
        train_steps=train_steps,
        eval_steps=eval_steps,
        **experiment_args
    )
  return _experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      help='GCS or local path to training data',
      required=True
  )
  parser.add_argument(
      '--train_batch_size',
      help='Batch size for training steps',
      type=int,
      default=100
  )
  parser.add_argument(
      '--eval_batch_size',
      help='Batch size for evaluation steps',
      type=int,
      default=100
  )
  parser.add_argument(
      '--train_steps',
      help='Steps to run the training job for.',
      type=int,
      default=10000
  )
  parser.add_argument(
      '--eval_steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='junk'
  )
  parser.add_argument(
      '--eval_delay_secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min_eval_frequency',
      help='Minimum number of training steps between evaluations',
      default=1,
      type=int
  )

  args = parser.parse_args()
  arguments = args.__dict__

  # unused args provided by service
  arguments.pop('job_dir', None)
  arguments.pop('job-dir', None)

  output_dir = arguments.pop('output_dir')

  # Run the training job
  learn_runner.run(generate_experiment_fn(**arguments), output_dir)
