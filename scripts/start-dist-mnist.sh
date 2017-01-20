#!/bin/bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

cd $(dirname $0)

ROLE=$(hostname | awk -F'-' '{ print $1 }')
INDEX=$(hostname | awk -F'-' '{ print $2 }')

export TF_CONFIG=$(sed "s/__INDEX__/$INDEX/;s/__ROLE__/$ROLE/" tf_config.json)

LOG_DIR="/tmp/logs"
MODEL_DIR="/tmp/model"
DATA_DIR="$HOME/data-pd"
WORK_FLAGS="--batch_size=100 --max_steps=10000 --local_data"

rm -rf $LOG_DIR
mkdir -p $LOG_DIR $MODEL_DIR
python trainer/task.py $WORK_FLAGS \
       --data_dir=$DATA_DIR --log_dir=$LOG_DIR --model_dir=$MODEL_DIR
