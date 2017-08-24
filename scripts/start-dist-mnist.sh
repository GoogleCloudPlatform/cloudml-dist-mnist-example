#!/bin/bash

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

ver=$(sudo pip list | grep -E '^tensorflow \(' | grep -oE '[0-9]+\.[0-9]+')
if [[ $ver == '1.3' ]]; then
  echo "TF1.3 currently has a compatibility issue with the prediction API."
  echo "Downgrading to TF1.2.1 as a tentative workaround..."
  sudo pip install tensorflow==1.2.1
fi

cd $(dirname $0)
DATADIR=$1
OUTDIR=$2

ROLE=$(hostname | awk -F'-' '{ print $1 }')
INDEX=$(hostname | awk -F'-' '{ print $2 }')

export TF_CONFIG=$(sed "s/__INDEX__/$INDEX/;s/__ROLE__/$ROLE/" tf_config.json)

python trainer/task.py --output_dir=$OUTDIR --data_dir=$DATADIR --train_steps=10000
