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

# Get the number of nodes
NUM_PS=$(gcloud compute instances list -r '^ps-\d+ ' | wc -l)
NUM_WORKER=$(gcloud compute instances list -r '^worker-\d+ ' | wc -l)

NUM_PS=$(( NUM_PS - 2 ))
NUM_WORKER=$(( NUM_WORKER - 2 ))

# Stop parameter servers
for  i in $(seq 0 $NUM_PS); do
  echo "Terminating ps-${i}..."
  gcloud compute ssh ps-${i} -- pkill -f trainer/task.py
done

# Stop workers
for  i in $(seq 0 $NUM_WORKER); do
  echo "Terminating worker-${i}..."
  gcloud compute ssh worker-${i} -- pkill -f trainer/task.py
done

# Stop a master
echo "Terminating master-0..."
gcloud compute ssh master-0 -- pkill -f trainer/task.py
