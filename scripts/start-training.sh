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

MAXWORKERS=5
WORKDIR=/tmp/workdir

if [[ $# -lt 1 ]]; then
  PROJECT_ID=$(gcloud config list project --format "value(core.project)")
  BUCKET="gs://${PROJECT_ID}-ml"
else
  BUCKET=$1
fi

JOBNAME=job_$(date -u +%y%m%d_%H%M%S)
DATADIR=${BUCKET}/data
OUTDIR=${BUCKET}/${JOBNAME}

pushd $(dirname $0) >/dev/null

# Force stop existing jobs
echo "Force stop existing jobs."
./stop-training.sh

# Get the number of nodes
NUM_PS=$(gcloud compute instances list | grep -E '^ps-[0-9]+ ' | wc -l)
NUM_WORKER=$(gcloud compute instances list | grep -E '^worker-[0-9]+ ' | wc -l)

NUM_PS=$(( NUM_PS - 1 ))
NUM_WORKER=$(( NUM_WORKER - 1 ))

if [[ $NUM_WORKER -gt $(( MAXWORKERS - 1 )) ]]; then
  NUM_WORKER=$(( MAXWORKERS - 1 ))
fi

# Create TF_CONFIG file
ps_entry="\"ps\": ["
for i in $(seq 0 $NUM_PS); do
  if [[ ! $i -eq $NUM_PS ]]; then
    ps_entry="${ps_entry}\"ps-${i}:2222\", "
  else
    ps_entry="${ps_entry}\"ps-${i}:2222\"],"
  fi
done

worker_entry="\"worker\": ["
for i in $(seq 0 $NUM_WORKER); do
  if [[ ! $i -eq $NUM_WORKER ]]; then
    worker_entry="${worker_entry}\"worker-${i}:2222\", "
  else
    worker_entry="${worker_entry}\"worker-${i}:2222\"],"
  fi
done

cat <<EOF > /tmp/tf_config.json
{
  "environment": "cloud",
  "cluster": {
    ${ps_entry}
    ${worker_entry}
    "master": ["master-0:2222"]
  },
  "task": {
    "index": __INDEX__,
    "type": "__ROLE__"
  }
}
EOF

echo "Start a training job."

# Start parameter servers in the background
for  i in $(seq 0 $NUM_PS); do
  echo "Starting ps-${i}..."
  gcloud compute ssh ps-${i} -- rm -rf $WORKDIR
  gcloud compute ssh ps-${i} -- mkdir -p $WORKDIR
  gcloud beta compute scp --recurse \
    /tmp/tf_config.json start-dist-mnist.sh ../trainer/ \
    ps-${i}:$WORKDIR
  gcloud compute ssh ps-${i} -- $WORKDIR/start-dist-mnist.sh $DATADIR $OUTDIR &
done

# Start workers in the background
for  i in $(seq 0 $NUM_WORKER); do
  echo "Starting worker-${i}..."
  gcloud compute ssh worker-${i} -- rm -rf $WORKDIR
  gcloud compute ssh worker-${i} -- mkdir -p $WORKDIR
  gcloud beta compute scp --recurse \
    /tmp/tf_config.json start-dist-mnist.sh ../trainer/ \
    worker-${i}:$WORKDIR
  gcloud compute ssh worker-${i} -- $WORKDIR/start-dist-mnist.sh $DATADIR $OUTDIR &
done

# Start a master
echo "Starting master-0..."
gcloud compute ssh master-0 -- rm -rf $WORKDIR
gcloud compute ssh master-0 -- mkdir -p $WORKDIR
gcloud beta compute scp --recurse \
  /tmp/tf_config.json start-dist-mnist.sh ../trainer/ \
  master-0:$WORKDIR
gcloud compute ssh master-0 -- $WORKDIR/start-dist-mnist.sh $DATADIR $OUTDIR

# Cleanup
echo "Done. Force stop remaining processes."
./stop-training.sh

ORIGIN=$(gsutil ls $BUCKET/$JOBNAME/export/Servo | tail -1)
echo ""
echo "Trained model is stored in $ORIGIN"

popd >/dev/null
