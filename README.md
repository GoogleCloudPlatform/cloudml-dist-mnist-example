# Cloud ML and GCE compatible TensorFlow distributed training example

Disclaimer: This is not an official Google product.

This is an example to demonstrate how to write distributed TensorFlow code
 which can be used on both Cloud Machine Learning and Google Compute Engine
 instances. This is deliberately made simple and straightforward to highlight
 essential aspects. If you are interested in a more sophisticated and
 practical example, see [this one][1].

## Products
- [TensorFlow][2]
- [Cloud Machine Learning][3]
- [Cloud Datalab][9]

[1]: https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/mnist/distributed/trainer
[2]: https://www.tensorflow.org/
[3]: https://cloud.google.com/ml/
[9]: https://cloud.google.com/datalab/

## Prerequisites
1. A Google Cloud Platform Account
2. [A new Google Cloud Platform Project][4] for this lab with billing enabled
3. Enable the Cloud Machine Learning Engine API from [the API Manager][5]

[4]: https://console.developers.google.com/project
[5]: https://console.developers.google.com

## Do this first
In this section you will start your [Google Cloud Shell][6] and clone the
 application code repository to it.

1. [Open the Cloud Console][7]

2. Click the Google Cloud Shell icon in the top-right and wait for your shell
 to open:

 <img src="docs/img/cloud-shell.png" width="300">

3. List the models to verify that the command returns an empty list:

  ```
  $ gcloud ml-engine models list
  Listed 0 items.
  ```

  Note: After you start creating models, you can see them listed by using this command.

4. Clone the lab repository in your cloud shell, then `cd` into that dir and checkout v2.0 branch.

  ```
  $ git clone https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example.git
  Cloning into 'cloudml-dist-mnist-example'...
  ...

  $ cd cloudml-dist-mnist-example
  $ git checkout v2.0
  ```

[6]: https://cloud.google.com/cloud-shell/docs/
[7]: https://console.cloud.google.com/

## Train the model on Cloud Machine Learning

1. Create a bucket used for training jobs.

  ```
  $ PROJECT_ID=$(gcloud config list project --format "value(core.project)")
  $ BUCKET="gs://${PROJECT_ID}-ml"
  $ gsutil mkdir $BUCKET
  ```

2. Upload MNIST dataset to the training bucket.

  ```
  $ ./scripts/create_records.py 
  $ gsutil cp /tmp/data/train.tfrecords gs://$BUCKET/data/
  $ gsutil cp /tmp/data/test.tfrecords gs://$BUCKET/data/
  ```

Note: The dataset is stored in the [TFRecords][10] format.

[10]: https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details

3. Submit a training job to Cloud Machine Learning.

  ```
  $ JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
  $ gcloud ml-engine jobs submit training ${JOB_ID} \
      --package-path trainer \
      --module-name trainer.task \
      --staging-bucket gs://${BUCKET} \
      --job-dir gs://${BUCKET}/${JOBNAME} \
      --runtime-version 1.0 \
      --region us-central1 \
      --config config/config.yaml \
      -- \
      --data_dir gs://${BUCKET}/data \
      --output_dir gs://${BUCKET}/${JOBNAME} \
      --train_steps 10000
  ```

  Note: `JOB_ID` can be arbitrary, but you can't reuse the same one.

  Note: Edit `config/config.yaml` to change the amount of resources
  to be allocated for the job.

  During the training, worker nodes show a training loss value (the total
  loss value for dataset in a single training batch) in some intervals.
  In addition, the master node shows a loss and accuracy for the testset
  about every 3 minutes.

  At the end of the training, the final evaluation against the testset is
  shown as below. In this example, it achieved 99.0% accuracy for the testset.

  ```
  INFO   2017-01-05 16:20:22 +0900   master-replica-0    Step: 10007, Test loss: 313.048615, Test accuracy: 0.990000
  ```

3. (Option) Visualize the training process with TensorBoard

  After the training, the summary data is stored in
  `gs://${BUCKET}/${JOBNAME}` and you can visualize them with TensorBoard.
  First, run the following command on the CloudShell to start TensorBoard.

  ```
  $ tensorboard --port 8080 --logdir gs://${BUCKET}/${JOBNAME}
  ```

  Select 'Preview on port 8080' from Web preview menu in the top-left corner
  to open a new browser window:

  ![](docs/img/web-preview.png)

  In the new window, you can use TensorBoard to see the training summary and
  the visualized network graph, etc.

  <img src="docs/img/tensorboard.png" width="600">

## Deploy the trained model for predictions

1. Deploy the trained model and set the default version.

  ```
  $ MODEL_NAME=MNIST
  $ ORIGIN=$(gsutil ls gs://${BUCKET}/${JOBNAME}/export/Servo | tail -1)
  $ gcloud ml-engine models create ${MODEL_NAME} --regions us-central1
  $ VERSION_NAME=v1
  $ gcloud ml-engine versions create \
      --origin ${ORIGIN} \
      --model ${MODEL_NAME} \
      ${VERSION_NAME}
  $ gcloud ml-engine versions set-default --model ${MODEL_NAME} ${VERSION_NAME}
  ```

  Note: `MODEL_NAME` ane `VERSION_NAME` can be arbitrary, but you can't
  reuse the same one. It may take a few minutes for the deployed model
  to become ready. Until it becomes ready, it returns a 503 error against
  requests.

2. Create a JSON request file.

  ```
  $ ./scripts/make_request.py
  ```

  This creates a JSON file `request.json` containing 10 test data for
  predictions. Each line contains a MNIST image and a sequential key value.

3. Submot an online prediction request.

  ```
  $ gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json

  ```

  `CLASSES` is the most probable digit of the given image, and `PROBABILITIES` shows the probabilites of each digit.

## Using online prediction from Datalab

You can use the Datalab notebook to demonstrate the online prediction feature in an interactive manner.

1. Launch Datalab from the Cloud Shell.

  ```
  $ datalab create mydatalab --zone us-central1-a
  ...
  Click on the *Web Preview* (up-arrow button at top-left), select *port 8081*, and start using Datalab.
  ```
  
2. Select 'Preview on port 8081' from Web preview menu in the top-left corner to open a Datalab window.

3. Open a new notebook and execute the following command.

  ```
  !git clone https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example
  
  Cloning into 'cloudml-dist-mnist-example'...
  remote: Counting objects: 66, done.
  remote: Compressing objects: 100% (15/15), done.
  remote: Total 66 (delta 3), reused 0 (delta 0), pack-reused 51
  Unpacking objects: 100% (66/66), done.
  Checking connectivity... done.
  ```
  
4. Go back to the notebook list window and open `Online prediction example.ipynb` in `cloudml-dist-mnist-example/notebooks` folder.

5. Follow the [instruction](https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example/blob/master/notebooks/Online%20prediction%20example.ipynb) in the notebook.

## (Option) Training on VM instances

Optionally, you can train the model using VM instances running on
 Google Compute Engine(GCE).
1. Launch four VM instances with the following options.

  - Hostname: ps-1, master-0, worker-0, worker-1
  - OS image: ubuntu-1604-lts
  - Machine type: n1-standard-1
  - Identity and API access: Set access for each API, Storage = 'Read Write'

  Note: Since instance roles are inferred from their hostnames,
  you must set hostnames exactly as specified.

2. Install TensorFlow

  Open ssh terminal and run the following commands on all instances.

  ```
  $ sudo apt-get update
  $ sudo apt-get install -y python-pip python-dev
  $ sudo pip install --upgrade tensorflow
  ```

3. Upload MNIST dataset to the training bucket.

  This is the same as the step.2 of "Train the model on Cloud Machine Learning".

4. Start training

  Run the following commands on the CloudShell. It distributes executable
  files to all instances and start a training job.

  ```
  $ gcloud config set compute/zone us-east1-c
  $ git clone https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example.git
  $ cd cloudml-dist-mnist-example
  $ ./scripts/start-training.sh
  ```

  Note: `us-east1-c` should be the zone of instances you have created.

  When the training is finished, it displayes the storage path
  containing the model binary.
  
## Clean up

Clean up is really easy, but also super important: if you don't follow these
 instructions, you will continue to be billed for the project you created.

To clean up, navigate to the [Google Developers Console Project List][8],
 choose the project you created for this lab, and delete it. That's it.

[8]: https://console.developers.google.com/project
