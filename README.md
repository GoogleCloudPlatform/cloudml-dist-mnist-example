# Cloud ML and GCE compatible TensorFlow distributed training example

Disclaimer: This is not an official Google product.

This is an example to demonstrate how to write distributed TensorFlow code
 which can be used on both Cloud Machine Leraning and Google Compute Engine
 instances. This is deliberately made simple and straightforward to highlight
 essential aspects. If you are interested in a more sophisticated and
 practical example, see [this one][1].

## Products
- [TensorFlow][2]
- [Cloud Machine Learning][3]

[1]: https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/mnist/distributed/trainer
[2]: https://www.tensorflow.org/
[3]: https://cloud.google.com/ml/

## Prerequisites
1. A Google Cloud Platform Account
2. [A new Google Cloud Platform Project][4] for this lab with billing enabled
3. Enable the Cloud Machine Learning API from [the API Manager][5]

[4]: https://console.developers.google.com/project
[5]: https://console.developers.google.com

## Do this first
In this section you will start your [Google Cloud Shell][6] and clone the
 application code repository to it.

1. [Open the Cloud Console][7]

2. Click the Google Cloud Shell icon in the top-right and wait for your shell
 to open:

 <img src="docs/img/cloud-shell.png" width="300">

3. Install Cloud Machine Learning SDK and initialize the project:

  ```
  $ sudo pip install --upgrade pillow
  $ curl https://storage.googleapis.com/cloud-ml/scripts/setup_cloud_shell.sh | bash
  $ export PATH=${HOME}/.local/bin:${PATH}
  $ gcloud beta ml init-project
  Cloud ML needs to add its service accounts to your project
  (Your project ID) as Editors. This will enable Cloud Machine
   Learning to access resources in your project when running your
  training and prediction jobs.

  Do you want to continue (Y/n)? Y
  ...
  ```

  Note: The first pip command is a workaround to avoid a version compatibility
  issue of the Pillow module.

4. Clone the lab repository in your cloud shell, then `cd` into that dir:

  ```
  $ git clone https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example.git
  Cloning into 'cloudml-dist-mnist-example'...
  ...

  $ cd cloudml-dist-mnist-example
  ```

[6]: https://cloud.google.com/cloud-shell/docs/
[7]: https://console.cloud.google.com/

## Train the model on Cloud Machine Learning

1. Create a bucket used for training jobs.

  ```
  $ PROJECT_ID=$(gcloud config list project --format "value(core.project)")
  $ TRAIN_BUCKET="gs://${PROJECT_ID}-ml"
  $ gsutil mkdir $TRAIN_BUCKET
  ```

2. Submit a training job to Cloud Machine Learning.

  ```
  $ JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
  $ gcloud beta ml jobs submit training ${JOB_ID} \
      --package-path trainer \
      --module-name trainer.task \
      --staging-bucket "${TRAIN_BUCKET}" \
      --region us-central1 \
      --config config/config.yaml \
      -- \
      --log_dir ${TRAIN_BUCKET}/${JOB_ID}/train \
      --model_dir ${TRAIN_BUCKET}/${JOB_ID}/model \
      --max_steps 10000
  ```

  Note: `JOB_ID` can be arbitrary, but you can't reuse the same one.

  Note: Edit `config/config.yaml` to change the amount of resources
  to be allocated for the job.

  During the training, each worker node shows a training loss value (the total
  loss value for dataset in a single training batch) in some intervals.
  In addition, the master node shows a loss and accuracy for the testset.

  At the end of the training, the final evaluation against the testset is
  shown as below. In this example, it achieved 99.0% accuracy for the testset.

  ```
  INFO   2017-01-05 16:20:22 +0900   master-replica-0    Step: 10007, Test loss: 313.048615, Test accuracy: 0.990000
  ```

3. (Option) Visualize the training process with TensorBoard

  After the training, the summary data is stored in
  `${TRAIN_BUCKET}/${JOB_ID}/train` and you can visualize them with TensorBoard.
  First, run the following command on the CloudShell to start TensorBoard.

  ```
  $ tensorboard --port 8080 --logdir ${TRAIN_BUCKET}/${JOB_ID}/train
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
  $ gcloud beta ml models create ${MODEL_NAME}
  $ VERSION_NAME=v1
  $ gcloud beta ml versions create \
      --origin $TRAIN_BUCKET/${JOB_ID}/model \
      --model ${MODEL_NAME} \
      ${VERSION_NAME}
  $ gcloud beta ml versions set-default --model ${MODEL_NAME} ${VERSION_NAME}
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
  $ gcloud beta ml predict --model ${MODEL_NAME} --json-instances request.json
KEY  SCORES
0    [5.491162212434286e-12, 1.4754857058374427e-10, 5.0145921193234244e-08, 9.679915820015594e-07, 1.1807515902517718e-11, 2.1074924791419924e-11, 7.73946938943083e-19, 0.9999980926513672, 6.3151288642870895e-09, 9.910554581438191e-07]
1    [2.9550360380881102e-08, 2.0272503320484248e-07, 0.9999997615814209, 5.557484472618057e-10, 7.705624062003674e-14, 1.2500921352492488e-14, 2.9164962112027126e-11, 3.5212078648109036e-15, 9.194990879812792e-10, 9.704423389468017e-17]
2    [6.980260369715552e-09, 0.9999717473983765, 1.7510066641079902e-07, 1.3880581128944414e-08, 1.2206406609038822e-05, 1.52971484368436e-08, 7.902426091277448e-07, 9.003126251627691e-06, 5.9756007431133185e-06, 1.860657405927668e-08]
3    [0.999998927116394, 1.8409378839054358e-13, 3.04105398640786e-08, 4.30901364936731e-12, 3.6688863058742527e-10, 2.4977113710633603e-09, 9.903883437800687e-07, 1.4073207876830196e-10, 9.441464277060163e-10, 2.3699602280657928e-09]
4    [8.716865007585284e-10, 5.250225809660947e-10, 5.023502724910145e-10, 2.899413998821987e-12, 0.9999929666519165, 5.790986440379342e-11, 1.9194641431852233e-09, 1.5732334546214588e-08, 8.654805205843275e-10, 7.070047558954684e-06]
5    [8.770115189626893e-10, 0.9999911785125732, 5.714275275181535e-09, 2.464804471635773e-10, 1.5737153944428428e-06, 4.692493615898741e-11, 7.134818069687299e-09, 7.002449365245411e-06, 2.3442017038632912e-07, 5.295051952458607e-09]
6    [5.615819969966539e-15, 4.6051244595446406e-08, 1.492788878620921e-11, 1.1583707951179356e-11, 0.9999597072601318, 1.657236126106909e-08, 3.130453882227435e-11, 1.6037419925396534e-07, 3.9096550608519465e-05, 9.630925887904596e-07]
7    [3.4089366746092864e-11, 1.3498856255012015e-08, 1.4886258448143508e-08, 4.7652913053752854e-05, 0.0007680997950956225, 2.1690282210329315e-06, 1.2528266184891335e-12, 2.3564155071653659e-07, 1.509065623395145e-05, 0.99916672706604]
8    [1.2211978983600602e-08, 6.400643641200931e-16, 3.752947819180008e-08, 6.670989871615518e-10, 1.6800930646709844e-09, 0.9996830224990845, 0.00021479142014868557, 6.799472790364192e-11, 9.631117427488789e-05, 5.793618583993521e-06]
9    [9.991658889152433e-11, 1.0411928480849597e-12, 8.161708114906574e-11, 8.24744432748048e-08, 2.505870179447811e-05, 7.725438067041068e-10, 8.123087299376566e-14, 1.051930939865997e-05, 7.270523201441392e-05, 0.9998916387557983]
  ```

  The prediction results (scores for labels) are shown with the associated key values.

## (Option) Training on VM instances

Optionally, you can train the model using VM instances running on
 Google Compute Engine(GCE).

1. Launch four VM instances with the following options

  - Hostname: ps-1, master-0, worker-0, worker-1
  - OS image: ubuntu-1604-lts
  - Machine type: n1-standard-1

  Note: Since instance roles are inferred from their hostnames,
  you must set hostnames exactly as specified.

2. Install TensorFlow

  Open ssh terminal and run the following commands on all instances.

  ```
  $ TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
  $ sudo apt-get update
  $ sudo apt-get install -y python-pip python-dev
  $ sudo pip install --upgrade $TF_BINARY_URL
  ```

3. Download training data

  Instead of fetching dynamically from the web, you place the training
  data in the local directory in advance.

  Run the following commands on master-0, worker-0 and worker-1.

  ```
  $ mkdir $HOME/data-pd
  $ cd $HOME/data-pd
  $ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
  $ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
  $ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
  $ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  ```

4. Start training

  Run the following commands on the CloudShell. It distributes executable
  files to all instances and start a training job.

  ```
  $ gcloud config set compute/zone us-east1-c
  $ git clone https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example.git
  $ cd cloudml-dist-mnist-example
  $ ./scrpits/start-dist-mnist.sh
  ```

  Note: `us-east1-c` should be the zone of instances you have created.

## Clean up

Clean up is really easy, but also super important: if you don't follow these
 instructions, you will continue to be billed for the project you created.

To clean up, navigate to the [Google Developers Console Project List][8],
 choose the project you created for this lab, and delete it. That's it.

[8]: https://console.developers.google.com/project
