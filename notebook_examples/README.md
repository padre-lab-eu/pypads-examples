# pypads-notebooks

This project has sample notebooks on how to run pypads and customize it to track different parameters and results. 
This will be mirrored to GitHub so that users can have a first hand look on how to use the Pads infrastructure to run their own experiments.

## How the notebooks are organized
The main folders here are single_run and multiple_runs. 
Single_run folder pertains to experiments with a simple straight forward execution and no hyperparameter search
Multiple_runs folder has experiments with advanced hyperparameter search and multiple estimators

## Requirements
The basic requirements for the notebooks are given in the requirements.txt.
PyPads needs Python 3.7 to run and you can install it via pip or compile it from the source.
The documentation for PyPads is available [here](https://pypads.readthedocs.io/en/latest/).
To run the Keras and PyTorch, user will have to install that independently.

## Starting Out
A good point to start out is to create a Conda environment or a Python virtual environment.
A primer for the Python virtual environment can be found 
[here](https://realpython.com/python-virtual-environments-a-primer/) and for Anaconda,
[here](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307).

Each experiment to be tracked in PyPads requires some environment variables to be set.
PyPads logs all the experiments to a MLFlow server and a MongoDB server. The necessary environment variables are listed
- AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXXXXXXXXX
- AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
- MLFLOW_S3_ENDPOINT_URL=http://s3.padre-lab.eu
- MLFLOW_TRACKING_URI=http://mlflow.padre-lab.eu
- MONGO_DB = pypads
- MONGO_USER=pypads
- MONGO_URL=mongodb://www.padre-lab.eu:2222
- MONGO_PW=XXXXXXXXXXXXXXXXXXXXXXXXX

### Custom Dataset
Users mostly need datasets other than scikit-learn datasets. In this example, a custom dataset is created, split into 
training and testing and classifies the data using a RandomForestClassifier from scikit-learn.

### PyTorch_NeuralNetworks
A PyTorch example with the Iris dataset. This experiment shows tracking by the PyPads infrastructure for a PyTorch experiment.
It also contains information to create your own Neural Network model, custom Data Loader and one hot encoding.
The custom dataloader is created by specifying a custom collate function that produces the batches.
The notebook as produces live graphs for the training loss, the accuracy of the network on the test set, and the time of execution for each epoch.
The model is tested at regular intervals and the accuracy results are logged to pypads via the different parameters.
The notebook uses the scikit-learn classification report and the accuracy metrics.
The notebook also uses ipy widgets to show the training progress.
The notebook also accesses a single metric at a particular step value and prints it
The notebook retrieves a parameter and converts it back to its original datatype from string.

We use custom tracking of parameters in this notebook, taking the learning rate and the network shape as examples.
Note that the network shape parameter is passed as a string because MLFlow accepts parameters as strings.

### Keras Neural Networks
The Keras example is taken from [here]( Example takes from keras documentation https://keras.io/examples/vision/mnist_convnet/)
Shows how to train a classifier on the MNIST dataset which is tracked by PyPads.
This notebook also uses both keras based metrics and tracked scikit-learn metrics.
We also visualize the progress in training by incorporating callbacks and providing a live graph of the loss and accuracy.

After completion of training, the different tracked artifacts are recalled and the values printed.
