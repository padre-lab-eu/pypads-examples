"""
==============================================
Template on how to use Pypads
==============================================
Configuration of environment variables used by pypads

Create a .config file under HOME_DIRECTORY/.pypads/ and copy paste the following into that file.

[DEFAULT_PYPADS]
AWS_ACCESS_KEY_ID=xxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MLFLOW_S3_ENDPOINT_URL=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MLFLOW_TRACKING_URI=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_DB=xxxxx
MONGO_USER=xxxxxxx
MONGO_URL=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_PW=xxxxxxxxxxx
[CONFIG]
mongo_db = True
"""
# Importing and initializing PyPads should be the first in your code
from pypads.app.base import PyPads

tracker = PyPads(autostart='Experiment Name')  # This would automatically create the experiment and start a run

# Do your imports here
import torch
import sklearn
import keras


# Tracking your Data
# If you are loading a dataset using a custom function we recommend decorating it,
# if you are using a dataset from one of the packages listed above it will be automatically tracked.


@tracker.decorators.dataset(name="Name of your dataset", target_columns=[], output_format=dict(), metadata=None)
def load_data(*args, **kwargs):
    """
    Parameters that can be passed to your decorator
    :param name: name of your dataset
    :param target_columns: indices/names of targets or labels columns in case the returned dataset is a single object.
    :param output_format: A dict describing the outputs of your custom function in case of multiple returned objects.
    :param metadata: a dict holding extra meta information on your dataset (kind of features, source, description, etc..)

    Example:
            def load_data():
                X, y = make_classification(n_samples=150)
                return X,y

            For this function we have to give the output_format as follows:
                output_format = {'X': 'features', 'y': 'targets'} such that entries of the dict have the same order
                as the outputs. Keys of the dict would dictate the names of the stored binaries:
                    - Generated_Dataset_X.pickle, Generated_Dataset_y.pickle

            @tracker.decorators.dataset(name="Generated Dataset", output_format={'X': 'features', 'y': 'targets'})
            def load_data():
                X, y = make_classification(n_samples=150)
                return X,y
    """
    # Do your loading here.

    # Supported formats of outputs are: numpy arrays, dataframes, tuples of such types.
    # In case your data is defined on more than one object then try to describe the output
    X, y = "X", "y"
    return X, y


# Do your preprocessing here


# Training your model

# When defining your model/experiment's hyperparameters, try to log them manually. As an example:
number_of_epochs = 100
batch_size = 50

# Log them as following
# tracker.api.log_param(key, value, value_format=None, description="", additional_data: dict = None)
tracker.api.log_param("number_of_epochs", 100, description="Number of training epochs")
tracker.api.log_param("batch_size", 50, description="Size of training/testing batch")


# For Torch model defined using a custom class
# In this case, you can use a decorator defined in pypads.

@tracker.decorators.watch(track="hyper-parameters", debugging=False)
class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # define your layers as attributes.
        self.InputLayer = ...

    def forward(self, x):
        # your forward function
        return


# Evaluating of your model
# When evaluating your model whether iteratively when training, or as a single time when testing. Try to log your metrics values manually.

# An example would be like the following:
from sklearn.metrics.classification import f1_score
score = f1_score(predicted, truth, average="macro")
tracker.api.log_metric('f1_score', score, description="F1 score evaluation of the model", step=0)

# For an iterative logging:
for epoch in range(number_of_epochs):
    # Train or test
    # Log your metric, loss, etc..
    tracker.api.log_metric('training_loss', loss, description="Training Loss of the model", step=epoch)


# End your run
tracker.api.end_run()
