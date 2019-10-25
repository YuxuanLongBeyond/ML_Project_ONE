# EPFL Machine Learning Higgs 2019

(**Python 3** is required and **Numpy** must be installed)
(To ensure that your Numpy works, you can go to the directory of this folder and type "pip install -r requirements.txt")

Please **first** put the train data (**train.csv**) and test data (**test.csv**) in the folder 'data', then directly run '**run.py**' to re-produce the submission file.

There are three important variables in 'run.py':

> train_validate --Only if it is True, then training on multiple models is performed. By default, it is set to False.

> final_test -- Only if it is True, then optimal parameters are automatically loaded and the test is performed (submission file will be generated). By default, it is set to True.

> method -- a string that is either 'ls', 'log' or 'dl' (they represent ridge regression, logistic regression and deep learning). By default, it is set to be 'dl'.
&nbsp;
&nbsp;

If you want to re-produce the optimal parameters, please set train_validate to True. However, it will take plenty of time ( at least over 20 minutes) to perform full training when we set method to 'dl'.

&nbsp;
The content of files is explained as below:

> The file 'run.py' organizes all algorithms to do training and testing, also including data pre-processing. The computed parameters are recorded in the folder 'parameters'.

> The file 'implementations.py' involves basic implementation of some ML algorithms, e.g. least squares and logistic regression.

> The file 'fast_simple_net.py' contains a simple neural network model, with forward and backward path.

> The file 'feature_processing.py' performs separation of training data, and stores stores 6 types of training data into the folder 'train_data'.
