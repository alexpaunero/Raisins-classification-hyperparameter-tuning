import codecademylib3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Categorical, Real
from tpot import TPOTClassifier

# Load the data set
raisins = pd.read_csv('Raisin_Dataset.csv')
X = raisins.drop('Class', axis=1)
y = raisins['Class']

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create an SVC model
svm = SVC()

# Dictionary of parameters for GridSearchCV
parameters = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [1, 10, 100]}

# Create a GridSearchCV model


# Fit the GridSearchCV model to the training data


# Print the model and hyperparameters obtained by GridSearchCV


# Print a table summarizing the results of GridSearchCV

# Print the accuracy of the final model on the test data


# Dictionary of parameters for BayesSearchCV


# Create a BayesSearchCV model


# Fit the BayesSearchCV model to the training data


# Print the model and hyperparameters obtained by BayesSearchCV


# Print the accuracy of the final model on the test data


# Create a TPOTClassifier model


# Fit the TPOTClassifier model to the training data


# Print the accuracy of the final model on the test data


# Export TPOTClassifier's final model to a separate file


