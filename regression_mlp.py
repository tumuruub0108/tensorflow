# MLP = The Multilayer Perceptron

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


"""
Hyperparameter                  Typical value
# hidden layers                 Depends on the problem, but typically 1 to 5
# neurons per hidden layer      Depends on the problem, but typically 10 to 100
# output neurons                1 per prediction dimension
Hidden activation               ReLU
Output activation               None, or ReLU/softplus (if positive outputs) or sigmoid/tanh (if bounded outputs)
Loss function                   MSE, or Huber if outliers

"""


housing = fetch_california_housing()
 
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_valid)
rmse = root_mean_squared_error(y_valid, y_pred) # about 0.505
print(rmse)