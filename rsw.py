import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


df = pd.read_excel('./dataset/RSW-dataset.xlsx')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print('train_df.shape:', train_df.shape)
print('test_df.shape:', test_df.shape)

input_cols = list(train_df.columns)[1:]
target_col = 'NUGGET.WIDTH.1'


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


print(train_inputs.describe().loc[['min', 'max']])

scaler = MinMaxScaler().fit(df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

print(train_inputs.describe().loc[['min', 'max']])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoder.fit(df[categorical_cols])

encoder.categories_

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)

X_train = train_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

def evalute(targets, predictions,  name=''):
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse =  root_mean_squared_error(targets, predictions)
    
    print(name)
    print("mae:", mae)
    print("mse:", mse)
    print("rmse=", rmse)
    
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
mlp_reg.fit(X_train, train_targets)


train_preds = mlp_reg.predict(X_train)
evalute(train_targets, train_preds, "Training")


test_preds = mlp_reg.predict(X_test)
evalute(test_targets, test_preds, "Testing")

print(mlp_reg.score(X_train, train_targets))
print(mlp_reg.score(X_test, test_targets))