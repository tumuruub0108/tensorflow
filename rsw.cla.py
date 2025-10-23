
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_excel('./dataset/RSW-dataset.xlsx')

# Step 1: compute thinner sheet thickness
df['t_min'] = df[['THICKNESS.1', 'THICKNESS.2']].min(axis=1)

# Step 2: compute standard-based minimum acceptable nugget width
df['D_min'] = 5 * np.sqrt(df['t_min'])

# Define maximum acceptable (e.g., 1.5× rule)
df['D_max'] = 1.5 * df['D_min']

# Classify
df['quality'] = np.where(
    (df['NUGGET.WIDTH.1'] >= df['D_min']) & (df['NUGGET.WIDTH.1'] <= df['D_max']),
    1,  # Acceptable
    0   # Unacceptable
)



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print('train_df.shape:', train_df.shape)
print('test_df.shape:', test_df.shape)

input_cols = list(train_df.columns)[:-1]
target_col = 'quality'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])

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
    print(name)
    accuracy = accuracy_score(targets, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    kappa = cohen_kappa_score(targets, predictions)
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    
    roc = roc_auc_score(targets, predictions)
    print(f"roc_auc_score: {roc:.4f}")
    
    print("confusion matrix")
    sns.heatmap(confusion_matrix(targets, predictions), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    
mlp = MLPClassifier(
    activation='tanh',  
    alpha=0.01,
    hidden_layer_sizes=(50,),                   
    max_iter=1000,     
    solver='adam',         
    random_state=42
)

mlp.fit(X_train, train_targets)


train_preds = mlp.predict(X_train)
evalute(train_targets, train_preds, "Training")


test_preds = mlp.predict(X_test)
evalute(test_targets, test_preds, "Testing")

print(mlp.score(X_train, train_targets))
print(mlp.score(X_test, test_targets))



"""
# Define the Hyperparameter Grid
# NOTE: Remove the 'mlp__' prefix because we are passing the MLPRegressor directly
param_grid = {
    'hidden_layer_sizes': [(50,), (100, 50), (50, 50, 50)], # Architectures to test
    'activation': ['relu', 'tanh'],                         # Activation functions
    'alpha': [0.0001, 0.001, 0.01],                         # Regularization strength
    'solver': ['adam'],                                     # The solver to use
    'max_iter': [1000]                                      # Important for convergence
}


# Instantiate GridSearchCV
# Use mlp_reg_base as the estimator (instead of the non-existent 'pipeline')
grid_search = GridSearchCV(
    estimator=mlp_reg_base, 
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,                 # Use 3-fold cross-validation
    verbose=2, 
    n_jobs=-1             # Use all available CPU cores
)

# Fit the grid search to your pre-processed training data
grid_search.fit(X_train, train_targets)

# --- Analyze Results ---
# The best R² is printed as the score when using neg_mean_squared_error scoring
# You need to convert best_score_ to the square root of the positive version for RMSE

print("Best parameters:", grid_search.best_params_)

# Since the scoring was 'neg_mean_squared_error', best_score_ is a negative MSE
best_mse = -grid_search.best_score_
best_rmse = np.sqrt(best_mse)
print(f"Best RMSE (cross-val): {best_rmse:.4f}")

# You can also check the score using the default scorer (R^2) on the test set
print("Test R² (using best model):", grid_search.best_estimator_.score(X_test, test_targets))
"""