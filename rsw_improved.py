import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


df = pd.read_excel('./dataset/RSW-dataset.xlsx')


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

input_cols = list(train_df.columns)[1:]
target_col = 'NUGGET.WIDTH.1'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

scaler = MinMaxScaler().fit(df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


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
    
# Dictionary to store testing results for comparison
results = {}
    
# Linear regression
print("\n--- Linear Regression ---")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_model = LinearRegression(fit_intercept=False, positive=False)
lr_model.fit(X_train_poly, train_targets)

train_preds_lr = lr_model.predict(X_train_poly)
evalute(train_targets, train_preds_lr, "Training")

test_preds_lr = lr_model.predict(X_test_poly)
evalute(test_targets, test_preds_lr, "Testing")

results['Linear Regression'] = {
    'MSE': mean_squared_error(test_targets, test_preds_lr),
    'RMSE': root_mean_squared_error(test_targets, test_preds_lr),
    'MAE': mean_absolute_error(test_targets, test_preds_lr)
}
    
  
# random forest   
print("\n--- Random Forest ---")
rf_model = RandomForestRegressor(
    max_depth=5,  
    max_features='sqrt',  
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
)
rf_model.fit(X_train, train_targets)


train_preds_rf = rf_model.predict(X_train)
evalute(train_targets, train_preds_rf, "Training")

test_preds_rf = rf_model.predict(X_test)
evalute(test_targets, test_preds_rf, "Testing")

results['Random Forest'] = {
    'MSE': mean_squared_error(test_targets, test_preds_rf),
    'RMSE': root_mean_squared_error(test_targets, test_preds_rf),
    'MAE': mean_absolute_error(test_targets, test_preds_rf)
}


# k-nearest
print("\n--- K-Nearest Neighbors ---")
knn_model = KNeighborsRegressor(n_neighbors=15, p=2, weights='uniform')
knn_model.fit(X_train, train_targets)

train_preds_knn = knn_model.predict(X_train)
evalute(train_targets, train_preds_knn, "Training")

test_preds_knn = knn_model.predict(X_test)
evalute(test_targets, test_preds_knn, "Testing")

results['K-Nearest Neighbor'] = {
    'MSE': mean_squared_error(test_targets, test_preds_knn),
    'RMSE': root_mean_squared_error(test_targets, test_preds_knn),
    'MAE': mean_absolute_error(test_targets, test_preds_knn)
}

# MLP
print("\n--- MLP Regressor ---")
mlp_reg = MLPRegressor( activation='relu', alpha=0.0001, hidden_layer_sizes=[100, 50], random_state=42, max_iter=1000, solver='adam') 
mlp_reg.fit(X_train, train_targets)


train_preds_mlp = mlp_reg.predict(X_train)
evalute(train_targets, train_preds_mlp, "Training")

test_preds_mlp = mlp_reg.predict(X_test)
evalute(test_targets, test_preds_mlp, "Testing")

results['MLP'] = {
    'MSE': mean_squared_error(test_targets, test_preds_mlp),
    'RMSE': root_mean_squared_error(test_targets, test_preds_mlp),
    'MAE': mean_absolute_error(test_targets, test_preds_mlp)
}


# ===============================================
# FINAL COMPARISON AND VISUALIZATION
# ===============================================

# Convert results dictionary to DataFrame for easy comparison
results_df = pd.DataFrame(results).T.sort_values(by='MSE')


# Create Bar Chart for MSE (the main metric used in the presentation)
plt.figure(figsize=(9, 6))
sns.barplot(x=results_df.index, y='MSE', data=results_df, palette='viridis')

plt.title('Comparison of Model Performance (Testing MSE)', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.ylim(1.4, results_df['MSE'].max() + 0.1) # Set y-limit for better visualization

# Add values on top of bars
for index, row in results_df.iterrows():
    plt.text(row.name, row.MSE + 0.01, f"{row.MSE:.3f}", color='black', ha="center", fontsize=10)

plt.show()