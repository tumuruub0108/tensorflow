
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel('./dataset/RSW-dataset.xlsx')

# Step 1: compute thinner sheet thickness
df['t_min'] = df[['THICKNESS.1', 'THICKNESS.2']].min(axis=1)

# Step 2: compute standard-based minimum acceptable nugget width
df['D_min'] = 5 * np.sqrt(df['t_min'])

df['D_max'] = 1.5 * df['D_min']

df['quality'] = np.where(
    (df['NUGGET.WIDTH.1'] >= df['D_min']) & (df['NUGGET.WIDTH.1'] <= df['D_max']),
    1,   # acceptable
    0    # unacceptable
)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(df['t_min'], df['NUGGET.WIDTH.1'], c=df['quality'], cmap='coolwarm', label='Welds')
plt.plot(df['t_min'], df['D_min'], 'g--', label='D_min (standard limit)')
plt.plot(df['t_min'], df['D_max'], 'r--', label='D_max (upper limit)')
plt.xlabel("Sheet Thickness (mm)")
plt.ylabel("Nugget Width (mm)")
plt.title("Nugget Width vs Standard Acceptable Range")
plt.legend()
plt.show()


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

input_cols = list(train_df.columns)[1:-1]
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


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


def evalute_and_get_metrics(targets, predictions,  name=''):
    print(name)
    accuracy = accuracy_score(targets, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    kappa = cohen_kappa_score(targets, predictions)
    print(f"Cohen's Kappa Score: {kappa:.2f}")
    
    roc = roc_auc_score(targets, predictions)
    print(f"roc_auc_score: {roc:.2f}")
    
    return {'accuracy': accuracy, 'kappa': kappa, 'AUC': roc}

# Dictionary to store testing results for comparison
results = {} 
    
# logistic regression
print("\n--- logistic regression ---")
log_model = LogisticRegression(solver='liblinear')
log_model.fit(X_train, train_targets)

train_preds_lr = log_model.predict(X_train)
train_metrics_lr =  evalute_and_get_metrics(train_targets, train_preds_lr, "Training")

test_preds_lr = log_model.predict(X_test)
test_metrics_lr = evalute_and_get_metrics(test_targets, test_preds_lr, "Testing")

results['Logistic Regression'] = test_metrics_lr


# naive_bayes
print("\n--- naive_bayes ---")
nb_model = GaussianNB()
nb_model.fit(X_train, train_targets)


train_preds_nv = nb_model.predict(X_train)
train_metrics_nv =  evalute_and_get_metrics(train_targets, train_preds_nv, "Training")


test_preds_nv = nb_model.predict(X_test)
test_metrics_nv = evalute_and_get_metrics(test_targets, test_preds_nv, "Testing")

results['Naïve Bayes'] = test_metrics_nv

# Random Forest
print("\n--- Random Forest ---")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, train_targets)

train_preds_rf = rf_model.predict(X_train)
train_metrics_rf = evalute_and_get_metrics(train_targets, train_preds_rf, "Training")

test_preds_rf = rf_model.predict(X_test)
test_metrics_rf = evalute_and_get_metrics(test_targets, test_preds_rf, "Testing")
 
results['Random Forest'] = test_metrics_rf   
    
# MLP
print("\n--- MLP Classifier ---")
mlp = MLPClassifier(
    activation='tanh',  
    alpha=0.01,
    hidden_layer_sizes=(50,),                   
    max_iter=1000,     
    solver='adam',         
    random_state=42
)

mlp.fit(X_train, train_targets)


train_preds_mlp = mlp.predict(X_train)
train_metrics_mlp = evalute_and_get_metrics(train_targets, train_preds_mlp, "Training")

test_preds_mlp = mlp.predict(X_test)
t_metrics_mlp = evalute_and_get_metrics(test_targets, test_preds_mlp, "Testing")

results['MLP'] = t_metrics_mlp




# ===============================================
# FINAL COMPARISON AND VISUALIZATION
# ===============================================

# Convert results dictionary to DataFrame for easy comparison
# Sort by AUC (highest is best, consistent with presentation)
results_df = pd.DataFrame(results).T.sort_values(by='accuracy', ascending=False)

# Create Bar Chart for accuracy (used in your presentation)
plt.figure(figsize=(9, 6))
sns.barplot(x=results_df.index, y='accuracy', data=results_df, palette='Spectral')

plt.title('Comparison of Classification Model Performance (Testing accuracy)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.ylim(0.80, results_df['accuracy'].max() + 0.02) # Set y-limit for better visualization

# Add values on top of bars
for index, row in results_df.iterrows():
    plt.text(index, row.accuracy + 0.001, f"{row.accuracy:.3f}", color='black', ha="center", fontsize=11)

plt.savefig('classification_accuracy_comparison.png')
plt.show()