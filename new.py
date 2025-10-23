
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
#2)MLP (Fully Connected Neural Network)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


df = pd.read_excel('./dataset/RSW-dataset.xlsx')


# Encode material types
le = LabelEncoder()
df['MATERIAL.1'] = le.fit_transform(df['MATERIAL.1'])
df['MATERIAL.2'] = le.fit_transform(df['MATERIAL.2'])


plt.figure(figsize=(8,4))
sns.histplot(df['NUGGET.WIDTH.1'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Nugget Width (Target Variable)")
plt.xlabel("Nugget Width (mm)")
plt.ylabel("Count")
plt.show()


#Correlation Analysis
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

df.corr()['NUGGET.WIDTH.1'].sort_values(ascending=False)


#Regression Task — Predict Nugget Width
X = df.drop('NUGGET.WIDTH.1', axis=1)
y = df['NUGGET.WIDTH.1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"\n{name}")
    print("R²:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    
#Visualize Regression Performance
# Plot Predicted vs Actual
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
preds = rf.predict(X_test_scaled)

plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Nugget Width")
plt.ylabel("Predicted Nugget Width")
plt.title("Predicted vs Actual Nugget Width (Random Forest)")
plt.plot([0,10],[0,10],'r--')
plt.show()




# Create Classification Dataset
# Define threshold based on domain assumption
df['NUGGET_CLASS'] = df['NUGGET.WIDTH.1'].apply(lambda x: 1 if 4 <= x <= 7 else 0)

df['NUGGET_CLASS'].value_counts()
sns.countplot(data=df, x='NUGGET_CLASS')
plt.title("Class Distribution: Acceptable (1) vs Unacceptable (0)")
plt.show()


#Train Classification Model
X = df.drop(['NUGGET.WIDTH.1', 'NUGGET_CLASS'], axis=1)
y = df['NUGGET_CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix and Feature Importance
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Acceptable vs Unacceptable Welds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance (Random Forest)")
plt.show()



# To “expand dataset for classification,” simulate small perturbations (noise) on numeric values — a data augmentation strategy.
df_aug = df.copy()
for col in ['WELD.CURRENT', 'WELD.FORCE', 'WELD.TIME']:
    df_aug[col] = df_aug[col] + np.random.normal(0, 0.05, len(df_aug))

expanded_df = pd.concat([df, df_aug])
print("Original:", len(df), "Expanded:", len(expanded_df))

# Define Multi-Class Labels
#Nugget Width (mm)	Class	Meaning
# < 3.5	            0	Small (Defective)
# 3.5 – 5.5	       1	   Medium (Acceptable)
# 5.5 – 7.5	       2	  Large (Acceptable)
# > 7.5	           3	  Oversized (Defective)

# Step 1: Create multi-class target variable
def nugget_class(x):
    if x < 3.5:
        return 0
    elif 3.5 <= x < 5.5:
        return 1
    elif 5.5 <= x < 7.5:
        return 2
    else:
        return 3

df['NUGGET_CLASS_MULTI'] = df['NUGGET.WIDTH.1'].apply(nugget_class)
sns.countplot(data=df, x='NUGGET_CLASS_MULTI')
plt.title("Multi-Class Distribution of Nugget Width")
plt.show()

X = df.drop(['NUGGET.WIDTH.1', 'NUGGET_CLASS_MULTI'], axis=1)
y = df['NUGGET_CLASS_MULTI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#1)Random Forest Classifier (Baseline)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest (Multi-Class)")
plt.show()





# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build MLP
mlp = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train_cat.shape[1], activation='softmax')
])

mlp.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = mlp.fit(X_train, y_train_cat, validation_split=0.2, epochs=200, batch_size=32, verbose=1)


plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("MLP Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



#3)CNN (1D Convolution on Tabular Data)

# Reshape for CNN input: (samples, timesteps, features)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_train_cat.shape[1], activation='softmax')
])

cnn.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn.fit(X_train_cnn, y_train_cat, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

plt.plot(history_cnn.history['accuracy'], label='Train')
plt.plot(history_cnn.history['val_accuracy'], label='Validation')
plt.title("CNN Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#Model Comparison
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

# Evaluate MLP
mlp_acc = mlp.evaluate(X_test, y_test_cat, verbose=0)[1]
cnn_acc = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
rf_acc = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {rf_acc:.3f}")
print(f"MLP Accuracy: {mlp_acc:.3f}")
print(f"CNN Accuracy: {cnn_acc:.3f}")
