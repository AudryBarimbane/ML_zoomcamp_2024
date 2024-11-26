#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle


# In[3]:


# 1. Download historical stock data
ticker = 'OR.PA'  # L'Oréal ticker on the Paris Stock Exchange
data = yf.download(ticker, start='2020-01-01', progress=False)
data.to_csv('loreal_historical_data.csv')  # Save data to a CSV file


# In[5]:


# Reload data with date processing
data = pd.read_csv('loreal_historical_data.csv', parse_dates=['Date'], index_col='Date')


# In[7]:


# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())


# In[9]:


# 2. Compute technical indicators
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA30'] = data['Close'].rolling(window=30).mean()
data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Volatility'] = data['Log Return'].rolling(window=30).std() * np.sqrt(30)


# In[11]:


# RSI calculation
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))


# In[13]:


# Fill missing values
data = data.fillna(0)


# In[15]:


# 3. Prepare the dataset
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # Create target variable


# In[17]:


# Select features
features = ['MA5', 'MA10', 'MA30', 'Volatility', 'RSI']
X = data[features]
y = data['Target']


# In[19]:


# Split the dataset
train_size = 0.6
validation_size = 0.2


# In[23]:


train_end = int(len(data) * train_size)
validation_end = int(len(data) * (train_size + validation_size))

X_train = X[:train_end]
y_train = y[:train_end]
X_validation = X[train_end:validation_end]
y_validation = y[train_end:validation_end]
X_test = X[validation_end:]
y_test = y[validation_end:]


# In[25]:


# Standardize data only for models that require it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)


# In[27]:


# 4. Train and evaluate models
def evaluate_model(model, X_train, y_train, X_validation, y_validation, X_test, y_test, scale=False):
    # Apply scaling if needed
    if scale:
        X_train, X_validation, X_test = X_train_scaled, X_validation_scaled, X_test_scaled
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_validation = model.predict(X_validation)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    validation_accuracy = accuracy_score(y_validation, y_pred_validation)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    validation_roc_auc = roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1]) if hasattr(model, "predict_proba") else None
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
    
    # Summary
    print(f"Validation Accuracy: {validation_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
    if validation_roc_auc and test_roc_auc:
        print(f"Validation ROC-AUC: {validation_roc_auc:.2f}, Test ROC-AUC: {test_roc_auc:.2f}")
    
    print("Classification Report (Validation):")
    print(classification_report(y_validation, y_pred_validation))
    print("Confusion Matrix (Validation):")
    print(confusion_matrix(y_validation, y_pred_validation))
    
    # Save the trained model with pickle
    model_name = model.__class__.__name__
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model {model_name} saved successfully!")
    return validation_accuracy, test_accuracy, validation_roc_auc, test_roc_auc


# In[29]:


# Logistic Regression
print("\nLogistic Regression:")
logistic_model = LogisticRegression()
evaluate_model(logistic_model, X_train, y_train, X_validation, y_validation, X_test, y_test, scale=True)


# In[31]:


# Random Forest
print("\nRandom Forest:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train, y_train, X_validation, y_validation, X_test, y_test)


# In[33]:


# XGBoost
print("\nXGBoost:")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
evaluate_model(xgb_model, X_train, y_train, X_validation, y_validation, X_test, y_test)


# In[35]:


# 5. Compare models using ROC curves
def plot_roc_curves(models, names, X_validation, y_validation, X_test, y_test):
    plt.figure(figsize=(12, 6))
    for model, name in zip(models, names):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

# Plot ROC curves for all models
plot_roc_curves(
    [logistic_model, rf_model, xgb_model],
    ["Logistic Regression", "Random Forest", "XGBoost"],
    X_validation_scaled, y_validation, X_test_scaled, y_test
)


# In[37]:


# 6. Load a model and use it for prediction
# Load the logistic regression model for example
with open('LogisticRegression_model.pkl', 'rb') as f:
    loaded_logistic_model = pickle.load(f)




# In[39]:


# Make predictions with the loaded model
y_pred_loaded_model = loaded_logistic_model.predict(X_test_scaled)
print("\nPredictions with Loaded Model:")
print(y_pred_loaded_model[:10])  # Print the first 10 predictions


# In[ ]:




