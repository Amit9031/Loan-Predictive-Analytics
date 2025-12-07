"""
Simple Machine Learning Model Training Script
No functions - Linear execution
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LOAN PREDICTION - MACHINE LEARNING MODEL TRAINING")
print("="*60)

# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1] Loading and preprocessing data...")
df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

# Drop Loan_ID
df = df.drop('Loan_ID', axis=1)

# Handle missing values - Numerical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Handle missing values - Categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                  'Self_Employed', 'Property_Area']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

# Handle '3+' in Dependents
df['Dependents'] = df['Dependents'].replace('3+', '3')

# Encode categorical variables
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Label encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)
label_encoders['Loan_Status'] = le_target

# Store feature names
feature_names = X.columns.tolist()

# Convert to numpy arrays
X = X.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data shape: {X.shape}")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Store models and results
models = {}
results = {}

# ==================== STEP 2: REGRESSION MODELS ====================
print("\n" + "="*60)
print("[STEP 2] TRAINING REGRESSION MODELS")
print("="*60)

# Simple Linear Regression
print("\n1. Simple Linear Regression...")
simple_lr = LinearRegression()
X_simple = X_train_scaled[:, 5:6]  # ApplicantIncome
y_simple = X_train_scaled[:, 7:8]  # LoanAmount
simple_lr.fit(X_simple, y_simple)
y_pred_simple = simple_lr.predict(X_test_scaled[:, 5:6])
y_test_simple = X_test_scaled[:, 7:8]

mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

results['Simple_Linear_Regression'] = {
    'MAE': float(mae_simple), 'MSE': float(mse_simple), 
    'RMSE': float(rmse_simple), 'R2': float(r2_simple)
}
models['Simple_Linear_Regression'] = simple_lr
print(f"MAE: {mae_simple:.4f}, MSE: {mse_simple:.4f}, RMSE: {rmse_simple:.4f}, R²: {r2_simple:.4f}")

# Multiple Linear Regression
print("\n2. Multiple Linear Regression...")
multiple_lr = LinearRegression()
multiple_lr.fit(X_train_scaled, y_train)
y_pred_multiple = multiple_lr.predict(X_test_scaled)

mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

results['Multiple_Linear_Regression'] = {
    'MAE': float(mae_multiple), 'MSE': float(mse_multiple), 
    'RMSE': float(rmse_multiple), 'R2': float(r2_multiple)
}
models['Multiple_Linear_Regression'] = multiple_lr
print(f"MAE: {mae_multiple:.4f}, MSE: {mse_multiple:.4f}, RMSE: {rmse_multiple:.4f}, R²: {r2_multiple:.4f}")

# Polynomial Regression
print("\n3. Polynomial Regression (degree=2)...")
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

results['Polynomial_Regression'] = {
    'MAE': float(mae_poly), 'MSE': float(mse_poly), 
    'RMSE': float(rmse_poly), 'R2': float(r2_poly)
}
models['Polynomial_Regression'] = poly_lr
models['Polynomial_Features'] = poly_features
print(f"MAE: {mae_poly:.4f}, MSE: {mse_poly:.4f}, RMSE: {rmse_poly:.4f}, R²: {r2_poly:.4f}")

# Logistic Regression
print("\n4. Logistic Regression...")
logistic_reg = LogisticRegression(max_iter=1000, random_state=42)
logistic_reg.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_reg.predict(X_test_scaled)
y_pred_proba_logistic = logistic_reg.predict_proba(X_test_scaled)[:, 1]

acc_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)
auc_logistic = roc_auc_score(y_test, y_pred_proba_logistic)
logloss_logistic = log_loss(y_test, y_pred_proba_logistic)
cm_logistic = confusion_matrix(y_test, y_pred_logistic)

results['Logistic_Regression'] = {
    'Accuracy': float(acc_logistic), 'Precision': float(precision_logistic), 
    'Recall': float(recall_logistic), 'F1_Score': float(f1_logistic),
    'AUC': float(auc_logistic), 'Log_Loss': float(logloss_logistic),
    'Confusion_Matrix': cm_logistic.tolist()
}
models['Logistic_Regression'] = logistic_reg
print(f"Accuracy: {acc_logistic:.4f}, Precision: {precision_logistic:.4f}, "
      f"Recall: {recall_logistic:.4f}, F1: {f1_logistic:.4f}, AUC: {auc_logistic:.4f}")

# ==================== STEP 3: CLASSIFICATION MODELS ====================
print("\n" + "="*60)
print("[STEP 3] TRAINING CLASSIFICATION MODELS")
print("="*60)

# Naive Bayes
print("\n1. Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
y_pred_proba_nb = nb.predict_proba(X_test_scaled)[:, 1]

acc_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_pred_proba_nb)
logloss_nb = log_loss(y_test, y_pred_proba_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

results['Naive_Bayes'] = {
    'Accuracy': float(acc_nb), 'Precision': float(precision_nb), 
    'Recall': float(recall_nb), 'F1_Score': float(f1_nb),
    'AUC': float(auc_nb), 'Log_Loss': float(logloss_nb),
    'Confusion_Matrix': cm_nb.tolist()
}
models['Naive_Bayes'] = nb
print(f"Accuracy: {acc_nb:.4f}, Precision: {precision_nb:.4f}, "
      f"Recall: {recall_nb:.4f}, F1: {f1_nb:.4f}, AUC: {auc_nb:.4f}")

# Decision Tree
print("\n2. Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
logloss_dt = log_loss(y_test, y_pred_proba_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

results['Decision_Tree'] = {
    'Accuracy': float(acc_dt), 'Precision': float(precision_dt), 
    'Recall': float(recall_dt), 'F1_Score': float(f1_dt),
    'AUC': float(auc_dt), 'Log_Loss': float(logloss_dt),
    'Confusion_Matrix': cm_dt.tolist()
}
models['Decision_Tree'] = dt
print(f"Accuracy: {acc_dt:.4f}, Precision: {precision_dt:.4f}, "
      f"Recall: {recall_dt:.4f}, F1: {f1_dt:.4f}, AUC: {auc_dt:.4f}")

# Support Vector Machine
print("\n3. Support Vector Machine...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_pred_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

acc_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
logloss_svm = log_loss(y_test, y_pred_proba_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

results['SVM'] = {
    'Accuracy': float(acc_svm), 'Precision': float(precision_svm), 
    'Recall': float(recall_svm), 'F1_Score': float(f1_svm),
    'AUC': float(auc_svm), 'Log_Loss': float(logloss_svm),
    'Confusion_Matrix': cm_svm.tolist()
}
models['SVM'] = svm
print(f"Accuracy: {acc_svm:.4f}, Precision: {precision_svm:.4f}, "
      f"Recall: {recall_svm:.4f}, F1: {f1_svm:.4f}, AUC: {auc_svm:.4f}")

# K-Nearest Neighbors
print("\n4. K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]

acc_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
logloss_knn = log_loss(y_test, y_pred_proba_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

results['KNN'] = {
    'Accuracy': float(acc_knn), 'Precision': float(precision_knn), 
    'Recall': float(recall_knn), 'F1_Score': float(f1_knn),
    'AUC': float(auc_knn), 'Log_Loss': float(logloss_knn),
    'Confusion_Matrix': cm_knn.tolist()
}
models['KNN'] = knn
print(f"Accuracy: {acc_knn:.4f}, Precision: {precision_knn:.4f}, "
      f"Recall: {recall_knn:.4f}, F1: {f1_knn:.4f}, AUC: {auc_knn:.4f}")

# Random Forest
print("\n5. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
logloss_rf = log_loss(y_test, y_pred_proba_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

results['Random_Forest'] = {
    'Accuracy': float(acc_rf), 'Precision': float(precision_rf), 
    'Recall': float(recall_rf), 'F1_Score': float(f1_rf),
    'AUC': float(auc_rf), 'Log_Loss': float(logloss_rf),
    'Confusion_Matrix': cm_rf.tolist()
}
models['Random_Forest'] = rf
print(f"Accuracy: {acc_rf:.4f}, Precision: {precision_rf:.4f}, "
      f"Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")

# Bagging
print("\n6. Bagging Classifier...")
try:
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                n_estimators=10, random_state=42)
except TypeError:
    bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), 
                                n_estimators=10, random_state=42)
bagging.fit(X_train_scaled, y_train)
y_pred_bag = bagging.predict(X_test_scaled)
y_pred_proba_bag = bagging.predict_proba(X_test_scaled)[:, 1]

acc_bag = accuracy_score(y_test, y_pred_bag)
precision_bag = precision_score(y_test, y_pred_bag)
recall_bag = recall_score(y_test, y_pred_bag)
f1_bag = f1_score(y_test, y_pred_bag)
auc_bag = roc_auc_score(y_test, y_pred_proba_bag)
logloss_bag = log_loss(y_test, y_pred_proba_bag)
cm_bag = confusion_matrix(y_test, y_pred_bag)

results['Bagging'] = {
    'Accuracy': float(acc_bag), 'Precision': float(precision_bag), 
    'Recall': float(recall_bag), 'F1_Score': float(f1_bag),
    'AUC': float(auc_bag), 'Log_Loss': float(logloss_bag),
    'Confusion_Matrix': cm_bag.tolist()
}
models['Bagging'] = bagging
print(f"Accuracy: {acc_bag:.4f}, Precision: {precision_bag:.4f}, "
      f"Recall: {recall_bag:.4f}, F1: {f1_bag:.4f}, AUC: {auc_bag:.4f}")

# AdaBoost
print("\n7. AdaBoost Classifier...")
boosting = AdaBoostClassifier(n_estimators=50, random_state=42)
boosting.fit(X_train_scaled, y_train)
y_pred_boost = boosting.predict(X_test_scaled)
y_pred_proba_boost = boosting.predict_proba(X_test_scaled)[:, 1]

acc_boost = accuracy_score(y_test, y_pred_boost)
precision_boost = precision_score(y_test, y_pred_boost)
recall_boost = recall_score(y_test, y_pred_boost)
f1_boost = f1_score(y_test, y_pred_boost)
auc_boost = roc_auc_score(y_test, y_pred_proba_boost)
logloss_boost = log_loss(y_test, y_pred_proba_boost)
cm_boost = confusion_matrix(y_test, y_pred_boost)

results['AdaBoost'] = {
    'Accuracy': float(acc_boost), 'Precision': float(precision_boost), 
    'Recall': float(recall_boost), 'F1_Score': float(f1_boost),
    'AUC': float(auc_boost), 'Log_Loss': float(logloss_boost),
    'Confusion_Matrix': cm_boost.tolist()
}
models['AdaBoost'] = boosting
print(f"Accuracy: {acc_boost:.4f}, Precision: {precision_boost:.4f}, "
      f"Recall: {recall_boost:.4f}, F1: {f1_boost:.4f}, AUC: {auc_boost:.4f}")

# ==================== STEP 4: CLUSTERING MODELS ====================
print("\n" + "="*60)
print("[STEP 4] TRAINING CLUSTERING MODELS")
print("="*60)

# K-Means Clustering
print("\n1. K-Means Clustering (k=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
labels_kmeans = kmeans.predict(X_test_scaled)
silhouette_kmeans = silhouette_score(X_test_scaled, labels_kmeans)

results['K_Means'] = {
    'Silhouette_Score': float(silhouette_kmeans),
    'Clusters': 2
}
models['K_Means'] = kmeans
print(f"Silhouette Score: {silhouette_kmeans:.4f}")

# Hierarchical Clustering
print("\n2. Hierarchical Clustering (Agglomerative)...")
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X_test_scaled)
silhouette_hierarchical = silhouette_score(X_test_scaled, labels_hierarchical)

results['Hierarchical_Clustering'] = {
    'Silhouette_Score': float(silhouette_hierarchical),
    'Clusters': 2
}
models['Hierarchical_Clustering'] = hierarchical
print(f"Silhouette Score: {silhouette_hierarchical:.4f}")

# ==================== STEP 5: PCA ====================
print("\n" + "="*60)
print("[STEP 5] PRINCIPAL COMPONENT ANALYSIS")
print("="*60)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"PCA components: {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Train Random Forest with PCA
print("\nTraining Random Forest with PCA features...")
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

results['PCA_Random_Forest'] = {
    'Accuracy': float(acc_pca),
    'Components': int(X_train_pca.shape[1])
}
models['PCA_Random_Forest'] = rf_pca
print(f"Accuracy with PCA: {acc_pca:.4f}")

# ==================== STEP 6: NEURAL NETWORKS ====================
print("\n" + "="*60)
print("[STEP 6] TRAINING NEURAL NETWORKS")
print("="*60)

try:
    import tensorflow as tf
    from tensorflow import keras
    try:
        from keras.models import Sequential
        from keras.layers import Dense, Conv1D, Flatten, LSTM
        from keras.optimizers import Adam
    except ImportError:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM
        from tensorflow.keras.optimizers import Adam
    
    # MLP
    print("\n1. Multi-layer Perceptron (MLP)...")
    mlp = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer=Adam(learning_rate=0.001), 
               loss='binary_crossentropy', 
               metrics=['accuracy'])
    mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
           validation_split=0.2, verbose=0)
    
    y_pred_proba_mlp = mlp.predict(X_test_scaled, verbose=0)
    y_pred_mlp = (y_pred_proba_mlp > 0.5).astype(int).flatten()
    
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    precision_mlp = precision_score(y_test, y_pred_mlp)
    recall_mlp = recall_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    auc_mlp = roc_auc_score(y_test, y_pred_proba_mlp)
    
    results['MLP'] = {
        'Accuracy': float(acc_mlp), 'Precision': float(precision_mlp),
        'Recall': float(recall_mlp), 'F1_Score': float(f1_mlp), 'AUC': float(auc_mlp)
    }
    models['MLP'] = mlp
    print(f"Accuracy: {acc_mlp:.4f}, Precision: {precision_mlp:.4f}, "
          f"Recall: {recall_mlp:.4f}, F1: {f1_mlp:.4f}, AUC: {auc_mlp:.4f}")
    
    # CNN
    print("\n2. Convolutional Neural Network (1D CNN)...")
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    cnn = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer=Adam(learning_rate=0.001), 
               loss='binary_crossentropy', 
               metrics=['accuracy'])
    cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, 
           validation_split=0.2, verbose=0)
    
    y_pred_proba_cnn = cnn.predict(X_test_cnn, verbose=0)
    y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int).flatten()
    
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    precision_cnn = precision_score(y_test, y_pred_cnn)
    recall_cnn = recall_score(y_test, y_pred_cnn)
    f1_cnn = f1_score(y_test, y_pred_cnn)
    auc_cnn = roc_auc_score(y_test, y_pred_proba_cnn)
    
    results['CNN'] = {
        'Accuracy': float(acc_cnn), 'Precision': float(precision_cnn),
        'Recall': float(recall_cnn), 'F1_Score': float(f1_cnn), 'AUC': float(auc_cnn)
    }
    models['CNN'] = cnn
    print(f"Accuracy: {acc_cnn:.4f}, Precision: {precision_cnn:.4f}, "
          f"Recall: {recall_cnn:.4f}, F1: {f1_cnn:.4f}, AUC: {auc_cnn:.4f}")
    
    # LSTM
    print("\n3. Recurrent Neural Network (LSTM)...")
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer=Adam(learning_rate=0.001), 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, 
            validation_split=0.2, verbose=0)
    
    y_pred_proba_lstm = lstm.predict(X_test_lstm, verbose=0)
    y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int).flatten()
    
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    precision_lstm = precision_score(y_test, y_pred_lstm)
    recall_lstm = recall_score(y_test, y_pred_lstm)
    f1_lstm = f1_score(y_test, y_pred_lstm)
    auc_lstm = roc_auc_score(y_test, y_pred_proba_lstm)
    
    results['RNN_LSTM'] = {
        'Accuracy': float(acc_lstm), 'Precision': float(precision_lstm),
        'Recall': float(recall_lstm), 'F1_Score': float(f1_lstm), 'AUC': float(auc_lstm)
    }
    models['RNN_LSTM'] = lstm
    print(f"Accuracy: {acc_lstm:.4f}, Precision: {precision_lstm:.4f}, "
          f"Recall: {recall_lstm:.4f}, F1: {f1_lstm:.4f}, AUC: {auc_lstm:.4f}")
    
except Exception as e:
    print(f"Error training neural networks: {e}")

# ==================== STEP 7: CROSS-VALIDATION ====================
print("\n" + "="*60)
print("[STEP 7] CROSS-VALIDATION EVALUATION")
print("="*60)

# K-Fold Cross-Validation
print("\nK-Fold Cross-Validation (k=5)...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_models = ['Logistic_Regression', 'Random_Forest', 'SVM', 'Decision_Tree']
cv_results = {}

for model_name in cv_models:
    if model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X_train_scaled, y_train, 
                               cv=kfold, scoring='accuracy')
        cv_results[model_name] = {
            'Mean_Accuracy': float(scores.mean()),
            'Std_Accuracy': float(scores.std())
        }
        print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

results['CV_Results'] = cv_results

# Leave-One-Out Cross-Validation
print("\nLeave-One-Out Cross-Validation (on 50 samples)...")
loo = LeaveOneOut()
X_loo = X_train_scaled[:50]
y_loo = y_train[:50]

loo_scores = []
for train_idx, test_idx in loo.split(X_loo):
    X_train_loo, X_test_loo = X_loo[train_idx], X_loo[test_idx]
    y_train_loo, y_test_loo = y_loo[train_idx], y_loo[test_idx]
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_loo, y_train_loo)
    score = model.score(X_test_loo, y_test_loo)
    loo_scores.append(score)

loo_mean = np.mean(loo_scores)
results['LOO_CV'] = {'Mean_Accuracy': float(loo_mean)}
print(f"LOO CV Accuracy: {loo_mean:.4f}")

# ==================== STEP 8: MODEL COMPARISON ====================
print("\n" + "="*60)
print("[STEP 8] MODEL COMPARISON")
print("="*60)

# Get classification models with accuracy
classification_models = {}
for model_name, model_results in results.items():
    if isinstance(model_results, dict) and 'Accuracy' in model_results:
        classification_models[model_name] = model_results['Accuracy']

# Sort by accuracy
sorted_models = sorted(classification_models.items(), key=lambda x: x[1], reverse=True)

print("\nModel Rankings (by Accuracy):")
print("-" * 60)
for i, (model_name, accuracy) in enumerate(sorted_models, 1):
    print(f"{i}. {model_name}: {accuracy:.4f}")

# Select top 2 models
if len(sorted_models) >= 2:
    best_model_1 = sorted_models[0][0]
    best_model_2 = sorted_models[1][0]
    
    print(f"\nTop 2 Models Selected:")
    print(f"1. {best_model_1} (Accuracy: {sorted_models[0][1]:.4f})")
    print(f"2. {best_model_2} (Accuracy: {sorted_models[1][1]:.4f})")
    
    best_models = [best_model_1, best_model_2]
else:
    best_models = [sorted_models[0][0]] if sorted_models else []

# ==================== STEP 9: SAVE MODELS ====================
print("\n" + "="*60)
print("[STEP 9] SAVING MODELS")
print("="*60)

# Save scaler and label encoders
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# Save all models
for model_name, model in models.items():
    if model_name not in ['Polynomial_Features']:
        if 'MLP' in model_name or 'CNN' in model_name or 'LSTM' in model_name:
            model.save(f'{model_name.lower()}.h5')
        else:
            joblib.dump(model, f'{model_name.lower()}.pkl')

# Save PCA
joblib.dump(pca, 'pca.pkl')

# Save results
with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Models and results saved successfully!")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nTotal Models Trained: {len(models)}")
print(f"Best Model: {best_models[0] if best_models else 'N/A'}")
print(f"Best Accuracy: {sorted_models[0][1]:.4f}" if sorted_models else "N/A")
print("\nYou can now run the web application with: streamlit run app.py")
print("="*60)
