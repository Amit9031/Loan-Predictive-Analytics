

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
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        
    def load_and_preprocess_data(self):
     
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path)
        
    
        df = df.drop('Loan_ID', axis=1)
        
     
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History']
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        
       
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area']
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
       
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        
       
        categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 
                               'Self_Employed', 'Property_Area']
        
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        
     
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        self.label_encoders['Loan_Status'] = le_target
        
       
        self.feature_names = X.columns.tolist()
        
     
        X = X.values
    
        
      
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
       
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data shape: {X.shape}")
        print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        return self
    
    def calculate_correlations(self):
        """Calculate correlation matrix"""
        print("\nCalculating correlations...")
        df_corr = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        correlations = df_corr.corr()
        return correlations
    
    def train_regression_models(self):
        """Train regression models"""
        print("\n" + "="*50)
        print("TRAINING REGRESSION MODELS")
        print("="*50)
        
   
        print("\n1. Simple Linear Regression...")
        simple_lr = LinearRegression()
      
        X_simple = self.X_train_scaled[:, 5:6] 
        y_simple = self.X_train_scaled[:, 7:8] 
        simple_lr.fit(X_simple, y_simple)
        y_pred_simple = simple_lr.predict(self.X_test_scaled[:, 5:6])
        y_test_simple = self.X_test_scaled[:, 7:8]
        
        mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
        mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
        rmse_simple = np.sqrt(mse_simple)
        r2_simple = r2_score(y_test_simple, y_pred_simple)
        
        self.results['Simple_Linear_Regression'] = {
            'MAE': mae_simple, 'MSE': mse_simple, 'RMSE': rmse_simple, 'R2': r2_simple
        }
        self.models['Simple_Linear_Regression'] = simple_lr
        print(f"MAE: {mae_simple:.4f}, MSE: {mse_simple:.4f}, RMSE: {rmse_simple:.4f}, R²: {r2_simple:.4f}")
        
        
        print("\n2. Multiple Linear Regression...")
        multiple_lr = LinearRegression()
        multiple_lr.fit(self.X_train_scaled, self.y_train)
        y_pred_multiple = multiple_lr.predict(self.X_test_scaled)
        
        mae_multiple = mean_absolute_error(self.y_test, y_pred_multiple)
        mse_multiple = mean_squared_error(self.y_test, y_pred_multiple)
        rmse_multiple = np.sqrt(mse_multiple)
        r2_multiple = r2_score(self.y_test, y_pred_multiple)
        
        self.results['Multiple_Linear_Regression'] = {
            'MAE': mae_multiple, 'MSE': mse_multiple, 'RMSE': rmse_multiple, 'R2': r2_multiple
        }
        self.models['Multiple_Linear_Regression'] = multiple_lr
        print(f"MAE: {mae_multiple:.4f}, MSE: {mse_multiple:.4f}, RMSE: {rmse_multiple:.4f}, R²: {r2_multiple:.4f}")
        
      
        print("\n3. Polynomial Regression (degree=2)...")
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(self.X_train_scaled)
        X_test_poly = poly_features.transform(self.X_test_scaled)
        
        poly_lr = LinearRegression()
        poly_lr.fit(X_train_poly, self.y_train)
        y_pred_poly = poly_lr.predict(X_test_poly)
        
        mae_poly = mean_absolute_error(self.y_test, y_pred_poly)
        mse_poly = mean_squared_error(self.y_test, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)
        r2_poly = r2_score(self.y_test, y_pred_poly)
        
        self.results['Polynomial_Regression'] = {
            'MAE': mae_poly, 'MSE': mse_poly, 'RMSE': rmse_poly, 'R2': r2_poly
        }
        self.models['Polynomial_Regression'] = poly_lr
        self.models['Polynomial_Features'] = poly_features
        print(f"MAE: {mae_poly:.4f}, MSE: {mse_poly:.4f}, RMSE: {rmse_poly:.4f}, R²: {r2_poly:.4f}")
        
        
        print("\n4. Logistic Regression...")
        logistic_reg = LogisticRegression(max_iter=1000, random_state=42)
        logistic_reg.fit(self.X_train_scaled, self.y_train)
        y_pred_logistic = logistic_reg.predict(self.X_test_scaled)
        y_pred_proba_logistic = logistic_reg.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_logistic = accuracy_score(self.y_test, y_pred_logistic)
        precision_logistic = precision_score(self.y_test, y_pred_logistic)
        recall_logistic = recall_score(self.y_test, y_pred_logistic)
        f1_logistic = f1_score(self.y_test, y_pred_logistic)
        auc_logistic = roc_auc_score(self.y_test, y_pred_proba_logistic)
        logloss_logistic = log_loss(self.y_test, y_pred_proba_logistic)
        cm_logistic = confusion_matrix(self.y_test, y_pred_logistic)
        
        self.results['Logistic_Regression'] = {
            'Accuracy': acc_logistic, 'Precision': precision_logistic, 
            'Recall': recall_logistic, 'F1_Score': f1_logistic,
            'AUC': auc_logistic, 'Log_Loss': logloss_logistic,
            'Confusion_Matrix': cm_logistic
        }
        self.models['Logistic_Regression'] = logistic_reg
        print(f"Accuracy: {acc_logistic:.4f}, Precision: {precision_logistic:.4f}, "
              f"Recall: {recall_logistic:.4f}, F1: {f1_logistic:.4f}, AUC: {auc_logistic:.4f}")
    
    def train_classification_models(self):
        """Train classification models"""
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*50)
        
        
        print("\n1. Naive Bayes...")
        nb = GaussianNB()
        nb.fit(self.X_train_scaled, self.y_train)
        y_pred_nb = nb.predict(self.X_test_scaled)
        y_pred_proba_nb = nb.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_nb = accuracy_score(self.y_test, y_pred_nb)
        precision_nb = precision_score(self.y_test, y_pred_nb)
        recall_nb = recall_score(self.y_test, y_pred_nb)
        f1_nb = f1_score(self.y_test, y_pred_nb)
        auc_nb = roc_auc_score(self.y_test, y_pred_proba_nb)
        logloss_nb = log_loss(self.y_test, y_pred_proba_nb)
        cm_nb = confusion_matrix(self.y_test, y_pred_nb)
        
        self.results['Naive_Bayes'] = {
            'Accuracy': acc_nb, 'Precision': precision_nb, 
            'Recall': recall_nb, 'F1_Score': f1_nb,
            'AUC': auc_nb, 'Log_Loss': logloss_nb,
            'Confusion_Matrix': cm_nb
        }
        self.models['Naive_Bayes'] = nb
        print(f"Accuracy: {acc_nb:.4f}, Precision: {precision_nb:.4f}, "
              f"Recall: {recall_nb:.4f}, F1: {f1_nb:.4f}, AUC: {auc_nb:.4f}")
        
       
        print("\n2. Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt.fit(self.X_train_scaled, self.y_train)
        y_pred_dt = dt.predict(self.X_test_scaled)
        y_pred_proba_dt = dt.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_dt = accuracy_score(self.y_test, y_pred_dt)
        precision_dt = precision_score(self.y_test, y_pred_dt)
        recall_dt = recall_score(self.y_test, y_pred_dt)
        f1_dt = f1_score(self.y_test, y_pred_dt)
        auc_dt = roc_auc_score(self.y_test, y_pred_proba_dt)
        logloss_dt = log_loss(self.y_test, y_pred_proba_dt)
        cm_dt = confusion_matrix(self.y_test, y_pred_dt)
        
        self.results['Decision_Tree'] = {
            'Accuracy': acc_dt, 'Precision': precision_dt, 
            'Recall': recall_dt, 'F1_Score': f1_dt,
            'AUC': auc_dt, 'Log_Loss': logloss_dt,
            'Confusion_Matrix': cm_dt
        }
        self.models['Decision_Tree'] = dt
        print(f"Accuracy: {acc_dt:.4f}, Precision: {precision_dt:.4f}, "
              f"Recall: {recall_dt:.4f}, F1: {f1_dt:.4f}, AUC: {auc_dt:.4f}")
        
        
        print("\n3. Support Vector Machine...")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(self.X_train_scaled, self.y_train)
        y_pred_svm = svm.predict(self.X_test_scaled)
        y_pred_proba_svm = svm.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_svm = accuracy_score(self.y_test, y_pred_svm)
        precision_svm = precision_score(self.y_test, y_pred_svm)
        recall_svm = recall_score(self.y_test, y_pred_svm)
        f1_svm = f1_score(self.y_test, y_pred_svm)
        auc_svm = roc_auc_score(self.y_test, y_pred_proba_svm)
        logloss_svm = log_loss(self.y_test, y_pred_proba_svm)
        cm_svm = confusion_matrix(self.y_test, y_pred_svm)
        
        self.results['SVM'] = {
            'Accuracy': acc_svm, 'Precision': precision_svm, 
            'Recall': recall_svm, 'F1_Score': f1_svm,
            'AUC': auc_svm, 'Log_Loss': logloss_svm,
            'Confusion_Matrix': cm_svm
        }
        self.models['SVM'] = svm
        print(f"Accuracy: {acc_svm:.4f}, Precision: {precision_svm:.4f}, "
              f"Recall: {recall_svm:.4f}, F1: {f1_svm:.4f}, AUC: {auc_svm:.4f}")
        
      
        print("\n4. K-Nearest Neighbors...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train_scaled, self.y_train)
        y_pred_knn = knn.predict(self.X_test_scaled)
        y_pred_proba_knn = knn.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_knn = accuracy_score(self.y_test, y_pred_knn)
        precision_knn = precision_score(self.y_test, y_pred_knn)
        recall_knn = recall_score(self.y_test, y_pred_knn)
        f1_knn = f1_score(self.y_test, y_pred_knn)
        auc_knn = roc_auc_score(self.y_test, y_pred_proba_knn)
        logloss_knn = log_loss(self.y_test, y_pred_proba_knn)
        cm_knn = confusion_matrix(self.y_test, y_pred_knn)
        
        self.results['KNN'] = {
            'Accuracy': acc_knn, 'Precision': precision_knn, 
            'Recall': recall_knn, 'F1_Score': f1_knn,
            'AUC': auc_knn, 'Log_Loss': logloss_knn,
            'Confusion_Matrix': cm_knn
        }
        self.models['KNN'] = knn
        print(f"Accuracy: {acc_knn:.4f}, Precision: {precision_knn:.4f}, "
              f"Recall: {recall_knn:.4f}, F1: {f1_knn:.4f}, AUC: {auc_knn:.4f}")
        
        
        print("\n5. Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(self.X_train_scaled, self.y_train)
        y_pred_rf = rf.predict(self.X_test_scaled)
        y_pred_proba_rf = rf.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_rf = accuracy_score(self.y_test, y_pred_rf)
        precision_rf = precision_score(self.y_test, y_pred_rf)
        recall_rf = recall_score(self.y_test, y_pred_rf)
        f1_rf = f1_score(self.y_test, y_pred_rf)
        auc_rf = roc_auc_score(self.y_test, y_pred_proba_rf)
        logloss_rf = log_loss(self.y_test, y_pred_proba_rf)
        cm_rf = confusion_matrix(self.y_test, y_pred_rf)
        
        self.results['Random_Forest'] = {
            'Accuracy': acc_rf, 'Precision': precision_rf, 
            'Recall': recall_rf, 'F1_Score': f1_rf,
            'AUC': auc_rf, 'Log_Loss': logloss_rf,
            'Confusion_Matrix': cm_rf
        }
        self.models['Random_Forest'] = rf
        print(f"Accuracy: {acc_rf:.4f}, Precision: {precision_rf:.4f}, "
              f"Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")
        
        
        print("\n6. Bagging Classifier...")
        try:
       
            bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                        n_estimators=10, random_state=42)
        except TypeError:
            
            bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), 
                                        n_estimators=10, random_state=42)
        bagging.fit(self.X_train_scaled, self.y_train)
        y_pred_bag = bagging.predict(self.X_test_scaled)
        y_pred_proba_bag = bagging.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_bag = accuracy_score(self.y_test, y_pred_bag)
        precision_bag = precision_score(self.y_test, y_pred_bag)
        recall_bag = recall_score(self.y_test, y_pred_bag)
        f1_bag = f1_score(self.y_test, y_pred_bag)
        auc_bag = roc_auc_score(self.y_test, y_pred_proba_bag)
        logloss_bag = log_loss(self.y_test, y_pred_proba_bag)
        cm_bag = confusion_matrix(self.y_test, y_pred_bag)
        
        self.results['Bagging'] = {
            'Accuracy': acc_bag, 'Precision': precision_bag, 
            'Recall': recall_bag, 'F1_Score': f1_bag,
            'AUC': auc_bag, 'Log_Loss': logloss_bag,
            'Confusion_Matrix': cm_bag
        }
        self.models['Bagging'] = bagging
        print(f"Accuracy: {acc_bag:.4f}, Precision: {precision_bag:.4f}, "
              f"Recall: {recall_bag:.4f}, F1: {f1_bag:.4f}, AUC: {auc_bag:.4f}")
        
       
        print("\n7. AdaBoost Classifier...")
        boosting = AdaBoostClassifier(n_estimators=50, random_state=42)
        boosting.fit(self.X_train_scaled, self.y_train)
        y_pred_boost = boosting.predict(self.X_test_scaled)
        y_pred_proba_boost = boosting.predict_proba(self.X_test_scaled)[:, 1]
        
        acc_boost = accuracy_score(self.y_test, y_pred_boost)
        precision_boost = precision_score(self.y_test, y_pred_boost)
        recall_boost = recall_score(self.y_test, y_pred_boost)
        f1_boost = f1_score(self.y_test, y_pred_boost)
        auc_boost = roc_auc_score(self.y_test, y_pred_proba_boost)
        logloss_boost = log_loss(self.y_test, y_pred_proba_boost)
        cm_boost = confusion_matrix(self.y_test, y_pred_boost)
        
        self.results['AdaBoost'] = {
            'Accuracy': acc_boost, 'Precision': precision_boost, 
            'Recall': recall_boost, 'F1_Score': f1_boost,
            'AUC': auc_boost, 'Log_Loss': logloss_boost,
            'Confusion_Matrix': cm_boost
        }
        self.models['AdaBoost'] = boosting
        print(f"Accuracy: {acc_boost:.4f}, Precision: {precision_boost:.4f}, "
              f"Recall: {recall_boost:.4f}, F1: {f1_boost:.4f}, AUC: {auc_boost:.4f}")
    
    def train_clustering_models(self):
        """Train clustering models"""
        print("\n" + "="*50)
        print("TRAINING CLUSTERING MODELS")
        print("="*50)
        
     
        print("\n1. K-Means Clustering (k=2)...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(self.X_train_scaled)
        labels_kmeans = kmeans.predict(self.X_test_scaled)
        
        
        from sklearn.metrics import silhouette_score
        silhouette_kmeans = silhouette_score(self.X_test_scaled, labels_kmeans)
        
        self.results['K_Means'] = {
            'Silhouette_Score': silhouette_kmeans,
            'Clusters': 2
        }
        self.models['K_Means'] = kmeans
        print(f"Silhouette Score: {silhouette_kmeans:.4f}")
        
       
        print("\n2. Hierarchical Clustering (Agglomerative)...")
        hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
        labels_hierarchical = hierarchical.fit_predict(self.X_test_scaled)
        silhouette_hierarchical = silhouette_score(self.X_test_scaled, labels_hierarchical)
        
        self.results['Hierarchical_Clustering'] = {
            'Silhouette_Score': silhouette_hierarchical,
            'Clusters': 2
        }
        self.models['Hierarchical_Clustering'] = hierarchical
        print(f"Silhouette Score: {silhouette_hierarchical:.4f}")
    
    def apply_pca(self):
        """Apply Principal Component Analysis"""
        print("\n" + "="*50)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*50)
        

        self.pca = PCA(n_components=0.95) 
        X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        X_test_pca = self.pca.transform(self.X_test_scaled)
        
        print(f"Original features: {self.X_train_scaled.shape[1]}")
        print(f"PCA components: {X_train_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
       
        print("\nTraining Random Forest with PCA features...")
        rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pca.fit(X_train_pca, self.y_train)
        y_pred_pca = rf_pca.predict(X_test_pca)
        
        acc_pca = accuracy_score(self.y_test, y_pred_pca)
        self.results['PCA_Random_Forest'] = {
            'Accuracy': acc_pca,
            'Components': X_train_pca.shape[1]
        }
        self.models['PCA_Random_Forest'] = rf_pca
        print(f"Accuracy with PCA: {acc_pca:.4f}")
        
        return X_train_pca, X_test_pca
    
    def train_neural_networks(self):
        """Train neural network models"""
        print("\n" + "="*50)
        print("TRAINING NEURAL NETWORKS")
        print("="*50)
        
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
            
            
            print("\n1. Multi-layer Perceptron (MLP)...")
            mlp = Sequential([
                Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            mlp.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='binary_crossentropy', 
                       metrics=['accuracy'])
            mlp.fit(self.X_train_scaled, self.y_train, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=0)
            
            y_pred_proba_mlp = mlp.predict(self.X_test_scaled, verbose=0)
            y_pred_mlp = (y_pred_proba_mlp > 0.5).astype(int).flatten()
            
            acc_mlp = accuracy_score(self.y_test, y_pred_mlp)
            precision_mlp = precision_score(self.y_test, y_pred_mlp)
            recall_mlp = recall_score(self.y_test, y_pred_mlp)
            f1_mlp = f1_score(self.y_test, y_pred_mlp)
            auc_mlp = roc_auc_score(self.y_test, y_pred_proba_mlp)
            
            self.results['MLP'] = {
                'Accuracy': acc_mlp, 'Precision': precision_mlp,
                'Recall': recall_mlp, 'F1_Score': f1_mlp, 'AUC': auc_mlp
            }
            self.models['MLP'] = mlp
            print(f"Accuracy: {acc_mlp:.4f}, Precision: {precision_mlp:.4f}, "
                  f"Recall: {recall_mlp:.4f}, F1: {f1_mlp:.4f}, AUC: {auc_mlp:.4f}")
            
           
            print("\n2. Convolutional Neural Network (1D CNN)...")
            X_train_cnn = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 
                                                      self.X_train_scaled.shape[1], 1)
            X_test_cnn = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 
                                                    self.X_test_scaled.shape[1], 1)
            
            cnn = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=(self.X_train_scaled.shape[1], 1)),
                Conv1D(64, 3, activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            cnn.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='binary_crossentropy', 
                       metrics=['accuracy'])
            cnn.fit(X_train_cnn, self.y_train, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=0)
            
            y_pred_proba_cnn = cnn.predict(X_test_cnn, verbose=0)
            y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int).flatten()
            
            acc_cnn = accuracy_score(self.y_test, y_pred_cnn)
            precision_cnn = precision_score(self.y_test, y_pred_cnn)
            recall_cnn = recall_score(self.y_test, y_pred_cnn)
            f1_cnn = f1_score(self.y_test, y_pred_cnn)
            auc_cnn = roc_auc_score(self.y_test, y_pred_proba_cnn)
            
            self.results['CNN'] = {
                'Accuracy': acc_cnn, 'Precision': precision_cnn,
                'Recall': recall_cnn, 'F1_Score': f1_cnn, 'AUC': auc_cnn
            }
            self.models['CNN'] = cnn
            print(f"Accuracy: {acc_cnn:.4f}, Precision: {precision_cnn:.4f}, "
                  f"Recall: {recall_cnn:.4f}, F1: {f1_cnn:.4f}, AUC: {auc_cnn:.4f}")
            
           
            print("\n3. Recurrent Neural Network (LSTM)...")
            X_train_lstm = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 
                                                       1, self.X_train_scaled.shape[1])
            X_test_lstm = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 
                                                    1, self.X_test_scaled.shape[1])
            
            lstm = Sequential([
                LSTM(50, activation='relu', input_shape=(1, self.X_train_scaled.shape[1])),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            lstm.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])
            lstm.fit(X_train_lstm, self.y_train, epochs=50, batch_size=32, 
                    validation_split=0.2, verbose=0)
            
            y_pred_proba_lstm = lstm.predict(X_test_lstm, verbose=0)
            y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int).flatten()
            
            acc_lstm = accuracy_score(self.y_test, y_pred_lstm)
            precision_lstm = precision_score(self.y_test, y_pred_lstm)
            recall_lstm = recall_score(self.y_test, y_pred_lstm)
            f1_lstm = f1_score(self.y_test, y_pred_lstm)
            auc_lstm = roc_auc_score(self.y_test, y_pred_proba_lstm)
            
            self.results['RNN_LSTM'] = {
                'Accuracy': acc_lstm, 'Precision': precision_lstm,
                'Recall': recall_lstm, 'F1_Score': f1_lstm, 'AUC': auc_lstm
            }
            self.models['RNN_LSTM'] = lstm
            print(f"Accuracy: {acc_lstm:.4f}, Precision: {precision_lstm:.4f}, "
                  f"Recall: {recall_lstm:.4f}, F1: {f1_lstm:.4f}, AUC: {auc_lstm:.4f}")
            
        except Exception as e:
            print(f"Error training neural networks: {e}")
    
    def cross_validation_evaluation(self):
        """Perform cross-validation evaluation"""
        print("\n" + "="*50)
        print("CROSS-VALIDATION EVALUATION")
        print("="*50)
        
     
        print("\nK-Fold Cross-Validation (k=5)...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_models = ['Logistic_Regression', 'Random_Forest', 'SVM', 'Decision_Tree']
        cv_results = {}
        
        for model_name in cv_models:
            if model_name in self.models:
                model = self.models[model_name]
                scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=kfold, scoring='accuracy')
                cv_results[model_name] = {
                    'Mean_Accuracy': scores.mean(),
                    'Std_Accuracy': scores.std()
                }
                print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        self.results['CV_Results'] = cv_results
        
       
        print("\nLeave-One-Out Cross-Validation (on 50 samples)...")
        loo = LeaveOneOut()
        X_loo = self.X_train_scaled[:50]
        y_loo = self.y_train[:50]
        
        loo_scores = []
        for train_idx, test_idx in loo.split(X_loo):
            X_train_loo, X_test_loo = X_loo[train_idx], X_loo[test_idx]
            y_train_loo, y_test_loo = y_loo[train_idx], y_loo[test_idx]
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_loo, y_train_loo)
            score = model.score(X_test_loo, y_test_loo)
            loo_scores.append(score)
        
        loo_mean = np.mean(loo_scores)
        self.results['LOO_CV'] = {'Mean_Accuracy': loo_mean}
        print(f"LOO CV Accuracy: {loo_mean:.4f}")
    
    def compare_models(self):
        """Compare all models and select the best ones"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
       
        classification_models = {}
        for model_name, results in self.results.items():
            if 'Accuracy' in results:
                classification_models[model_name] = results['Accuracy']
        
   
        sorted_models = sorted(classification_models.items(), key=lambda x: x[1], reverse=True)
        
        print("\nModel Rankings (by Accuracy):")
        print("-" * 50)
        for i, (model_name, accuracy) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}: {accuracy:.4f}")
        
       
        if len(sorted_models) >= 2:
            best_model_1 = sorted_models[0][0]
            best_model_2 = sorted_models[1][0]
            
            print(f"\nTop 2 Models Selected:")
            print(f"1. {best_model_1} (Accuracy: {sorted_models[0][1]:.4f})")
            print(f"2. {best_model_2} (Accuracy: {sorted_models[1][1]:.4f})")
            
            self.best_models = [best_model_1, best_model_2]
        else:
            self.best_models = [sorted_models[0][0]] if sorted_models else []
        
        return self.best_models
    
    def save_models(self):
        """Save trained models and preprocessors"""
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
       
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        
        for model_name, model in self.models.items():
            if model_name not in ['Polynomial_Features']:
                if 'MLP' in model_name or 'CNN' in model_name or 'LSTM' in model_name:
                    model.save(f'{model_name.lower()}.h5')
                else:
                    joblib.dump(model, f'{model_name.lower()}.pkl')
        
        
        if self.pca:
            joblib.dump(self.pca, 'pca.pkl')
        
        
        import json
        
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_json[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) 
                                    else v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in value.items()}
            else:
                results_json[key] = value
        
        with open('model_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print("Models and results saved successfully!")
    
    def train_all(self):
        """Train all models"""
        self.load_and_preprocess_data()
        self.calculate_correlations()
        self.train_regression_models()
        self.train_classification_models()
        self.train_clustering_models()
        self.apply_pca()
        self.train_neural_networks()
        self.cross_validation_evaluation()
        self.compare_models()
        self.save_models()
        
        return self

if __name__ == "__main__":
    trainer = ModelTrainer('train_u6lujuX_CVtuZ9i (1).csv')
    trainer.train_all()
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)

