"""
Streamlit Web Application for Loan Prediction
Interactive interface to use trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Prediction Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Load classification models
        models = {}
        model_files = {
            'Logistic_Regression': 'logistic_regression.pkl',
            'Random_Forest': 'random_forest.pkl',
            'SVM': 'svm.pkl',
            'Decision_Tree': 'decision_tree.pkl',
            'Naive_Bayes': 'naive_bayes.pkl',
            'KNN': 'knn.pkl',
            'Bagging': 'bagging.pkl',
            'AdaBoost': 'adaboost.pkl'
        }
        
        for model_name, file_path in model_files.items():
            try:
                models[model_name] = joblib.load(file_path)
            except:
                pass
        
        # Load neural network models
        try:
            from tensorflow import keras
            models['MLP'] = keras.models.load_model('mlp.h5')
            models['CNN'] = keras.models.load_model('cnn.h5')
            models['RNN_LSTM'] = keras.models.load_model('rnn_lstm.h5')
        except:
            pass
        
        # Load results
        try:
            with open('model_results.json', 'r') as f:
                results = json.load(f)
        except:
            results = {}
        
        return scaler, label_encoders, feature_names, models, results
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run model_training.py first to train the models.")
        return None, None, None, {}, {}

def preprocess_input(data, label_encoders, feature_names):
    """Preprocess user input for prediction"""
    # Create a DataFrame with the input data
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform([df[col].iloc[0]])[0]
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    return df.values

def make_prediction(model, X_scaled, model_name):
    """Make prediction using the model"""
    try:
        if 'MLP' in model_name or 'CNN' in model_name or 'LSTM' in model_name:
            # Neural network models
            if 'CNN' in model_name:
                X_scaled = X_scaled.reshape(1, X_scaled.shape[0], 1)
            elif 'LSTM' in model_name:
                X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[0])
            
            proba = model.predict(X_scaled, verbose=0)[0][0]
            prediction = 1 if proba > 0.5 else 0
            return prediction, proba
        else:
            # Traditional ML models
            prediction = model.predict(X_scaled)[0]
            try:
                proba = model.predict_proba(X_scaled)[0][1]
            except:
                proba = prediction
            return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🏦 Loan Prediction Analytics</h1>', unsafe_allow_html=True)
    
    # Load models
    scaler, label_encoders, feature_names, models, results = load_models()
    
    if scaler is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", 
                           ["📊 Model Comparison", "🔮 Make Prediction", "📈 Model Performance", "📋 Dataset Info"])
    
    if page == "📊 Model Comparison":
        show_model_comparison(results)
    
    elif page == "🔮 Make Prediction":
        show_prediction_page(scaler, label_encoders, feature_names, models, results)
    
    elif page == "📈 Model Performance":
        show_performance_metrics(results)
    
    elif page == "📋 Dataset Info":
        show_dataset_info()

def show_model_comparison(results):
    """Display model comparison"""
    st.header("📊 Model Comparison & Rankings")
    
    if not results:
        st.warning("No model results available. Please train models first.")
        return
    
    # Extract classification models with accuracy
    classification_results = {}
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'Accuracy' in metrics:
            classification_results[model_name] = metrics['Accuracy']
    
    if not classification_results:
        st.warning("No classification model results found.")
        return
    
    # Sort by accuracy
    sorted_models = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
    
    # Display rankings
    st.subheader("Model Rankings (by Accuracy)")
    
    # Create DataFrame for better visualization
    df_rankings = pd.DataFrame(sorted_models, columns=['Model', 'Accuracy'])
    df_rankings['Rank'] = range(1, len(df_rankings) + 1)
    df_rankings = df_rankings[['Rank', 'Model', 'Accuracy']]
    df_rankings['Accuracy'] = df_rankings['Accuracy'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(df_rankings, use_container_width=True)
    
    # Visualize with bar chart
    fig = px.bar(
        x=[m[0] for m in sorted_models],
        y=[m[1] for m in sorted_models],
        labels={'x': 'Model', 'y': 'Accuracy'},
        title="Model Accuracy Comparison",
        color=[m[1] for m in sorted_models],
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top 2 models
    if len(sorted_models) >= 2:
        st.success(f"🏆 **Best Model:** {sorted_models[0][0]} (Accuracy: {sorted_models[0][1]:.4f})")
        st.info(f"🥈 **Second Best:** {sorted_models[1][0]} (Accuracy: {sorted_models[1][1]:.4f})")

def show_prediction_page(scaler, label_encoders, feature_names, models, results):
    """Show prediction interface"""
    st.header("🔮 Loan Prediction")
    
    # Get best models
    classification_results = {}
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'Accuracy' in metrics:
            classification_results[model_name] = metrics['Accuracy']
    
    sorted_models = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
    best_models = [m[0] for m in sorted_models[:2]] if len(sorted_models) >= 2 else [sorted_models[0][0]] if sorted_models else []
    
    if not best_models:
        st.warning("No trained models available. Please train models first.")
        return
    
    st.info(f"Using top 2 models: **{best_models[0]}** and **{best_models[1]}**")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100, step=10)
        loan_amount_term = st.number_input("Loan Amount Term (days)", min_value=0, value=360, step=30)
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    if st.button("🔮 Predict Loan Status", type="primary", use_container_width=True):
        # Preprocess input
        X = preprocess_input(input_data, label_encoders, feature_names)
        X_scaled = scaler.transform(X)
        
        # Make predictions with best models
        predictions = {}
        probabilities = {}
        
        for model_name in best_models:
            if model_name in models:
                pred, proba = make_prediction(models[model_name], X_scaled, model_name)
                predictions[model_name] = pred
                probabilities[model_name] = proba
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        for i, model_name in enumerate(best_models):
            if model_name in predictions:
                pred = predictions[model_name]
                proba = probabilities[model_name]
                
                with col1 if i == 0 else col2:
                    status = "✅ Approved" if pred == 1 else "❌ Rejected"
                    color = "green" if pred == 1 else "red"
                    
                    st.markdown(f"### {model_name}")
                    st.markdown(f"<h2 style='color: {color};'>{status}</h2>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{proba*100:.2f}%")
                    
                    # Progress bar for probability
                    st.progress(float(proba))
        
        # Show feature importance if available
        if 'Random_Forest' in best_models and 'Random_Forest' in models:
            st.subheader("Feature Importance (Random Forest)")
            rf_model = models['Random_Forest']
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

def show_performance_metrics(results):
    """Show detailed performance metrics"""
    st.header("📈 Model Performance Metrics")
    
    if not results:
        st.warning("No model results available.")
        return
    
    # Select model to view
    classification_models = [name for name, metrics in results.items() 
                            if isinstance(metrics, dict) and 'Accuracy' in metrics]
    
    if not classification_models:
        st.warning("No classification models found.")
        return
    
    selected_model = st.selectbox("Select Model", classification_models)
    
    if selected_model in results:
        metrics = results[selected_model]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if 'Accuracy' in metrics:
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        if 'Precision' in metrics:
            col2.metric("Precision", f"{metrics['Precision']:.4f}")
        if 'Recall' in metrics:
            col3.metric("Recall", f"{metrics['Recall']:.4f}")
        if 'F1_Score' in metrics:
            col4.metric("F1 Score", f"{metrics['F1_Score']:.4f}")
        
        # Additional metrics
        if 'AUC' in metrics:
            st.metric("Area Under Curve (AUC)", f"{metrics['AUC']:.4f}")
        if 'Log_Loss' in metrics:
            st.metric("Logarithmic Loss", f"{metrics['Log_Loss']:.4f}")
        
        # Confusion Matrix
        if 'Confusion_Matrix' in metrics:
            st.subheader("Confusion Matrix")
            cm = np.array(metrics['Confusion_Matrix'])
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=["Rejected", "Approved"],
                y=["Rejected", "Approved"],
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regression metrics if available
        if 'MAE' in metrics:
            st.subheader("Regression Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{metrics['MAE']:.4f}")
            col2.metric("MSE", f"{metrics['MSE']:.4f}")
            col3.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col4.metric("R² Score", f"{metrics['R2']:.4f}")

def show_dataset_info():
    """Show dataset information"""
    st.header("📋 Dataset Information")
    
    try:
        df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
        
        st.subheader("Dataset Overview")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Total Features:** {len(df.columns) - 1}")  # Excluding target
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        st.subheader("Target Variable Distribution")
        if 'Loan_Status' in df.columns:
            status_counts = df['Loan_Status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                        title="Loan Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()



