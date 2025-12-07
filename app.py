"""
Streamlit Web Application for Loan Prediction Analytics
Comprehensive dashboard with visualizations and detailed predictions
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

# Custom CSS for dark theme dashboard
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #9467bd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
try:
    df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
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
    
    try:
        with open('model_results.json', 'r') as f:
            results = json.load(f)
    except:
        results = {}
        
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Main Header
st.markdown('<h1 class="main-header">🏦 Loan Prediction Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time insights into loan approval patterns and predictive analytics</p>', unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", 
                       ["📊 Dashboard", "🔮 Detailed Prediction", "📈 Model Performance", "📋 Data Analysis", "💻 Python Code"])

# ==================== DASHBOARD PAGE ====================
if page == "📊 Dashboard":
    
    # Key Metrics Cards
    total_loans = len(df)
    approved = len(df[df['Loan_Status'] == 'Y'])
    rejected = len(df[df['Loan_Status'] == 'N'])
    approval_rate = (approved / total_loans) * 100
    avg_income = df['ApplicantIncome'].mean()
    avg_loan = df['LoanAmount'].mean() if 'LoanAmount' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("TOTAL LOANS", f"{total_loans:,}", "All applications")
    
    with col2:
        st.metric("APPROVAL RATE", f"{approval_rate:.1f}%", f"{approved} approved")
    
    with col3:
        st.metric("AVG INCOME", f"₹{avg_income:,.0f}", "Applicant income")
    
    with col4:
        st.metric("AVG LOAN AMT", f"₹{avg_loan:,.0f}", "Loan amount")
    
    st.markdown("---")
    
    # Best Model Performance Section
    if results:
        classification_results = {}
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'Accuracy' in metrics:
                classification_results[model_name] = metrics['Accuracy']
        
        if classification_results:
            sorted_models = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
            best_model_name = sorted_models[0][0]
            best_model_accuracy = sorted_models[0][1]
            
            # Get additional metrics for best model
            best_model_metrics = results.get(best_model_name, {})
            
            st.subheader("🏆 Best Performing Model")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Model", best_model_name, "")
            
            with col2:
                st.metric("Accuracy", f"{best_model_accuracy:.2%}", "Best Performance")
            
            with col3:
                precision = best_model_metrics.get('Precision', 0)
                st.metric("Precision", f"{precision:.2%}" if precision > 0 else "N/A", "")
            
            with col4:
                recall = best_model_metrics.get('Recall', 0)
                st.metric("Recall", f"{recall:.2%}" if recall > 0 else "N/A", "")
            
            with col5:
                f1 = best_model_metrics.get('F1_Score', 0)
                st.metric("F1 Score", f"{f1:.2%}" if f1 > 0 else "N/A", "")
            
            # Model Rankings Chart
            st.markdown("**📊 All Models Performance Ranking**")
            ranking_df = pd.DataFrame(sorted_models, columns=['Model', 'Accuracy'])
            ranking_df['Rank'] = range(1, len(ranking_df) + 1)
            ranking_df['Accuracy_Percent'] = ranking_df['Accuracy'] * 100
            
            fig_ranking = px.bar(
                ranking_df,
                x='Accuracy_Percent',
                y='Model',
                orientation='h',
                color='Accuracy_Percent',
                color_continuous_scale='RdYlGn',
                text=ranking_df['Accuracy_Percent'].round(2),
                labels={'Accuracy_Percent': 'Accuracy (%)', 'Model': 'Model'},
                title="Model Performance Ranking (Best to Worst)"
            )
            fig_ranking.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_ranking.update_layout(height=max(400, len(ranking_df) * 50), showlegend=False)
            fig_ranking.add_vline(x=best_model_accuracy*100, line_dash="dash", line_color="red", 
                                 annotation_text=f"Best: {best_model_accuracy:.2%}")
            st.plotly_chart(fig_ranking, use_container_width=True)
            
            st.markdown("---")
    
    # Row 1: Loan Status Distribution and Approval Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Loan Status Distribution")
        status_counts = df['Loan_Status'].value_counts()
        fig_pie = px.pie(
            values=status_counts.values, 
            names=['Approved' if x == 'Y' else 'Rejected' for x in status_counts.index],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Approval Rate by Category")
        category = st.selectbox("Select Category", ["Gender", "Married", "Education", "Property_Area", "Self_Employed"], key="cat1")
        
        category_data = df.groupby([category, 'Loan_Status']).size().unstack(fill_value=0)
        category_data['Approval_Rate'] = (category_data.get('Y', 0) / (category_data.get('Y', 0) + category_data.get('N', 0))) * 100
        
        fig_bar = px.bar(
            x=category_data.index,
            y=category_data['Approval_Rate'],
            labels={'x': category, 'y': 'Approval Rate (%)'},
            color=category_data['Approval_Rate'],
            color_continuous_scale='Viridis',
            text=category_data['Approval_Rate'].round(1)
        )
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_bar.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Income Analysis and Loan Amount Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Income Distribution by Loan Status")
        fig_income = px.box(
            df,
            x='Loan_Status',
            y='ApplicantIncome',
            color='Loan_Status',
            labels={'Loan_Status': 'Loan Status', 'ApplicantIncome': 'Applicant Income (₹)'},
            color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'}
        )
        fig_income.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_income, use_container_width=True)
    
    with col2:
        st.subheader("💵 Loan Amount vs Income")
        df_clean = df.dropna(subset=['LoanAmount', 'ApplicantIncome'])
        fig_scatter = px.scatter(
            df_clean,
            x='ApplicantIncome',
            y='LoanAmount',
            color='Loan_Status',
            size='LoanAmount',
            hover_data=['Gender', 'Education', 'Property_Area'],
            labels={'ApplicantIncome': 'Applicant Income (₹)', 'LoanAmount': 'Loan Amount (₹)'},
            color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'},
            title="Relationship between Income and Loan Amount"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Regional Analysis and Credit History Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Property Area Analysis")
        area_data = df.groupby(['Property_Area', 'Loan_Status']).size().unstack(fill_value=0)
        area_data['Total'] = area_data.sum(axis=1)
        area_data['Approval_Rate'] = (area_data.get('Y', 0) / area_data['Total']) * 100
        
        fig_area = px.bar(
            x=area_data.index,
            y=area_data['Approval_Rate'],
            labels={'x': 'Property Area', 'y': 'Approval Rate (%)'},
            color=area_data['Approval_Rate'],
            color_continuous_scale='Plasma',
            text=area_data['Approval_Rate'].round(1)
        )
        fig_area.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_area.update_layout(height=400)
        st.plotly_chart(fig_area, use_container_width=True)
    
    with col2:
        st.subheader("✅ Credit History Impact")
        credit_data = df.groupby(['Credit_History', 'Loan_Status']).size().unstack(fill_value=0)
        credit_data.index = ['No Credit History' if x == 0 else 'Has Credit History' for x in credit_data.index]
        
        fig_credit = go.Figure()
        fig_credit.add_trace(go.Bar(
            x=credit_data.index,
            y=credit_data.get('Y', 0),
            name='Approved',
            marker_color='#2ecc71'
        ))
        fig_credit.add_trace(go.Bar(
            x=credit_data.index,
            y=credit_data.get('N', 0),
            name='Rejected',
            marker_color='#e74c3c'
        ))
        fig_credit.update_layout(
            barmode='group',
            height=400,
            xaxis_title="Credit History",
            yaxis_title="Number of Loans",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_credit, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Feature Correlation Heatmap
    st.subheader("🔗 Feature Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Loan_Status' in df.columns:
        df_corr = df.copy()
        df_corr['Loan_Status'] = df_corr['Loan_Status'].map({'Y': 1, 'N': 0})
        numeric_cols = [col for col in numeric_cols if col != 'Loan_ID'] if 'Loan_ID' in numeric_cols else numeric_cols
        numeric_cols.append('Loan_Status')
        corr_matrix = df_corr[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu',
            aspect="auto",
            text_auto=True
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ==================== DETAILED PREDICTION PAGE ====================
elif page == "🔮 Detailed Prediction":
    
    st.header("🔮 Detailed Loan Prediction Analysis")
    
    # Get best models
    classification_results = {}
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'Accuracy' in metrics:
            classification_results[model_name] = metrics['Accuracy']
    
    sorted_models = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
    best_models = [m[0] for m in sorted_models[:2]] if len(sorted_models) >= 2 else [sorted_models[0][0]] if sorted_models else []
    
    if best_models:
        # Enhanced Best Model Display
        best_model_name = sorted_models[0][0]
        best_model_acc = sorted_models[0][1]
        second_model_name = sorted_models[1][0] if len(sorted_models) > 1 else None
        second_model_acc = sorted_models[1][1] if len(sorted_models) > 1 else None
        
        # Get best model metrics
        best_model_metrics = results.get(best_model_name, {})
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>🏆 Best Performing Model</h2>
            <h1 style='color: white; margin: 0.5rem 0;'>{}</h1>
            <p style='color: white; font-size: 1.2rem; margin: 0;'>Accuracy: {:.2%}</p>
        </div>
        """.format(best_model_name, best_model_acc), unsafe_allow_html=True)
        
        # Show detailed metrics for best model
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precision", f"{best_model_metrics.get('Precision', 0):.2%}" if best_model_metrics.get('Precision', 0) > 0 else "N/A")
        with col2:
            st.metric("Recall", f"{best_model_metrics.get('Recall', 0):.2%}" if best_model_metrics.get('Recall', 0) > 0 else "N/A")
        with col3:
            st.metric("F1 Score", f"{best_model_metrics.get('F1_Score', 0):.2%}" if best_model_metrics.get('F1_Score', 0) > 0 else "N/A")
        with col4:
            st.metric("AUC", f"{best_model_metrics.get('AUC', 0):.4f}" if best_model_metrics.get('AUC', 0) > 0 else "N/A")
        
        if second_model_name:
            st.info(f"🥈 **Second Best Model:** {second_model_name} (Accuracy: {second_model_acc:.2%})")
        
        st.markdown("---")
    
    # Input Form
    st.subheader("📝 Enter Applicant Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        married = st.selectbox("Married", ["Yes", "No"], key="pred_married")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3"], key="pred_dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="pred_education")
    
    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"], key="pred_self")
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=100, key="pred_income")
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=100, key="pred_coincome")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000, step=1000, key="pred_loan")
    
    with col3:
        loan_amount_term = st.number_input("Loan Term (days)", min_value=0, value=360, step=30, key="pred_term")
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="pred_credit")
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"], key="pred_area")
    
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
        df_input = pd.DataFrame([input_data])
        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            if col in df_input.columns and col in label_encoders:
                df_input[col] = label_encoders[col].transform([df_input[col].iloc[0]])[0]
        df_input = df_input[feature_names]
        X_scaled = scaler.transform(df_input.values)
        
        # Make predictions with all available models
        all_predictions = {}
        all_probabilities = {}
        
        for model_name in models.keys():
            try:
                model = models[model_name]
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else pred
                all_predictions[model_name] = pred
                all_probabilities[model_name] = proba
            except:
                pass
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Display predictions in cards
        cols = st.columns(min(len(all_predictions), 4))
        for idx, (model_name, pred) in enumerate(all_predictions.items()):
            with cols[idx % 4]:
                proba = all_probabilities[model_name]
                status = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
                color = "#2ecc71" if pred == 1 else "#e74c3c"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}15 0%, {color}25 100%); 
                            padding: 1rem; border-radius: 10px; border-left: 4px solid {color};'>
                    <h4>{model_name}</h4>
                    <h2 style='color: {color}; margin: 0.5rem 0;'>{status}</h2>
                    <p style='font-size: 1.2rem; font-weight: bold;'>Confidence: {proba*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Analysis Section
        st.subheader("📈 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction Confidence Chart
            st.markdown("**Model Confidence Levels**")
            pred_df = pd.DataFrame({
                'Model': list(all_probabilities.keys()),
                'Confidence': [p*100 for p in all_probabilities.values()]
            })
            pred_df = pred_df.sort_values('Confidence', ascending=True)
            
            fig_conf = px.bar(
                pred_df,
                x='Confidence',
                y='Model',
                orientation='h',
                color='Confidence',
                color_continuous_scale='RdYlGn',
                text=pred_df['Confidence'].round(1),
                labels={'Confidence': 'Confidence (%)', 'Model': 'Model'}
            )
            fig_conf.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_conf.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Feature Comparison with Dataset
            st.markdown("**Your Profile vs Dataset Average**")
            
            try:
                # Calculate dataset averages safely
                avg_app_income = df['ApplicantIncome'].mean() if 'ApplicantIncome' in df.columns else 0
                avg_coapp_income = df['CoapplicantIncome'].mean() if 'CoapplicantIncome' in df.columns else 0
                avg_loan_amt = df['LoanAmount'].mean() if 'LoanAmount' in df.columns and not df['LoanAmount'].isna().all() else 0
                avg_loan_term = df['Loan_Amount_Term'].mean() if 'Loan_Amount_Term' in df.columns and not df['Loan_Amount_Term'].isna().all() else 0
                
                comparison_data = {
                    'Metric': ['Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Term'],
                    'Your Value': [
                        float(applicant_income),
                        float(coapplicant_income),
                        float(loan_amount),
                        float(loan_amount_term)
                    ],
                    'Dataset Average': [
                        float(avg_app_income),
                        float(avg_coapp_income),
                        float(avg_loan_amt),
                        float(avg_loan_term)
                    ]
                }
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Calculate difference safely (avoid division by zero)
                comp_df['Difference (%)'] = comp_df.apply(
                    lambda row: ((row['Your Value'] - row['Dataset Average']) / row['Dataset Average'] * 100) 
                    if row['Dataset Average'] != 0 else 0, axis=1
                ).round(1)
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='Your Value',
                    x=comp_df['Metric'],
                    y=comp_df['Your Value'],
                    marker_color='#3498db',
                    text=comp_df['Your Value'].round(0),
                    textposition='outside'
                ))
                fig_comp.add_trace(go.Bar(
                    name='Dataset Average',
                    x=comp_df['Metric'],
                    y=comp_df['Dataset Average'],
                    marker_color='#95a5a6',
                    text=comp_df['Dataset Average'].round(0),
                    textposition='outside'
                ))
                fig_comp.update_layout(
                    barmode='group',
                    height=400,
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Show difference table
                st.dataframe(comp_df[['Metric', 'Your Value', 'Dataset Average', 'Difference (%)']], use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error in comparison: {e}")
        
        # Similar Cases Analysis
        st.markdown("---")
        st.subheader("🔍 Similar Cases in Dataset")
        
        try:
            # Find similar cases
            df_similar = df.copy()
            df_similar['Score'] = 0
            
            # Add score for matching categorical features
            if 'Gender' in df_similar.columns and gender in df_similar['Gender'].values:
                df_similar.loc[df_similar['Gender'] == gender, 'Score'] += 1
            if 'Married' in df_similar.columns and married in df_similar['Married'].values:
                df_similar.loc[df_similar['Married'] == married, 'Score'] += 1
            if 'Education' in df_similar.columns and education in df_similar['Education'].values:
                df_similar.loc[df_similar['Education'] == education, 'Score'] += 1
            if 'Property_Area' in df_similar.columns and property_area in df_similar['Property_Area'].values:
                df_similar.loc[df_similar['Property_Area'] == property_area, 'Score'] += 1
            if 'Credit_History' in df_similar.columns:
                df_similar.loc[df_similar['Credit_History'] == credit_history, 'Score'] += 1
            
            # Filter by income range (±20%)
            if 'ApplicantIncome' in df_similar.columns:
                income_range = df_similar[
                    (df_similar['ApplicantIncome'] >= applicant_income * 0.8) &
                    (df_similar['ApplicantIncome'] <= applicant_income * 1.2)
                ]
            else:
                income_range = df_similar
            
            if len(income_range) > 0:
                similar_cases = income_range.nlargest(10, 'Score')
                
                # Select available columns
                display_cols = []
                for col in ['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Loan_Status']:
                    if col in similar_cases.columns:
                        display_cols.append(col)
                
                if display_cols:
                    similar_display = similar_cases[display_cols].copy()
                    
                    # Map Loan_Status if it exists
                    if 'Loan_Status' in similar_display.columns:
                        similar_display['Loan_Status'] = similar_display['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected'})
                    
                    st.dataframe(similar_display, use_container_width=True)
                    
                    if 'Loan_Status' in similar_display.columns:
                        similar_approved = len(similar_display[similar_display['Loan_Status'] == 'Approved'])
                        similar_rate = (similar_approved / len(similar_display)) * 100 if len(similar_display) > 0 else 0
                        st.info(f"📊 **Similar Cases Analysis:** {similar_approved}/{len(similar_display)} ({similar_rate:.1f}%) were approved in similar profiles")
                else:
                    st.warning("No similar cases found with matching criteria.")
            else:
                st.warning("No similar cases found in the dataset.")
                
        except Exception as e:
            st.error(f"Error in similar cases analysis: {e}")
            st.info("Please check that the dataset contains the required columns.")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📈 Model Performance":
    
    st.header("📈 Model Performance Metrics")
    
    if not results:
        st.warning("No model results available.")
    else:
        # Extract classification models
        classification_models = [name for name, metrics in results.items() 
                                if isinstance(metrics, dict) and 'Accuracy' in metrics]
        
        if classification_models:
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
                
                # Confusion Matrix
                if 'Confusion_Matrix' in metrics:
                    st.subheader("Confusion Matrix")
                    cm = np.array(metrics['Confusion_Matrix'])
                    
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Rejected", "Approved"],
                        y=["Rejected", "Approved"],
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

# ==================== DATA ANALYSIS PAGE ====================
elif page == "📋 Data Analysis":
    
    st.header("📋 Comprehensive Data Analysis")
    
    # Dataset Overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Features", f"{len(df.columns) - 1}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    
    st.markdown("---")
    
    # Dataset Preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Missing Values Analysis
    st.subheader("🔍 Missing Values Analysis")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': (missing.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) > 0:
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Percentage',
            labels={'Column': 'Feature', 'Percentage': 'Missing %'},
            color='Percentage',
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values in the dataset!")
    
    st.markdown("---")
    
    # Categorical Features Distribution
    st.subheader("📊 Categorical Features Distribution")
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    selected_cat = st.selectbox("Select Feature", categorical_cols)
    
    cat_counts = df[selected_cat].value_counts()
    fig_cat = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        labels={'x': selected_cat, 'y': 'Count'},
        color=cat_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

# ==================== PYTHON CODE PAGE ====================
elif page == "💻 Python Code":
    
    st.header("💻 Python Code & Models Used")
    
    # Models Used Section
    st.subheader("🤖 Machine Learning Models Implemented")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Regression Models:**
        - Simple Linear Regression
        - Multiple Linear Regression
        - Polynomial Regression
        - Logistic Regression
        """)
    
    with col2:
        st.markdown("""
        **Classification Models:**
        - Naive Bayes
        - Decision Trees
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Random Forest
        - Bagging
        - AdaBoost
        """)
    
    with col3:
        st.markdown("""
        **Advanced Models:**
        - K-Means Clustering
        - Hierarchical Clustering
        - Principal Component Analysis (PCA)
        - Multi-layer Perceptron (MLP)
        - Convolutional Neural Network (CNN)
        - Recurrent Neural Network (LSTM)
        """)
    
    st.markdown("---")
    
    # Code Files Section
    st.subheader("📁 Python Files in Project")
    
    code_files = {
        "model_training.py": "Main ML training script with all algorithms",
        "app.py": "Streamlit web application",
        "run_training.py": "Quick training script runner",
        "test_imports.py": "Import verification script"
    }
    
    selected_file = st.selectbox("Select Python File to View", list(code_files.keys()))
    st.info(f"**Description:** {code_files[selected_file]}")
    
    st.markdown("---")
    
    # Display selected code file
    st.subheader(f"📄 {selected_file}")
    
    try:
        if selected_file == "model_training.py":
            with open('model_training.py', 'r', encoding='utf-8') as f:
                code_content = f.read()
            st.code(code_content, language='python')
        
        elif selected_file == "app.py":
            st.info("This is the current file. Showing structure only.")
            st.markdown("""
            ```python
            # Main Streamlit Application
            # Contains:
            # - Dashboard with visualizations
            # - Detailed prediction interface
            # - Model performance metrics
            # - Data analysis tools
            # - Python code viewer (this section)
            ```
            """)
            st.info("💡 The full app.py code is the file you're currently viewing!")
        
        elif selected_file == "run_training.py":
            with open('run_training.py', 'r', encoding='utf-8') as f:
                code_content = f.read()
            st.code(code_content, language='python')
        
        elif selected_file == "test_imports.py":
            with open('test_imports.py', 'r', encoding='utf-8') as f:
                code_content = f.read()
            st.code(code_content, language='python')
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
    
    st.markdown("---")
    
    # Libraries and Dependencies
    st.subheader("📚 Python Libraries Used")
    
    libraries = {
        "Data Processing": ["pandas", "numpy"],
        "Machine Learning": ["scikit-learn", "tensorflow", "keras"],
        "Visualization": ["plotly", "matplotlib", "seaborn"],
        "Web Framework": ["streamlit"],
        "Utilities": ["joblib", "scipy", "json"]
    }
    
    for category, libs in libraries.items():
        with st.expander(f"📦 {category}"):
            for lib in libs:
                st.write(f"- `{lib}`")
    
    st.markdown("---")
    
    # Key Code Snippets
    st.subheader("🔑 Key Code Snippets")
    
    snippet_tabs = st.tabs(["Data Preprocessing", "Model Training", "Model Evaluation", "Prediction"])
    
    with snippet_tabs[0]:
        st.code("""
# Data Preprocessing Example
def load_and_preprocess_data(self):
    df = pd.read_csv(self.data_path)
    df = df.drop('Loan_ID', axis=1)
    
    # Handle missing values
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale features
    self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        """, language='python')
    
    with snippet_tabs[1]:
        st.code("""
# Model Training Example
# Logistic Regression
logistic_reg = LogisticRegression(max_iter=1000, random_state=42)
logistic_reg.fit(self.X_train_scaled, self.y_train)
y_pred = logistic_reg.predict(self.X_test_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(self.X_train_scaled, self.y_train)
y_pred = rf.predict(self.X_test_scaled)

# Neural Network (MLP)
mlp = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer=Adam(), loss='binary_crossentropy')
mlp.fit(self.X_train_scaled, self.y_train, epochs=50)
        """, language='python')
    
    with snippet_tabs[2]:
        st.code("""
# Model Evaluation Example
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(self.y_test, y_pred)
precision = precision_score(self.y_test, y_pred)
recall = recall_score(self.y_test, y_pred)
f1 = f1_score(self.y_test, y_pred)
auc = roc_auc_score(self.y_test, y_pred_proba)

# Cross-Validation
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        """, language='python')
    
    with snippet_tabs[3]:
        st.code("""
# Prediction Example
# Preprocess input
df_input = pd.DataFrame([input_data])
for col in categorical_cols:
    df_input[col] = label_encoders[col].transform([df_input[col].iloc[0]])[0]

X_scaled = scaler.transform(df_input.values)

# Make prediction
prediction = model.predict(X_scaled)[0]
probability = model.predict_proba(X_scaled)[0][1]

# Result
status = "Approved" if prediction == 1 else "Rejected"
        """, language='python')
    
    st.markdown("---")
    
    # Download Code Button
    st.subheader("💾 Download Code")
    st.info("All code files are available in the project directory. You can also view them on GitHub.")
    
    if st.button("📥 View on GitHub", use_container_width=True):
        st.markdown("[🔗 GitHub Repository](https://github.com/Amit9031/Loan-Predictive-Analytics)")

if __name__ == "__main__":
    pass
