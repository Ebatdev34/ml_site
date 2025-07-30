import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer
from utils.task_detector import TaskDetector
import io

st.set_page_config(
    page_title="ML Universe",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    st.markdown("<h1 style='text-align:center; color:#0ff0fc; text-shadow: 0 0 8px #0ff0fc;'>🌌 ML Universe</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>**Democratizing Machine Learning for Everyone**</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a step:", [
        "📁 Data Upload",
        "🔍 Data Exploration", 
        "🤖 Model Selection & Training",
        "📊 Results & Analysis",
        "💾 Export Results"
    ])
    
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔍 Data Exploration":
        data_exploration_page()
    elif page == "🤖 Model Selection & Training":
        model_training_page()
    elif page == "📊 Results & Analysis":
        results_page()
    elif page == "💾 Export Results":
        export_page()

def data_upload_page():
    st.header("📁 Data Upload")
    st.markdown("Upload your CSV dataset to get started with machine learning!")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with your dataset. Make sure it includes column headers."
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"✅ Dataset uploaded successfully! Shape: {data.shape}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Show data types
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Target column selection
            st.subheader("Target Column Selection")
            target_col = st.selectbox(
                "Select the target column (what you want to predict):",
                options=data.columns.tolist(),
                help="This is the column that contains the values you want to predict."
            )
            
            if target_col:
                detector = TaskDetector()
                task_type = detector.detect_task_type(data[target_col])
                st.session_state.task_type = task_type
                
                st.info(f"🎯 Task detected: **{task_type.title()}**")
                
                if task_type == 'classification':
                    unique_classes = data[target_col].nunique()
                    st.write(f"Number of classes: {unique_classes}")
                    st.write("Class distribution:")
                    st.bar_chart(data[target_col].value_counts())
                else:
                    st.write("Target statistics:")
                    st.write(data[target_col].describe())
                
                if st.button("🔄 Process Data"):
                    with st.spinner("Processing data..."):
                        processor = DataProcessor()
                        processed_data = processor.preprocess_data(data, target_col, task_type)
                        st.session_state.processed_data = processed_data
                        st.success("✅ Data processed successfully!")
                        st.info("Navigate to 'Data Exploration' to see the processed data.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV with proper formatting.")

def data_exploration_page():
    st.header("🔍 Data Exploration")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data first in the Data Upload section!")
        return
    
    data = st.session_state.data
    processed_data = st.session_state.processed_data
    
    st.subheader("Original vs Processed Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Data**")
        st.dataframe(data.head(), use_container_width=True)
    
    with col2:
        st.write("**Processed Data**")
        st.dataframe(processed_data['X'].head(), use_container_width=True)
    
    st.subheader("Preprocessing Steps Applied")
    preprocessing_info = processed_data.get('preprocessing_info', {})
    
    if preprocessing_info:
        if 'encoded_columns' in preprocessing_info:
            st.write("**Categorical Encoding:**")
            for col in preprocessing_info['encoded_columns']:
                st.write(f"- {col}: Label encoded")
        
        if 'scaled_columns' in preprocessing_info:
            st.write("**Feature Scaling:**")
            for col in preprocessing_info['scaled_columns']:
                st.write(f"- {col}: Standard scaled")
    
    st.subheader("Feature Correlations")
    numeric_data = processed_data['X'].select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        visualizer = Visualizer()
        fig = visualizer.plot_correlation_matrix(numeric_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numerical features for correlation analysis.")

def model_training_page():
    st.header("🤖 Model Selection & Training")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process your data first!")
        return
    
    task_type = st.session_state.task_type
    
    st.subheader(f"Available Models for {task_type.title()}")
    
    if task_type == 'classification':
        available_models = {
            'Logistic Regression': 'logistic_regression',
            'Decision Tree': 'decision_tree',
            'Random Forest': 'random_forest'
        }
    else:
        available_models = {
            'Linear Regression': 'linear_regression',
            'Decision Tree': 'decision_tree_regressor',
            'Random Forest': 'random_forest_regressor'
        }
    
    selected_models = st.multiselect(
        "Select models to train:",
        options=list(available_models.keys()),
        default=list(available_models.keys())[:2],
        help="You can select multiple models to compare their performance."
    )
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model!")
        return
    
    with st.expander("⚙️ Advanced Options"):
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, help="Set for reproducible results")
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    
    if st.button("🚀 Train Models"):
        trainer = ModelTrainer()
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            model_key = available_models[model_name]
            status_text.text(f"Training {model_name}...")
            
            try:
                result = trainer.train_model(
                    st.session_state.processed_data,
                    model_key,
                    task_type,
                    test_size=test_size,
                    random_state=random_state,
                    cv_folds=cv_folds
                )
                results[model_name] = result
                progress_bar.progress((i + 1) / len(selected_models))
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
        
        status_text.text("Training completed!")
        st.session_state.results = results
        
        if results:
            st.success(f"✅ Successfully trained {len(results)} models!")
            st.info("Navigate to 'Results & Analysis' to see detailed results.")

def results_page():
    st.header("📊 Results & Analysis")
    
    if not st.session_state.results:
        st.warning("⚠️ Please train some models first!")
        return
    
    results = st.session_state.results
    task_type = st.session_state.task_type
    
    st.subheader("Model Comparison")
    
    comparison_data = []
    for model_name, result in results.items():
        if task_type == 'classification':
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['test_accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'CV Score': result['cv_score_mean']
            })
        else:
            comparison_data.append({
                'Model': model_name,
                'R²': result['test_r2'],
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'CV Score': result['cv_score_mean']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    if task_type == 'classification':
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_score = comparison_df['Accuracy'].max()
        st.success(f"🏆 Best performing model: **{best_model}** (Accuracy: {best_score:.4f})")
    else:
        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        best_score = comparison_df['R²'].max()
        st.success(f"🏆 Best performing model: **{best_model}** (R²: {best_score:.4f})")
    
    for model_name, result in results.items():
        with st.expander(f"📈 Detailed Results: {model_name}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Performance Metrics:**")
                if task_type == 'classification':
                    st.write(f"- Accuracy: {result['test_accuracy']:.4f}")
                    st.write(f"- Precision: {result['precision']:.4f}")
                    st.write(f"- Recall: {result['recall']:.4f}")
                    st.write(f"- F1-Score: {result['f1_score']:.4f}")
                else:
                    st.write(f"- R² Score: {result['test_r2']:.4f}")
                    st.write(f"- Mean Absolute Error: {result['mae']:.4f}")
                    st.write(f"- Mean Squared Error: {result['mse']:.4f}")
                    st.write(f"- Root Mean Squared Error: {result['rmse']:.4f}")
                
                st.write(f"- Cross-validation Score: {result['cv_score_mean']:.4f} (±{result['cv_score_std']:.4f})")
            
            with col2:
                visualizer = Visualizer()
                
                if task_type == 'classification' and 'confusion_matrix' in result:
                    st.write("**Confusion Matrix:**")
                    fig = visualizer.plot_confusion_matrix(result['confusion_matrix'])
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'feature_importance' in result:
                    st.write("**Feature Importance:**")
                    fig = visualizer.plot_feature_importance(
                        result['feature_importance'], 
                        st.session_state.processed_data['feature_names']
                    )
                    st.plotly_chart(fig, use_container_width=True)

def export_page():
    st.header("💾 Export Results")
    
    if not st.session_state.results:
        st.warning("⚠️ No results to export. Please train some models first!")
        return
    
    st.subheader("Export Options")
    
    results = st.session_state.results
    task_type = st.session_state.task_type
    
    export_data = []
    for model_name, result in results.items():
        export_row = {'Model': model_name}
        if task_type == 'classification':
            export_row.update({
                'Accuracy': result['test_accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'CV_Score_Mean': result['cv_score_mean'],
                'CV_Score_Std': result['cv_score_std']
            })
        else:
            export_row.update({
                'R2_Score': result['test_r2'],
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'CV_Score_Mean': result['cv_score_mean'],
                'CV_Score_Std': result['cv_score_std']
            })
        export_data.append(export_row)
    
    export_df = pd.DataFrame(export_data)
    
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name="ml_universe_results.csv",
        mime="text/csv",
        help="Download a CSV file with all model results and metrics."
    )
    
    st.subheader("Results Summary")
    st.dataframe(export_df, use_container_width=True)
    
    st.subheader("💡 Tips for Using Your Results")
    st.markdown("""
    - **Model Selection**: Choose the model with the best performance metrics for your use case.
    - **Cross-validation Scores**: Look for models with high mean scores and low standard deviation for consistency.
    - **Feature Importance**: Use this information to understand which features are most predictive.
    - **Further Steps**: Consider collecting more data, feature engineering, or trying ensemble methods for better performance.
    """)

if __name__ == "__main__":
    main()
