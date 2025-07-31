import streamlit as st
import pandas as pd
import numpy as np
import io
import openpyxl
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer
from utils.task_detector import TaskDetector

st.set_page_config(page_title="ML Universe", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

# Initialize session state variables
for key in ['data', 'processed_data', 'task_type', 'models', 'results']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ['models', 'results'] else {}

def main():
    st.markdown("<h1 style='text-align:center; color:#0ff0fc; text-shadow: 0 0 8px #0ff0fc;'>🌌 ML Universe</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>**Democratizing Machine Learning for Everyone**</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.sidebar.selectbox("Choose a step:", [
        "📁 Data Upload",
        "🔍 Data Exploration",
        "🤖 Model Selection & Training",
        "📊 Results & Analysis",
        "💾 Export Results"
    ])

    pages = {
        "📁 Data Upload": data_upload_page,
        "🔍 Data Exploration": data_exploration_page,
        "🤖 Model Selection & Training": model_training_page,
        "📊 Results & Analysis": results_page,
        "💾 Export Results": export_page
    }
    pages[page]()

def data_upload_page():
    st.header("📁 Data Upload")
    st.markdown("Upload your CSV or Excel dataset to get started with machine learning!")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xls", "xlsx"],
        help="Upload a CSV or Excel file with your dataset. Make sure it includes column headers."
    )

    if uploaded_file:
        file_ext = uploaded_file.name.lower().split('.')[-1]

        try:
            if file_ext in ['xls', 'xlsx']:
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)

            st.session_state.data = data

            st.success(f"✅ Dataset uploaded! Shape: {data.shape}")
            cols = st.columns(3)
            cols[0].metric("Rows", data.shape[0])
            cols[1].metric("Columns", data.shape[1])
            cols[2].metric("Missing Values", data.isnull().sum().sum())

            st.subheader("Data Preview")
            st.dataframe(data.head(10), use_container_width=True)

            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)

            st.subheader("Target Column Selection")
            target_col = st.selectbox("Select the target column:", data.columns.tolist())
            if target_col:
                task_type = TaskDetector().detect_task_type(data[target_col])
                st.session_state.task_type = task_type
                st.info(f"🎯 Task detected: **{task_type.title()}**")

                if task_type == 'classification':
                    st.write(f"Number of classes: {data[target_col].nunique()}")
                    st.bar_chart(data[target_col].value_counts())
                else:
                    st.write("Target statistics:")
                    st.write(data[target_col].describe())

                if st.button("🔄 Process Data"):
                    if file_ext in ['xls', 'xlsx']:
                        st.error("❌ We process data of xls and xlsx at 7/35/2025, please stick to CSV.")
                        return
                    with st.spinner("Processing data..."):
                        processed = DataProcessor().preprocess_data(data, target_col, task_type)
                        st.session_state.processed_data = processed
                        st.success("✅ Data processed!")
                        st.info("Go to 'Data Exploration' to view the processed data.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


def data_exploration_page():
    st.header("🔍 Data Exploration")
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data first!")
        return

    st.subheader("Original vs Processed Data")
    col1, col2 = st.columns(2)
    col1.write("**Original Data**")
    col1.dataframe(st.session_state.data.head(), use_container_width=True)
    col2.write("**Processed Data**")
    col2.dataframe(st.session_state.processed_data['X'].head(), use_container_width=True)

    st.subheader("Preprocessing Steps Applied")
    preprocessing_info = st.session_state.processed_data.get('preprocessing_info', {})
    if 'encoded_columns' in preprocessing_info:
        st.write("**Categorical Encoding:**")
        for col in preprocessing_info['encoded_columns']:
            st.write(f"- {col}: Label encoded")
    if 'scaled_columns' in preprocessing_info:
        st.write("**Feature Scaling:**")
        for col in preprocessing_info['scaled_columns']:
            st.write(f"- {col}: Standard scaled")

    st.subheader("Feature Correlations")
    numeric = st.session_state.processed_data['X'].select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        fig = Visualizer().plot_correlation_matrix(numeric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numerical features for correlation analysis.")

def model_training_page():
    st.header("🤖 Model Selection & Training")
    if st.session_state.processed_data is None:
        st.warning("⚠️ Upload and process data first!")
        return

    task_type = st.session_state.task_type
    st.subheader(f"Available Models for {task_type.title()}")

    models = {
        'classification': {
            'Logistic Regression': 'logistic_regression',
            'Decision Tree': 'decision_tree',
            'Random Forest': 'random_forest'
        },
        'regression': {
            'Linear Regression': 'linear_regression',
            'Decision Tree': 'decision_tree_regressor',
            'Random Forest': 'random_forest_regressor'
        }
    }

    available_models = models.get(task_type, {})
    selected = st.multiselect("Select models to train:", list(available_models.keys()), default=list(available_models.keys())[:2])

    if not selected:
        st.warning("⚠️ Select at least one model!")
        return

    with st.expander("⚙️ Advanced Options"):
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42)
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

    if st.button("🚀 Train Models"):
        trainer = ModelTrainer()
        results = {}
        progress_bar = st.progress(0)
        status = st.empty()

        for i, name in enumerate(selected):
            status.text(f"Training {name}...")
            try:
                res = trainer.train_model(
                    st.session_state.processed_data,
                    available_models[name],
                    task_type,
                    test_size=test_size,
                    random_state=random_state,
                    cv_folds=cv_folds
                )
                results[name] = res
                progress_bar.progress((i + 1) / len(selected))
            except Exception as e:
                st.error(f"❌ Error training {name}: {e}")

        status.text("Training completed!")
        st.session_state.results = results
        st.success(f"✅ Trained {len(results)} models! Check 'Results & Analysis'.")

def results_page():
    st.header("📊 Results & Analysis")
    if not st.session_state.results:
        st.warning("⚠️ Train models first!")
        return

    task_type = st.session_state.task_type
    results = st.session_state.results

    data = []
    for model, res in results.items():
        if task_type == 'classification':
            data.append({
                'Model': model,
                'Accuracy': res['test_accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1_score'],
                'CV Score': res['cv_score_mean']
            })
        else:
            data.append({
                'Model': model,
                'R²': res['test_r2'],
                'MAE': res['mae'],
                'MSE': res['mse'],
                'RMSE': res['rmse'],
                'CV Score': res['cv_score_mean']
            })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    best_metric = 'Accuracy' if task_type == 'classification' else 'R²'
    best_model = df.loc[df[best_metric].idxmax()]
    st.success(f"🏆 Best model: **{best_model['Model']}** ({best_metric}: {best_model[best_metric]:.4f})")

    for model, res in results.items():
        with st.expander(f"📈 Details: {model}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Performance Metrics:**")
                if task_type == 'classification':
                    st.write(f"- Accuracy: {res['test_accuracy']:.4f}")
                    st.write(f"- Precision: {res['precision']:.4f}")
                    st.write(f"- Recall: {res['recall']:.4f}")
                    st.write(f"- F1-Score: {res['f1_score']:.4f}")
                else:
                    st.write(f"- R² Score: {res['test_r2']:.4f}")
                    st.write(f"- MAE: {res['mae']:.4f}")
                    st.write(f"- MSE: {res['mse']:.4f}")
                    st.write(f"- RMSE: {res['rmse']:.4f}")
                st.write(f"- CV Score: {res['cv_score_mean']:.4f} (±{res['cv_score_std']:.4f})")

            with col2:
                viz = Visualizer()
                if task_type == 'classification' and 'confusion_matrix' in res:
                    st.write("**Confusion Matrix:**")
                    fig = viz.plot_confusion_matrix(res['confusion_matrix'])
                    st.plotly_chart(fig, use_container_width=True)
                if 'feature_importance' in res:
                    st.write("**Feature Importance:**")
                    fig = viz.plot_feature_importance(res['feature_importance'], st.session_state.processed_data['feature_names'])
                    st.plotly_chart(fig, use_container_width=True)

def export_page():
    st.header("💾 Export Results")
    if not st.session_state.results:
        st.warning("⚠️ Train some models first!")
        return

    data = []
    task_type = st.session_state.task_type
    for model, res in st.session_state.results.items():
        row = {'Model': model}
        if task_type == 'classification':
            row.update({
                'Accuracy': res['test_accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1_Score': res['f1_score'],
                'CV_Score_Mean': res['cv_score_mean'],
                'CV_Score_Std': res['cv_score_std']
            })
        else:
            row.update({
                'R2_Score': res['test_r2'],
                'MAE': res['mae'],
                'MSE': res['mse'],
                'RMSE': res['rmse'],
                'CV_Score_Mean': res['cv_score_mean'],
                'CV_Score_Std': res['cv_score_std']
            })
        data.append(row)

    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Results (CSV)", data=csv_buffer.getvalue(), file_name="ml_universe_results.csv", mime="text/csv")

    st.subheader("Results Summary")
    st.dataframe(df, use_container_width=True)

    st.subheader("💡 Tips for Using Your Results")
    st.markdown("""
    - **Model Selection**: Choose models with best metrics for your use case.  
    - **Cross-validation**: Look for high mean and low std scores.  
    - **Feature Importance**: Understand which features matter most.  
    - **Next Steps**: Consider more data, feature engineering, or ensembles.
    """)
# --- Dev Chat Sidebar (simple shared message board) ---
with st.sidebar:
    st.subheader("💬 Dev Chat Room")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    new_message = st.text_input("Leave a message", key="dev_chat_input")

    if st.button("Send"):
        if new_message.strip():
            st.session_state.chat_messages.append(new_message)
            st.experimental_rerun()

    if st.session_state.chat_messages:
        st.write("### 📜 Messages")
        for i, msg in enumerate(reversed(st.session_state.chat_messages), 1):
            st.markdown(f"**Dev {len(st.session_state.chat_messages) - i + 1}:** {msg}")

if __name__ == "__main__":
    main()
