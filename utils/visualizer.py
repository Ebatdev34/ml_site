import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

class Visualizer:
    """Creates visualizations for ML results"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set2
    
    def plot_confusion_matrix(self, cm):
        """Create confusion matrix heatmap"""
        cm_df = pd.DataFrame(cm)
        
        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, importance, feature_names, top_n=15):
        """Create feature importance bar chart"""
        if len(importance) != len(feature_names):
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        df = df.tail(top_n)
        
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top {min(top_n, len(df))} Feature Importances",
            color='Importance',
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            height=max(400, len(df) * 25),
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(self, data):
        """Create correlation matrix heatmap"""
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        
        fig.update_layout(
            width=600,
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, results, task_type):
        """Create model comparison chart"""
        models = list(results.keys())
        
        if task_type == 'classification':
            metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        else:
            metrics = ['test_r2', 'mae', 'mse', 'rmse']
            metric_names = ['R²', 'MAE', 'MSE', 'RMSE']
        
        comparison_data = []
        for model in models:
            for metric, name in zip(metrics, metric_names):
                comparison_data.append({
                    'Model': model,
                    'Metric': name,
                    'Value': results[model][metric]
                })
        
        df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title="Model Performance Comparison"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    def plot_prediction_scatter(self, y_true, y_pred, model_name):
        """Create scatter plot of predictions vs actual values (for regression)"""
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        min_val = min(df['Actual'].min(), df['Predicted'].min())
        max_val = max(df['Actual'].max(), df['Predicted'].max())
        
        fig = px.scatter(
            df,
            x='Actual',
            y='Predicted',
            title=f"Predictions vs Actual Values - {model_name}",
            trendline="ols"
        )
        
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
        )
        
        fig.add_annotation(
            x=max_val * 0.1,
            y=max_val * 0.9,
            text="Perfect Prediction Line",
            showarrow=False,
            font=dict(color="red")
        )
        
        return fig
    
    def plot_cross_validation_scores(self, results):
        """Create box plot of cross-validation scores"""
        cv_data = []
        for model_name, result in results.items():
            for score in result['cv_scores']:
                cv_data.append({
                    'Model': model_name,
                    'CV Score': score
                })
        
        df = pd.DataFrame(cv_data)
        
        fig = px.box(
            df,
            x='Model',
            y='CV Score',
            title="Cross-Validation Score Distribution"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        return fig