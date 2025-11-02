"""
Visualization nodes for generating comparison charts and plots.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def generate_metrics_comparison_plots(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    output_dir: str = "data/08_reports"
) -> Dict[str, str]:
    """
    Generate comparison plots for classification and regression metrics.
    
    Args:
        classification_metrics: DataFrame with classification model metrics
        regression_metrics: DataFrame with regression model metrics
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with paths to generated plots
    """
    logger.info("Generating metrics comparison plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Classification Metrics Comparison
    if not classification_metrics.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Models Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            if metric in classification_metrics.columns:
                # Plot bars with error bars if cv_std is available
                x_pos = range(len(classification_metrics))
                values = classification_metrics[metric].values
                
                bars = ax.bar(x_pos, values, alpha=0.7, color='steelblue')
                
                # Add CV scores as error bars if available
                if 'cv_std' in classification_metrics.columns:
                    cv_mean = classification_metrics['cv_mean'].values if 'cv_mean' in classification_metrics.columns else values
                    cv_std = classification_metrics['cv_std'].values
                    ax.errorbar(x_pos, cv_mean, yerr=cv_std, fmt='o', color='red', 
                              capsize=5, label='CV Mean ± Std')
                
                ax.set_xlabel('Model', fontweight='bold')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_title(f'{metric.replace("_", " ").title()} by Model')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(classification_metrics['model'].values, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        clf_plot_path = output_path / "classification_metrics_comparison.png"
        plt.savefig(clf_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['classification_comparison'] = str(clf_plot_path)
        logger.info(f"Saved classification comparison plot to {clf_plot_path}")
    
    # 2. Regression Metrics Comparison
    if not regression_metrics.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Models Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['r2', 'mae', 'rmse', 'mape']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            if metric in regression_metrics.columns:
                x_pos = range(len(regression_metrics))
                values = regression_metrics[metric].values
                
                # For error metrics (mae, rmse, mape), lower is better
                color = 'lightcoral' if metric in ['mae', 'rmse', 'mape'] else 'lightgreen'
                bars = ax.bar(x_pos, values, alpha=0.7, color=color)
                
                # Add CV scores as error bars if available
                cv_metric_mean = f'cv_{metric}_mean'
                cv_metric_std = f'cv_{metric}_std'
                
                if cv_metric_mean in regression_metrics.columns and cv_metric_std in regression_metrics.columns:
                    cv_mean = regression_metrics[cv_metric_mean].values
                    cv_std = regression_metrics[cv_metric_std].values
                    ax.errorbar(x_pos, cv_mean, yerr=cv_std, fmt='o', color='darkblue', 
                              capsize=5, label='CV Mean ± Std')
                
                ax.set_xlabel('Model', fontweight='bold')
                ax.set_ylabel(metric.upper(), fontweight='bold')
                ax.set_title(f'{metric.upper()} by Model')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(regression_metrics['model'].values, rotation=45, ha='right')
                if cv_metric_mean in regression_metrics.columns:
                    ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        reg_plot_path = output_path / "regression_metrics_comparison.png"
        plt.savefig(reg_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['regression_comparison'] = str(reg_plot_path)
        logger.info(f"Saved regression comparison plot to {reg_plot_path}")
    
    # 3. Cross-Validation Comparison
    if not classification_metrics.empty and 'cv_mean' in classification_metrics.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = range(len(classification_metrics))
        cv_means = classification_metrics['cv_mean'].values
        cv_stds = classification_metrics['cv_std'].values if 'cv_std' in classification_metrics.columns else None
        
        bars = ax.bar(x_pos, cv_means, alpha=0.7, color='mediumseagreen')
        if cv_stds is not None:
            ax.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Cross-Validation Score', fontweight='bold')
        ax.set_title('Classification Models - Cross-Validation Performance (Mean ± Std)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classification_metrics['model'].values, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds if cv_stds is not None else [0]*len(cv_means))):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}\n±{std:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        cv_plot_path = output_path / "classification_cv_comparison.png"
        plt.savefig(cv_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['classification_cv'] = str(cv_plot_path)
        logger.info(f"Saved CV comparison plot to {cv_plot_path}")
    
    # 4. Regression CV Comparison
    if not regression_metrics.empty and 'cv_r2_mean' in regression_metrics.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = range(len(regression_metrics))
        cv_means = regression_metrics['cv_r2_mean'].values
        cv_stds = regression_metrics['cv_r2_std'].values if 'cv_r2_std' in regression_metrics.columns else None
        
        bars = ax.bar(x_pos, cv_means, alpha=0.7, color='skyblue')
        if cv_stds is not None:
            ax.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Cross-Validation R² Score', fontweight='bold')
        ax.set_title('Regression Models - Cross-Validation Performance (Mean ± Std)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regression_metrics['model'].values, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds if cv_stds is not None else [0]*len(cv_means))):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}\n±{std:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        reg_cv_plot_path = output_path / "regression_cv_comparison.png"
        plt.savefig(reg_cv_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['regression_cv'] = str(reg_cv_plot_path)
        logger.info(f"Saved regression CV comparison plot to {reg_cv_plot_path}")
    
    logger.info(f"Generated {len(plots)} comparison plots")
    return plots


def create_summary_table(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a consolidated summary table of all models.
    
    Args:
        classification_metrics: DataFrame with classification metrics
        regression_metrics: DataFrame with regression metrics
        
    Returns:
        Consolidated summary DataFrame
    """
    logger.info("Creating summary table...")
    
    summary_rows = []
    
    # Classification models
    if not classification_metrics.empty:
        for _, row in classification_metrics.iterrows():
            summary_rows.append({
                'Pipeline': 'Classification',
                'Model': row['model'],
                'Primary_Metric': f"F1: {row.get('f1_score', 0):.4f}",
                'CV_Score': row.get('cv_scores', 'N/A'),
                'Best_Params': row.get('best_params', 'N/A')
            })
    
    # Regression models
    if not regression_metrics.empty:
        for _, row in regression_metrics.iterrows():
            summary_rows.append({
                'Pipeline': 'Regression',
                'Model': row['model'],
                'Primary_Metric': f"R²: {row.get('r2', 0):.4f}",
                'CV_Score': row.get('cv_scores', 'N/A'),
                'Best_Params': row.get('best_params', 'N/A')
            })
    
    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"Created summary table with {len(summary_df)} model entries")
    
    return summary_df
