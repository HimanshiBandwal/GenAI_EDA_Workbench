"""
EDA Workbench: A tool for exploratory data analysis on CSV datasets.

This script loads a CSV dataset, cleans it, generates visualizations, supports natural language queries,
and performs fraud detection modeling. It was last updated on May 17, 2025, at 15:01 IST, and is designed for the
`creditcard.csv` dataset, which contains credit card transaction data with features like `Time`, `V1` to `V28`
(PCA-transformed), `Amount`, and `Class` (0 for non-fraud, 1 for fraud).

Version: 3.0 (Added support for 'top features correlated with Class' query and version logging)

Features:
- Data cleaning: Fills missing numerical values, drops missing categorical rows, caps outliers.
- Visualizations: Histograms, box plots, scatter plots, heatmaps, pair plots, violin plots, count plots,
  class-stratified plots, feature importance, confusion matrix, precision-recall curve.
- Queries: Supports average, max/min, "who", "how many", correlation, conditional averages, fraud rate,
  top features correlated with Class.
- Fraud Detection: Prepares data with SMOTE, trains a Random Forest model with cross-validation, and evaluates performance.
"""

import pandas as pd
import os
import logging
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Version check
SCRIPT_VERSION = "3.0"
logger.info(f"Running EDA Workbench script version {SCRIPT_VERSION}")

def setup_logging(output_dir):
    """Set up logging to both file and console, ensuring the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, 'eda_workbench.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

def clean_outputs_folder(output_dir):
    """Remove older files with timestamps in their names from the output directory."""
    try:
        keep_files = [
            'scatter_first_two.png', 'correlation_heatmap.png',
            'pairplot_numerical.png', 'dataset_summary.txt',
            'cleaned_dataset.csv', 'eda_report.txt', 'eda_workbench.log',
            'countplot_Class.png', 'countplot_Class_log.png',
            'histogram_Amount_by_Class.png', 'feature_importance.png',
            'boxplot_Amount_by_Class.png', 'confusion_matrix.png',
            'precision_recall_curve.png'
        ]
        
        timestamp_pattern = re.compile(r'_\d{8}_\d{6}\.(png|txt)$')
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if timestamp_pattern.search(filename) and filename not in keep_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error cleaning output folder: {str(e)}")

def clean_dataset(df):
    """Clean the dataset by handling missing values and outliers."""
    try:
        logger.info("Starting data cleaning process...")
        
        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df_cleaned = df.copy()
        
        for col in numerical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                median_val = df_cleaned[col].median()
                df_cleaned.loc[:, col] = df_cleaned[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        if len(categorical_cols) > 0:
            before_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=categorical_cols)
            after_rows = len(df_cleaned)
            if before_rows != after_rows:
                logger.info(f"Dropped {before_rows - after_rows} rows with missing categorical values.")
        
        for col in numerical_cols:
            if col != 'Class':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)][col]
                if not outliers.empty:
                    logger.info(f"Outliers detected in {col}: {outliers.tolist()[:10]}")
                    df_cleaned.loc[:, col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in {col} between {lower_bound} and {upper_bound}")
        
        for col in numerical_cols:
            if df_cleaned[col].dtype in ['float64', 'float32', 'float']:
                df_cleaned[col] = df_cleaned[col].astype('float32')
            elif df_cleaned[col].dtype in ['int64', 'int32', 'int']:
                df_cleaned[col] = df_cleaned[col].astype('int32')
        
        logger.info("Data cleaning completed.")
        return df_cleaned
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        return df

def load_dataset(file_path, chunksize=10000):
    """Load a CSV file in chunks for scalability and perform validation."""
    try:
        logger.info(f"Attempting to load dataset from: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist.")
            return None, None
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Dataset file size: {file_size:.2f} MB")
        
        if file_size > 100:
            logger.info(f"File size exceeds 100 MB. Loading in chunks of {chunksize} rows...")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunksize, on_bad_lines='skip'):
                chunk = clean_dataset(chunk)
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            df = clean_dataset(df)
        
        if df.empty:
            logger.error("The dataset is empty after loading and cleaning.")
            return None, None
        
        logger.info("Dataset loaded successfully!")
        
        info_buffer = StringIO()
        df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()
        logger.info("Dataset Info (after cleaning):")
        print("\nDataset Info (after cleaning):")
        print(info_str)
        
        logger.info("Column Data Types (after cleaning):")
        print("\nColumn Data Types (after cleaning):")
        print(df.dtypes)
        
        logger.info("First 5 rows (after cleaning):")
        print("\nFirst 5 rows (after cleaning):")
        print(df.head())
        
        logger.info("Missing Values (after cleaning):")
        print("\nMissing Values (after cleaning):")
        print(df.isnull().sum())
        
        return df, info_str
    
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error in CSV file: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None, None

def compute_summary_statistics(df):
    """Compute summary statistics for numerical columns."""
    try:
        numerical_cols = df.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            summary = df[numerical_cols].describe()
            logger.info("Summary Statistics for Numerical Columns:")
            print("\nSummary Statistics for Numerical Columns:")
            print(summary)
            return summary
        else:
            logger.warning("No numerical columns found for summary statistics.")
            return None
    except Exception as e:
        logger.error(f"Error computing summary statistics: {str(e)}")
        return None

def generate_visualizations(df, output_dir, max_points=1000, max_cols_to_plot=5):
    """Generate basic visualizations for numerical columns with downsampling."""
    try:
        numerical_cols = df.select_dtypes(include=np.number).columns
        if numerical_cols.empty:
            logger.warning("No numerical columns found for visualizations.")
            return
        
        if len(df) > max_points:
            logger.info(f"Downsampling data to {max_points} points for visualizations...")
            df_sample = df.sample(n=max_points, random_state=42)
        else:
            df_sample = df
        
        numerical_cols_to_plot = numerical_cols[:max_cols_to_plot]
        logger.info(f"Limiting visualizations to the first {max_cols_to_plot} numerical columns: {list(numerical_cols_to_plot)}")
        
        sns.set(style="whitegrid")
        
        for col in numerical_cols_to_plot:
            if col != 'Class':
                plt.figure(figsize=(8, 6))
                sns.histplot(df_sample[col], kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'histogram_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Histogram for {col} saved to {plot_path}")
        
        for col in numerical_cols_to_plot:
            if col != 'Class':
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df_sample[col])
                plt.title(f'Box Plot of {col}')
                plt.ylabel(col)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'boxplot_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Box plot for {col} saved to {plot_path}")
        
        if len(numerical_cols) >= 2:
            col1, col2 = numerical_cols[0], numerical_cols[1]
            if col1 == 'Class':
                col1 = numerical_cols[1]
                col2 = numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0]
            elif col2 == 'Class':
                col2 = numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0]
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df_sample[col1], y=df_sample[col2])
            plt.title(f'Scatter Plot of {col1} vs. {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'scatter_first_two.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Scatter plot of {col1} vs. {col2} saved to {plot_path}")
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(8, 6))
            corr = df[numerical_cols].corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap of Numerical Columns')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Correlation heatmap saved to {plot_path}")
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def generate_advanced_visualizations(df, output_dir, max_points=1000, max_cols_to_plot=5):
    """Generate advanced visualizations with downsampling and include Class-specific plots."""
    try:
        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(df) > max_points:
            logger.info(f"Downsampling data to {max_points} points for advanced visualizations...")
            df_sample = df.sample(n=max_points, random_state=42)
        else:
            df_sample = df
        
        numerical_cols_to_plot = numerical_cols[:max_cols_to_plot]
        logger.info(f"Limiting advanced visualizations to the first {max_cols_to_plot} numerical columns: {list(numerical_cols_to_plot)}")
        
        sns.set(style="whitegrid")
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 8))
            pair_plot = sns.pairplot(df_sample[numerical_cols_to_plot])
            plot_path = os.path.join(output_dir, 'pairplot_numerical.png')
            pair_plot.savefig(plot_path)
            plt.close()
            logger.info(f"Pair plot for numerical columns saved to {plot_path}")
        
        for col in numerical_cols_to_plot:
            if col != 'Class':
                plt.figure(figsize=(8, 6))
                sns.violinplot(y=df_sample[col])
                plt.title(f'Violin Plot of {col}')
                plt.ylabel(col)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'violin_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Violin plot for {col} saved to {plot_path}")
        
        for col in categorical_cols:
            if col != 'name':
                plt.figure(figsize=(8, 6))
                sns.countplot(x=col, data=df_sample)
                plt.title(f'Count Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'countplot_{col}.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Count plot for {col} saved to {plot_path}")
        
        if 'Class' in df.columns:
            # Regular count plot
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Class', data=df_sample)
            plt.title('Count Plot of Class')
            plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'countplot_Class.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Count plot for Class saved to {plot_path}")
            
            # Log-scale count plot
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Class', data=df_sample)
            plt.title('Count Plot of Class (Log Scale)')
            plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
            plt.ylabel('Count (Log Scale)')
            plt.yscale('log')
            plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'countplot_Class_log.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Log-scale count plot for Class saved to {plot_path}")
        
        # Class-stratified histogram for Amount
        if 'Class' in df.columns and 'Amount' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df_sample, x='Amount', hue='Class', multiple='stack', bins=50)
            plt.title('Histogram of Amount by Class')
            plt.xlabel('Amount')
            plt.ylabel('Count')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'histogram_Amount_by_Class.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Histogram of Amount by Class saved to {plot_path}")
            
            # Box plot of Amount by Class
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Class', y='Amount', data=df_sample)
            plt.title('Box Plot of Amount by Class')
            plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
            plt.ylabel('Amount')
            plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'boxplot_Amount_by_Class.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Box plot of Amount by Class saved to {plot_path}")
    
    except Exception as e:
        logger.error(f"Error generating advanced visualizations: {str(e)}")

def save_summary(df, info_str, summary_stats, output_dir):
    """Save a summary of the dataset to a text file."""
    try:
        summary_file = os.path.join(output_dir, 'dataset_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("Dataset Summary\n")
            f.write("=" * 20 + "\n")
            f.write("Dataset Info (after cleaning):\n")
            f.write(info_str + "\n")
            f.write("First 5 rows:\n")
            f.write(df.head().to_string() + "\n\n")
            f.write("Missing Values:\n")
            f.write(df.isnull().sum().to_string() + "\n\n")
            if summary_stats is not None:
                f.write("Summary Statistics:\n")
                f.write(summary_stats.to_string() + "\n")
            if 'Class' in df.columns:
                fraud_counts = df['Class'].value_counts()
                fraud_rate = (fraud_counts.get(1, 0) / len(df)) * 100
                f.write("\nClass Distribution:\n")
                f.write(fraud_counts.to_string() + "\n")
                f.write(f"Fraud Rate: {fraud_rate:.2f}%\n")
        
        logger.info(f"Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving summary: {str(e)}")

def export_cleaned_dataset(df, output_dir):
    """Export the cleaned dataset to a CSV file."""
    try:
        cleaned_file = os.path.join(output_dir, 'cleaned_dataset.csv')
        df.to_csv(cleaned_file, index=False)
        logger.info(f"Cleaned dataset exported to {cleaned_file}")
    except PermissionError as e:
        logger.error(f"Permission denied: Unable to write to {cleaned_file}.")
    except Exception as e:
        logger.error(f"Error exporting cleaned dataset: {str(e)}")

def generate_report(df, summary_stats, output_dir, max_cols_to_plot=5):
    """Generate a text report summarizing the EDA process."""
    try:
        report_file = os.path.join(output_dir, 'eda_report.txt')
        numerical_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols_to_plot = numerical_cols[:max_cols_to_plot]
        
        with open(report_file, 'w') as f:
            f.write("Exploratory Data Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Dataset Overview\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of rows: {len(df)}\n")
            f.write(f"Number of columns: {len(df.columns)}\n\n")
            
            if summary_stats is not None:
                f.write("Summary Statistics for Numerical Columns\n")
                f.write("-" * 40 + "\n")
                f.write(summary_stats.to_string() + "\n\n")
            
            if 'Class' in df.columns:
                fraud_counts = df['Class'].value_counts()
                fraud_rate = (fraud_counts.get(1, 0) / len(df)) * 100
                f.write("Class Distribution\n")
                f.write("-" * 20 + "\n")
                f.write(fraud_counts.to_string() + "\n")
                f.write(f"Fraud Rate: {fraud_rate:.2f}%\n\n")
            
            f.write("Basic Visualizations\n")
            f.write("-" * 20 + "\n")
            for col in numerical_cols_to_plot:
                if col != 'Class':
                    f.write(f"Histogram of {col}: See histogram_{col}.png\n")
                    f.write(f"Box Plot of {col}: See boxplot_{col}.png\n")
            if len(numerical_cols) >= 2:
                col1, col2 = numerical_cols[0], numerical_cols[1]
                if col1 == 'Class':
                    col1 = numerical_cols[1]
                    col2 = numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0]
                elif col2 == 'Class':
                    col2 = numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0]
                f.write(f"Scatter Plot of {col1} vs. {col2}: See scatter_first_two.png\n")
            if len(numerical_cols) > 1:
                f.write("Correlation Heatmap: See correlation_heatmap.png\n")
            f.write("\n")
            
            f.write("Advanced Visualizations\n")
            f.write("-" * 20 + "\n")
            if len(numerical_cols) > 1:
                f.write("Pair Plot of Numerical Columns: See pairplot_numerical.png\n")
            for col in numerical_cols_to_plot:
                if col != 'Class':
                    f.write(f"Violin Plot of {col}: See violin_{col}.png\n")
            for col in categorical_cols:
                if col != 'name':
                    f.write(f"Count Plot of {col}: See countplot_{col}.png\n")
            if 'Class' in df.columns:
                f.write("Count Plot of Class: See countplot_Class.png\n")
                f.write("Count Plot of Class (Log Scale): See countplot_Class_log.png\n")
                f.write("Histogram of Amount by Class: See histogram_Amount_by_Class.png\n")
                f.write("Box Plot of Amount by Class: See boxplot_Amount_by_Class.png\n")
            f.write("\n")
            
            f.write("Model Visualizations\n")
            f.write("-" * 20 + "\n")
            f.write("Feature Importance Plot: See feature_importance.png\n")
            f.write("Confusion Matrix: See confusion_matrix.png\n")
            f.write("Precision-Recall Curve: See precision_recall_curve.png\n")
            f.write("\n")
            
            f.write("Notes\n")
            f.write("-" * 20 + "\n")
            f.write("This dataset has been cleaned by filling missing numerical values with medians, dropping rows with missing categorical values, and capping outliers using the IQR method.\n")
            f.write("Project completed as of May 17, 2025, with comprehensive EDA and fraud detection modeling.\n")
        
        logger.info(f"EDA report generated at {report_file}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")

def process_query(df, query):
    """Process a natural language query about the dataset."""
    try:
        if df.empty:
            return "Error: The dataset is empty. Cannot process queries."
        
        tokens = word_tokenize(query.lower())
        tokens = ['average' if token == 'averge' else token for token in tokens]
        
        numerical_cols = df.select_dtypes(include=np.number).columns
        string_cols = df.select_dtypes(include=['object']).columns
        numerical_cols_lower = {col.lower(): col for col in numerical_cols}
        logger.info(f"Numerical columns available: {list(numerical_cols)}")
        
        # Fraud rate query
        if 'fraud' in tokens and 'rate' in tokens and 'Class' in df.columns:
            fraud_counts = df['Class'].value_counts()
            fraud_rate = (fraud_counts.get(1, 0) / len(df)) * 100
            return f"The fraud rate is {fraud_rate:.2f}%."
        
        # Top features correlated with Class
        if 'top' in tokens and 'features' in tokens and 'correlated' in tokens and 'class' in df.columns:
            try:
                correlations = {}
                for col in numerical_cols:
                    if col != 'Class':
                        if df[col].std() == 0 or df['Class'].std() == 0:
                            logger.warning(f"Skipping correlation for {col} due to zero variance.")
                            continue
                        corr = df[col].corr(df['Class'])
                        correlations[col] = abs(corr) if not pd.isna(corr) else 0
                if not correlations:
                    return "Unable to compute correlations due to zero variance in features or Class."
                sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                top_n = 5
                for i, token in enumerate(tokens):
                    if token == 'top' and i + 1 < len(tokens):
                        try:
                            top_n = int(tokens[i + 1])
                            break
                        except ValueError:
                            logger.warning(f"Invalid top N value in query: {tokens[i+1]}. Using default top_n=5.")
                            pass
                top_n = min(top_n, len(sorted_corrs))  # Ensure we don't exceed available features
                result = f"Top {top_n} features correlated with Class:\n"
                for feature, corr in sorted_corrs[:top_n]:
                    result += f"{feature}: {corr:.4f}\n"
                return result
            except Exception as e:
                logger.error(f"Error computing top features correlated with Class: {str(e)}")
                return "Error computing top correlated features."
        
        # Correlation between two columns
        if 'correlation' in tokens and 'between' in tokens:
            col1, col2 = None, None
            between_idx = tokens.index('between')
            remaining_tokens = tokens[between_idx + 1:]
            for i, token in enumerate(remaining_tokens):
                if token in numerical_cols_lower:
                    if not col1:
                        col1 = numerical_cols_lower[token]
                    elif not col2 and token != col1.lower():
                        col2 = numerical_cols_lower[token]
                        break
            if col1 and col2:
                if col1 == 'Class' or col2 == 'Class':
                    return "Correlation with Class is not meaningful as it is a binary categorical variable."
                if df[col1].std() == 0 or df[col2].std() == 0:
                    return f"Cannot compute correlation between {col1} and {col2}: one of the columns has zero variance."
                corr = df[col1].corr(df[col2])
                if pd.isna(corr):
                    return f"Cannot compute correlation between {col1} and {col2}: correlation is undefined."
                return f"The correlation between {col1} and {col2} is {corr:.2f}."
            return f"Please specify two numerical columns for correlation (e.g., 'What is the correlation between {numerical_cols[0]} and {numerical_cols[1]}?')."
        
        # Average or mean query
        if 'average' in tokens or 'mean' in tokens:
            target_col = None
            for i, token in enumerate(tokens):
                if token in numerical_cols_lower:
                    if i > 0 and (tokens[i-1] in ['average', 'mean']):
                        target_col = numerical_cols_lower[token]
                        break
            if not target_col:
                return f"Please specify a numerical column for the average (e.g., 'average {numerical_cols[0]}')."
            
            condition_col = None
            condition_value = None
            
            if 'for' in tokens and ('class' in tokens or 'people' in tokens):
                for_idx = tokens.index('for')
                class_idx = tokens.index('class') if 'class' in tokens else tokens.index('people')
                if 'class' in tokens and class_idx == for_idx + 1:
                    condition_tokens = tokens[class_idx + 1:]
                    for i, token in enumerate(condition_tokens):
                        if token in ['0', '1']:
                            condition_col = 'Class'
                            condition_value = int(token)
                            break
                elif 'people' in tokens and class_idx == for_idx + 1:
                    condition_tokens = tokens[class_idx + 1:]
                    for i, token in enumerate(condition_tokens):
                        if token in ['above', 'over']:
                            if i + 1 < len(condition_tokens) and condition_tokens[i+1] in numerical_cols_lower and condition_tokens[i+1] != target_col.lower():
                                condition_col = numerical_cols_lower[condition_tokens[i+1]]
                                if i + 2 < len(condition_tokens):
                                    try:
                                        condition_value = float(condition_tokens[i+2])
                                    except ValueError:
                                        return f"Invalid numerical value in query: '{condition_tokens[i+2]}'. Please provide a valid number (e.g., 'above {condition_col} 50000')."
                        elif token in ['below', 'under']:
                            if i + 1 < len(condition_tokens) and condition_tokens[i+1] in numerical_cols_lower and condition_tokens[i+1] != target_col.lower():
                                condition_col = numerical_cols_lower[condition_tokens[i+1]]
                                if i + 2 < len(condition_tokens):
                                    try:
                                        condition_value = float(condition_tokens[i+2])
                                    except ValueError:
                                        return f"Invalid numerical value in query: '{condition_tokens[i+2]}'. Please provide a valid number (e.g., 'below {condition_col} 50000')."
                        if condition_col and condition_value is not None:
                            break
            
            if condition_col:
                if condition_col == 'Class':
                    filtered_df = df[df['Class'] == condition_value]
                    if filtered_df.empty:
                        return f"No entries found with Class {condition_value}."
                    avg = filtered_df[target_col].mean()
                    if pd.isna(avg):
                        return f"Cannot compute average {target_col} for Class {condition_value}: no valid data."
                    count = len(filtered_df)
                    return f"The average {target_col} for {count} entries with Class {condition_value} (Fraud: {bool(condition_value)}) is {avg:.2f}."
                elif condition_col in numerical_cols:
                    condition_value_display = int(condition_value) if condition_value.is_integer() else condition_value
                    if 'above' in tokens or 'over' in tokens:
                        filtered_df = df[df[condition_col] > condition_value]
                        if filtered_df.empty:
                            return f"No entries found with {condition_col} above {condition_value_display}."
                        avg = filtered_df[target_col].mean()
                        if pd.isna(avg):
                            return f"Cannot compute average {target_col} for {condition_col} above {condition_value_display}: no valid data."
                        count = len(filtered_df)
                        return f"The average {target_col} for {count} entries with {condition_col} above {condition_value_display} is {avg:.2f}."
                    elif 'below' in tokens or 'under' in tokens:
                        filtered_df = df[df[condition_col] < condition_value]
                        if filtered_df.empty:
                            return f"No entries found with {condition_col} below {condition_value_display}."
                        avg = filtered_df[target_col].mean()
                        if pd.isna(avg):
                            return f"Cannot compute average {target_col} for {condition_col} below {condition_value_display}: no valid data."
                        count = len(filtered_df)
                        return f"The average {target_col} for {count} entries with {condition_col} below {condition_value_display} is {avg:.2f}."
            
            avg = df[target_col].mean()
            if pd.isna(avg):
                return f"Cannot compute average {target_col}: no valid data."
            return f"The average {target_col} is {avg:.2f}."
        
        # How many query
        if 'how' in tokens and 'many' in tokens:
            col = None
            for token in tokens:
                if token in numerical_cols_lower:
                    col = numerical_cols_lower[token]
                    break
            if not col:
                return f"Please specify a numerical column (e.g., 'how many entries are above {numerical_cols[0]} 50000'). Available columns: {list(numerical_cols)}."
            if 'above' in tokens or 'over' in tokens:
                above_idx = tokens.index('above') if 'above' in tokens else tokens.index('over')
                for i in range(above_idx + 1, len(tokens)):
                    if tokens[i] == col.lower():
                        continue
                    try:
                        value = float(tokens[i])
                        value_display = int(value) if value.is_integer() else value
                        count = len(df[df[col] > value])
                        entry_str = "entry" if count == 1 else "entries"
                        return f"There are {count} {entry_str} with {col} above {value_display}."
                    except ValueError:
                        continue
                return f"Please provide a valid number (e.g., 'above {col} 50000')."
            elif 'below' in tokens or 'under' in tokens:
                below_idx = tokens.index('below') if 'below' in tokens else tokens.index('under')
                for i in range(below_idx + 1, len(tokens)):
                    if tokens[i] == col.lower():
                        continue
                    try:
                        value = float(tokens[i])
                        value_display = int(value) if value.is_integer() else value
                        count = len(df[df[col] < value])
                        entry_str = "entry" if count == 1 else "entries"
                        return f"There are {count} {entry_str} with {col} below {value_display}."
                    except ValueError:
                        continue
                return f"Please provide a valid number (e.g., 'below {col} 50000')."
        
        # Who query (for datasets with a 'name' column)
        if 'who' in tokens and 'name' in df.columns:
            for token in tokens:
                if token in numerical_cols_lower:
                    col = numerical_cols_lower[token]
                    if 'highest' in tokens or 'max' in tokens:
                        max_val = df[col].max()
                        if pd.isna(max_val):
                            return f"Cannot find maximum {col}: no valid data."
                        max_row = df[df[col] == max_val]
                        name = max_row['name'].iloc[0]
                        return f"{name} has the highest {col} of {max_val}."
                    elif 'lowest' in tokens or 'min' in tokens:
                        min_val = df[col].min()
                        if pd.isna(min_val):
                            return f"Cannot find minimum {col}: no valid data."
                        min_row = df[df[col] == min_val]
                        name = min_row['name'].iloc[0]
                        return f"{name} has the lowest {col} of {min_val}."
            return f"No numerical column found in query. Available columns: {list(numerical_cols)}."
        
        # Max/min queries
        if 'max' in tokens or 'maximum' in tokens or 'highest' in tokens:
            for token in tokens:
                if token in numerical_cols_lower:
                    col = numerical_cols_lower[token]
                    max_val = df[col].max()
                    if pd.isna(max_val):
                        return f"Cannot find maximum {col}: no valid data."
                    return f"The maximum {col} is {max_val}."
            return f"No numerical column found in query. Available columns: {list(numerical_cols)}."
        
        if 'min' in tokens or 'minimum' in tokens or 'lowest' in tokens:
            for token in tokens:
                if token in numerical_cols_lower:
                    col = numerical_cols_lower[token]
                    min_val = df[col].min()
                    if pd.isna(min_val):
                        return f"Cannot find minimum {col}: no valid data."
                    return f"The minimum {col} is {min_val}."
            return f"No numerical column found in query. Available columns: {list(numerical_cols)}."
        
        # Enhanced logging for unrecognized queries
        logger.warning(f"Unrecognized query pattern: {query}. Tokens: {tokens}")
        return "Sorry, I didn't understand the query. Try one of the supported query types listed at the start of the session."
    
    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}")
        return "Error processing query."

def train_fraud_detection_model(df, output_dir):
    """Train a Random Forest model for fraud detection with SMOTE, cross-validation, and advanced evaluation."""
    try:
        if 'Class' not in df.columns:
            logger.warning("Class column not found. Skipping fraud detection modeling.")
            return
        
        logger.info("Starting fraud detection modeling...")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        minority_count = sum(y_train == 1)
        k_neighbors = min(5, max(1, minority_count - 1))
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_smote, y_train_smote)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
        logger.info(f"Cross-validation AUC-ROC scores: {cv_scores}")
        logger.info(f"Mean CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        auc_roc = roc_auc_score(y_test, y_pred_prob)
        
        # Feature importance
        feature_importance = model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=features)
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {plot_path}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Non-Fraud', 'Fraud'])
        plt.yticks([0.5, 1.5], ['Non-Fraud', 'Fraud'], rotation=0)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {plot_path}")
        
        # Precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Precision-recall curve saved to {plot_path}")
        
        model_report_file = os.path.join(output_dir, 'fraud_detection_model_report.txt')
        with open(model_report_file, 'w') as f:
            f.write("Fraud Detection Model Report\n")
            f.write("=" * 40 + "\n\n")
            f.write("Model: Random Forest with SMOTE\n")
            f.write(f"Cross-validation AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nAUC-ROC Score (Test Set):\n")
            f.write(f"{auc_roc:.4f}\n")
        
        logger.info(f"Fraud detection model report saved to {model_report_file}")
        print("\nFraud Detection Model Report:")
        print(f"Cross-validation AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(report)
        print(f"AUC-ROC Score (Test Set): {auc_roc:.4f}")
    
    except Exception as e:
        logger.error(f"Error training fraud detection model: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="EDA Workbench: Load, analyze, and model a CSV dataset.")
    parser.add_argument('--file', type=str, default="../data/sample_dataset.csv", help="Path to the CSV file")
    parser.add_argument('--output-dir', type=str, default="../outputs", help="Directory to save outputs")
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    clean_outputs_folder(args.output_dir)
    
    logger.info("Starting EDA process...")
    df, info_str = load_dataset(args.file)
    
    if df is not None:
        summary_stats = compute_summary_statistics(df)
        generate_visualizations(df, args.output_dir)
        generate_advanced_visualizations(df, args.output_dir)
        save_summary(df, info_str, summary_stats, args.output_dir)
        export_cleaned_dataset(df, args.output_dir)
        generate_report(df, summary_stats, args.output_dir)
        train_fraud_detection_model(df, args.output_dir)
        logger.info("Basic EDA and modeling completed.")
        
        numerical_cols = df.select_dtypes(include=np.number).columns
        first_num_col = numerical_cols[0] if len(numerical_cols) > 0 else "column"
        second_num_col = numerical_cols[1] if len(numerical_cols) > 1 else "column"
        has_name_col = 'name' in df.columns
        has_class_col = 'Class' in df.columns
        
        print("\nStarting query session...")
        print("Supported query types:")
        print(f"- Average/mean: 'What is the average {first_num_col}?'")
        print(f"- Conditional average (numerical): 'What is the average {first_num_col} for people above {second_num_col} 50000?'")
        if has_class_col:
            print(f"- Conditional average (Class): 'What is the average {first_num_col} for Class 1?'")
        print(f"- Correlation: 'What is the correlation between {first_num_col} and {second_num_col}?'")
        if has_class_col:
            print(f"- Top features: 'What are the top 5 features correlated with Class?'")
        print(f"- Max/min: 'What is the maximum {first_num_col}?'")
        if has_name_col:
            print(f"- Who: 'Who has the highest {first_num_col}?'")
        print(f"- How many: 'How many entries are above {first_num_col} 50000?'")
        if has_class_col:
            print("- Fraud rate: 'What is the fraud rate?'")
        print("Type 'exit' to quit.")
        while True:
            query = input("Enter your query: ")
            if query.lower() == 'exit':
                break
            answer = process_query(df, query)
            print(answer)
            logger.info(f"Query: {query} | Answer: {answer}")
        
        logger.info("Query session ended.")
    else:
        logger.warning("EDA process failed due to errors in loading the dataset.")

if __name__ == "__main__":
    main()