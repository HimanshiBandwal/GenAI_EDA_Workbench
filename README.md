# GenAI_EDA_Workbench
GenAI EDA Workbench

A tool for exploratory data analysis (EDA) and fraud detection modeling on credit card transaction data, developed as of May 17, 2025.

# Overview

This project performs EDA on the creditcard.csv dataset, generates visualizations, supports natural language queries, and trains a fraud detection model using Random Forest with SMOTE.

# Features

Data Cleaning: Handles missing values and caps outliers using IQR.
Visualizations: Includes histograms, box plots, heatmaps, pair plots, and more.
Queries: Answers questions like “What are the top 5 features correlated with Class?”
Fraud Detection: Trains a Random Forest model to detect fraud.

# Dataset

The creditcard.csv dataset is not included due to privacy concerns.

How to Obtain the Dataset
Download creditcard.csv from Kaggle: Credit Card Fraud Detection Dataset
Place it in the data/ directory.

# Installation
Clone the repository:
git clone https://github.com/<your-username>/GenAI_EDA_Workbench.git
cd GenAI_EDA_Workbench
Create a virtual environment and install dependencies:
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
Download NLTK data:
import nltk
nltk.download('punkt')

Usage

# Run the script:

python scripts/eda_workbench.py --file data/creditcard.csv --output-dir outputs
Outputs will be saved in outputs/.
Enter the query session to ask questions like “What is the fraud rate?” Type exit to quit.

# Project Structure
scripts/eda_workbench.py: Main script for EDA and modeling.
data/: Directory for the dataset (not included).
outputs/: Directory for generated outputs.
requirements.txt: List of dependencies.
.gitignore: Excludes sensitive and generated files.
LICENSE: MIT License.

License

This project uses the MIT License. See the LICENSE file for details.

# Acknowledgments
Dataset provided by MLG, ULB via Kaggle.
Built with Python, pandas, scikit-learn, and NLTK.
