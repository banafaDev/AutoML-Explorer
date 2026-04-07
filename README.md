# AutoML Explorer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-00A3E0?style=flat&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**Upload → Clean → Train → Download**

**Automated Machine Learning in Minutes**

[🚀 Live Demo](https://huggingface.co/spaces/ahhmedgr/automl-explorer) • [📧 Contact](mailto:ahhmedgr@gmail.com) • [💼 LinkedIn](https://www.linkedin.com/in/ahmed-banafi-4b5034313/)

</div>

---

## 📖 Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Model Training Process](#-model-training-process)
- [Evaluation & Export](#-evaluation--export)
- [Advanced Features](#-advanced-features)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 About

**AutoML Explorer** is a production-ready automated machine learning platform that eliminates the complexity of traditional ML workflows. Built with Streamlit and powered by PyCaret, it automatically handles data preprocessing, model selection, hyperparameter tuning, and evaluation—all through an intuitive web interface.

Whether you're a data scientist looking to accelerate model development or a domain expert without ML expertise, AutoML Explorer provides enterprise-grade machine learning capabilities without writing a single line of code.

### Why AutoML Explorer?

- **Zero Code Required** - Complete ML pipeline through visual interface
- **Production Ready** - Download trained models ready for deployment
- **Intelligent Automation** - PyCaret compares 15+ algorithms automatically
- **Full Control** - Manual override for every preprocessing decision
- **Educational** - Learn ML best practices through guided workflow

---

## ✨ Key Features

### 🔍 **Intelligent Data Analysis**
- **Automatic Type Detection** - Identifies numeric, categorical, and boolean columns
- **Duplicate Detection** - Finds and removes duplicate rows
- **Missing Value Analysis** - Comprehensive null value reporting per column
- **Data Quality Checks** - Pre-flight validation before training
- **Statistical Summary** - Descriptive statistics for all features

### 🛠️ **Advanced Data Preprocessing**

#### Missing Value Handling
- **Categorical Features**:
  - Mode imputation (most frequent value)
  - Fill with custom text (e.g., "Unknown")
  - Drop rows with missing values
  - Per-column method selection
  
- **Numerical Features**:
  - Mean imputation (with intelligent rounding for integers)
  - Median imputation (robust to outliers)
  - Mode imputation
  - Drop rows option
  - Automatic type preservation (int vs float)

#### Feature Engineering
- **Type Conversion** - Convert columns between text and numeric
- **Outlier Removal** - IQR-based outlier detection and removal (regression)
- **Feature Normalization** - 4 methods:
  - Z-score (standard scaling)
  - Min-Max (0-1 scaling)
  - MaxAbs (-1 to 1 scaling)
  - Robust (outlier-resistant)

#### Categorical Encoding
- **Label Encoding** - Custom ordinal mapping with manual value assignment
- **One-Hot Encoding** - For nominal categories
- **Intelligent Defaults** - Auto-detects best encoding method
- **Target Encoding Support** - Preserves target labels for confusion matrix

### 🧠 **Automated Model Training**

#### PyCaret Integration
- **15+ Algorithms** compared automatically:
  - **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees, Decision Tree, KNN, Naive Bayes, SVM, Ridge, Ada Boost, Gradient Boosting, QDA, LDA
  - **Regression**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees, Gradient Boosting, Ada Boost, Decision Tree, KNN, SVR, Huber
  
- **Smart Configuration**:
  - Automatic imbalance detection and handling
  - Configurable cross-validation folds (2-10)
  - Adjustable train/test split (10-90%)
  - Multiple config comparison (Default, High CV, Outlier-removed)
  
- **Intelligent Tuning**:
  - Only tunes the best model from comparison
  - Compares tuned vs original performance
  - Keeps better performing version
  - Full tuning history tracking

### 📊 **Comprehensive Model Evaluation**

#### Classification Metrics
- **Core Metrics**:
  - Accuracy (overall correctness)
  - Precision (positive prediction accuracy)
  - Recall (sensitivity, true positive rate)
  - F1 Score (harmonic mean of precision/recall)
  - AUC-ROC (area under curve)
  
- **Advanced Analytics**:
  - Confusion Matrix with percentage display
  - Per-class precision/recall/F1
  - Support counts per class
  - ROC Curve (binary classification)
  - Original label preservation for interpretability

#### Regression Metrics
- **Performance Indicators**:
  - R² Score (coefficient of determination)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  
- **Visualization**:
  - Learning Curve (train vs validation loss)
  - Actual vs Predicted scatter plot
  - Actual vs Predicted line plot
  - Residual analysis

#### Feature Importance
- Automatic extraction for tree-based models
- Horizontal bar charts for interpretability
- Ranked by importance score

### 🎓 **Overfitting Detection**
- **Automatic Diagnosis**:
  - Compares train vs test performance
  - Gap analysis with threshold alerts
  - Actionable recommendations
  - Three-tier severity (no issue, slight variance, severe overfitting)

- **Smart Suggestions**:
  - Increase cross-validation folds
  - Remove irrelevant features
  - Try simpler models
  - Enable regularization

### 📈 **Exploratory Data Analysis (EDA)**

- **Summary Statistics** - Styled tables with count, mean, std, min, max, quartiles
- **Distribution Plots** - Bar charts for categorical features
- **Correlation Heatmap** - Seaborn heatmap with custom dark theme
- **Scatter Plots** - Feature relationship visualization
- **All Features View** - Grid layout of all categorical distributions

### 🎨 **Professional UI/UX**

- **Custom Dark Theme** - Eye-friendly design with accent colors
- **Responsive Layout** - Works on desktop and tablet
- **Real-time Feedback** - Progress bars, status messages, alerts
- **Contextual Help** - Tooltips and explanations throughout
- **Sidebar Metrics** - Live dataset statistics
- **Styled Tables** - Color-coded, gradient backgrounds

### 📥 **Model Export**

- **Production-Ready Models**:
  - Trained on 100% of data (optional toggle)
  - Joblib serialization (.pkl format)
  - Includes preprocessing pipeline
  - Ready for inference

- **Sample Predictions**:
  - 15 random test samples
  - Side-by-side actual vs predicted
  - Accuracy/error metrics
  - Color-coded correctness (classification)

---

## 🛠 Technology Stack

### Core Framework
- **[Streamlit](https://streamlit.io/)** `1.x` - Interactive web application framework
  - Custom CSS theming
  - Session state management
  - File upload handling
  - Dynamic UI components

### Machine Learning
- **[PyCaret](https://pycaret.org/)** `3.x` - Low-code ML library
  - Automated model comparison
  - Preprocessing pipeline
  - Model finalization
  - Experiment tracking

- **[scikit-learn](https://scikit-learn.org/)** `1.x` - ML algorithms and utilities
  - Classification algorithms
  - Regression algorithms
  - Preprocessing tools
  - Model evaluation metrics

### Data Processing
- **[Pandas](https://pandas.pydata.org/)** `2.x` - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** `1.x` - Numerical computing

### Visualization
- **[Matplotlib](https://matplotlib.org/)** `3.x` - Static plotting
- **[Seaborn](https://seaborn.pydata.org/)** `0.x` - Statistical data visualization

### File Handling
- **[openpyxl](https://openpyxl.readthedocs.io/)** - Excel (.xlsx) support
- **[xlrd](https://xlrd.readthedocs.io/)** - Legacy Excel (.xls) support
- **[joblib](https://joblib.readthedocs.io/)** - Model serialization

### Deployment
- **[Hugging Face Spaces](https://huggingface.co/spaces)** - Cloud hosting platform
- **Docker** - Containerization for reproducibility

---

## 🔄 How It Works

### **Step 01: Upload Dataset**
- Drag and drop CSV or Excel files
- Automatic format detection
- Dataset preview with all columns
- Supports files with any number of rows/columns

### **Step 02: Data Inspection**
- **Duplicate Detection**: Identifies and removes duplicate rows
- **Column Type Review**: Shows current dtypes with sample values
- **Type Conversion**: Convert between text and numeric
- **Missing Value Analysis**: Per-column null counts

### **Step 03: Data Cleaning**
- **Target Column Selection**: Choose prediction target
- **Remove Irrelevant Columns**: Drop ID, name, or unused features
- **Categorical Imputation**: Handle missing text values with mode, "Unknown", or row deletion
- **Numerical Imputation**: Handle missing numbers with mean, median, mode, or row deletion

### **Step 04: Exploratory Analysis**
- View summary statistics for all features
- Analyze distributions with bar charts
- Explore feature correlations with heatmap
- Examine relationships with scatter plots

### **Step 05: Feature Encoding**
- **Label Encoding**: Assign custom numeric values to ordered categories
- **One-Hot Encoding**: Create binary columns for nominal features
- **Validation**: Ensures all categorical features are encoded before training

### **Step 06: Task Configuration**
- **Automatic Detection**: Classification vs Regression based on target
- **Manual Override**: Change task type if needed
- **Target Analysis**: Shows unique value count and ratio

### **Step 07: Training Settings**
- **Cross-Validation Folds**: 2-10 folds (default: 5)
- **Test Set Size**: 10-90% (default: 20%)
- **Training Mode**: Fast (quick screening) or Thorough (deep evaluation)
- **Advanced Options**:
  - Feature normalization toggle
  - Normalization method selection
  - Outlier removal (regression only)
  - Train on full dataset option

### **Step 08: Model Training**
- PyCaret compares all available models
- Ranks by F1/Accuracy (classification) or R² (regression)
- Tests multiple configurations if Thorough mode enabled
- Real-time progress updates

### **Step 09: Model Selection**
- View top 3 models with metrics
- Recommended model highlighted
- Manual selection option
- Detailed hyperparameters displayed

### **Step 10: Evaluation**
- Comprehensive metrics dashboard
- Confusion matrix (classification)
- Learning curves (regression)
- Actual vs Predicted plots
- Feature importance charts
- Overfitting analysis
- Sample predictions on test set

### **Step 11: Export**
- Download trained model as .pkl file
- Model trained on full dataset (optional)
- Ready for production deployment

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/banafaDev/automl-explorer.git
cd automl-explorer
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open browser**
Navigate to `http://localhost:8501`

### Docker Deployment

```bash
docker build -t automl-explorer .
docker run -p 7860:7860 automl-explorer
```

---

## 📚 Usage Guide

### Example Workflow: Classification

1. **Upload** a CSV with customer data (e.g., `churn.csv`)
2. **Select** "Churn" as target column
3. **Remove** CustomerID column
4. **Handle** missing values:
   - Gender → Mode
   - Age → Median
5. **Encode** categorical features:
   - Gender → Label Encoding (Male=0, Female=1)
   - City → One-Hot Encoding
6. **Train** with:
   - Fast mode
   - 50 iterations
7. **Review** top 3 models (e.g., XGBoost, Random Forest, LightGBM)
8. **Evaluate** best model:
   - Accuracy: 0.92
   - F1: 0.89
   - Confusion Matrix shows good balance
9. **Download** trained model

### Example Workflow: Regression

1. **Upload** a CSV with house prices (e.g., `housing.csv`)
2. **Select** "Price" as target
3. **Remove** Address, ListingID columns
4. **Handle** missing values:
   - Bedrooms → Mode
   - SquareFeet → Median
5. **Remove** outliers in Price (IQR method)
6. **Encode**:
   - Neighborhood → One-Hot Encoding
7. **Train** with:
   - Thorough mode
   - Normalization enabled (zscore)
8. **Review** metrics:
   - R²: 0.87
   - RMSE: 45,000
   - Learning curve shows no overfitting
9. **Download** model

---

## 🧪 Model Training Process

### PyCaret Workflow

1. **Setup Phase**:
```python
from pycaret.classification import setup
experiment = setup(
    data=df,
    target='target_column',
    session_id=42,
    fix_imbalance=True,  # if detected
    fold=5,
    train_size=0.8,
    normalize=True,
    normalize_method='zscore',
    verbose=False
)
```

2. **Model Comparison**:
```python
from pycaret.classification import compare_models
models = compare_models(
    verbose=False,
    sort='F1',  # or 'R2' for regression
    n_select=15  # all available models
)
```

3. **Hyperparameter Tuning**:
```python
from pycaret.classification import tune_model
tuned_model = tune_model(
    best_model,
    optimize='F1',
    n_iter=50,
    search_algorithm='tpe',
    verbose=False
)
```

4. **Finalization**:
```python
from pycaret.classification import finalize_model
final_model = finalize_model(tuned_model)
```

---

## 📊 Evaluation & Export

### Metrics Explained

#### Classification
- **Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)
- **Precision**: Positive prediction accuracy TP/(TP+FP)
- **Recall**: Sensitivity, true positive rate TP/(TP+FN)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

#### Regression
- **R²**: Proportion of variance explained (1 = perfect, 0 = baseline)
- **MAE**: Mean absolute error (average prediction error)
- **RMSE**: Root mean squared error (penalizes large errors)
- **MSE**: Mean squared error (RMSE squared)

### Model Export Format

Downloaded `.pkl` files contain:
- Trained model object
- Preprocessing pipeline (scaling, encoding)
- Feature names
- Target encoder (if applicable)

### Using Exported Models

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.pkl')

# Prepare new data (same format as training)
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ... all features
})

# Make prediction
prediction = model.predict(new_data)
print(f"Prediction: {prediction[0]}")
```

---

## 🎛 Advanced Features

### Imbalanced Data Handling
- Automatic detection when minority class < 30%
- SMOTE oversampling for training
- Stratified cross-validation
- Weighted F1 score for evaluation

### Overfitting Prevention
- Cross-validation with multiple folds
- Train/test split validation
- Automatic gap detection
- Actionable recommendations

### Model Comparison
- Side-by-side metric comparison
- Highlighted best performer
- Detailed hyperparameters
- One-click model switching

### Custom Preprocessing
- Per-column imputation strategy
- Manual type conversion
- Custom label encoding values
- Flexible normalization options

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.

---

## 📧 Contact

**Ahmed Banafa**

- 📧 Email: [ahhmedgr@gmail.com](mailto:ahhmedgr@gmail.com)
- 💼 LinkedIn: [Ahmed Banafa](https://www.linkedin.com/in/ahmed-banafi-4b5034313/)
- 🚀 Live Demo: [AutoML Explorer on Hugging Face](https://huggingface.co/spaces/ahhmedgr/automl-explorer)
- 📦 GitHub: [banafaDev/automl-explorer](https://github.com/banafaDev/automl-explorer)

---

## 🙏 Acknowledgments

- [PyCaret](https://pycaret.org/) - Amazing low-code ML library
- [Streamlit](https://streamlit.io/) - Fantastic web framework
- [Hugging Face](https://huggingface.co/) - Free hosting platform

---

<div align="center">

### ⭐ Star this repository if AutoML Explorer helped you!

**Made with ❤️ by Ahmed Banafa**

</div>
