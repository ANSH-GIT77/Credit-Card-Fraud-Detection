# Credit-Card-Fraud-Detection
This project focuses on building a machine learning model to detect fraudulent transactions in credit card data. It uses supervised learning algorithms and data preprocessing techniques to identify anomalies and reduce false positives.
credit-card-fraud-detection/
├── data/ # Raw and cleaned datasets
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── src/ # Source code for model and preprocessing
│ ├── data_preprocessing.py
│ ├── model_training.py
│ └── utils.py
├── models/ # Saved models
├── outputs/ # Visualizations, confusion matrix, reports
├── app/ # Optional: Streamlit or Flask app
│ └── app.py
├── README.md # Project overview
├── requirements.txt # Python dependencies
└── fraud_detection.ipynb # Main notebook

yaml
Copy
Edit

---

##  Tech Stack & Tools

- Language: Python 3.x  
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`
  - `imbalanced-learn` (for SMOTE)
- **IDE:** Jupyter Notebook / VS Code  
- **Visualization:** seaborn, matplotlib  
- **Optional App:** Streamlit or Flask

---

##  Machine Learning Models Used

- Logistic Regression  
- Random Forest  
- XGBoost  
- Isolation Forest (optional for unsupervised approach)

---

##  Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in 2013.
- Total Records: 284,807
- Fraudulent Records: 492 (highly imbalanced)

---

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix

---

##  Installation

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
 Usage
bash
Copy
Edit
# Run preprocessing and training scripts
python src/data_preprocessing.py
python src/model_training.py
