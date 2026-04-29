# ✈️ Airline Passenger Satisfaction Prediction

[![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Streamlit-red)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Best Model](https://img.shields.io/badge/🏆_Best_Model-Gradient_Boosting_96.3%25-brightgreen)](https://github.com)

An end-to-end **Machine Learning** project that predicts airline passenger satisfaction using survey data.  
Includes a fully interactive **Streamlit dashboard** with EDA, preprocessing, model training, and real-time prediction.

> 🎯 **Goal:** Help airlines understand key factors driving customer satisfaction and predict at-risk passengers.

---

# 📌 Problem Statement

Airlines need to identify dissatisfied passengers early to improve service quality.  
Using passenger feedback and flight details, we build classification models to predict whether a passenger will be **satisfied** or **neutral/dissatisfied**.

---

# 📊 Dataset

- **Source:** [Airline Passenger Satisfaction on Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Samples:** ~130,000 passengers
- **Features:** 24 (demographics, flight details, service ratings)
- **Target:** `satisfaction` (satisfied / neutral or dissatisfied)

| Feature Group | Examples |
|---------------|----------|
| Passenger info | Gender, Age, Customer Type, Type of Travel, Class |
| Flight details | Flight Distance, Departure/Arrival Delay |
| Service ratings (0–5) | Inflight wifi, Seat comfort, Online boarding, Cleanliness, etc. |

---

# 🧠 Methodology

## 1. Data Preprocessing
- Handled missing values (Arrival Delay median imputation)
- Removed outliers in Flight Distance (IQR method)
- Created new features:
  - `Total Delay` = Departure Delay + Arrival Delay
  - `Delay Level` (No/Medium/High Delay)
  - `Age Group` (binned)
  - `Class_Service_Avg` (average inflight service per class)

## 2. Feature Engineering
- One-hot encoding for categorical variables (`Gender`, `Customer Type`, `Type of Travel`, `Class`)
- Binary target column (`satisfaction_binary`)

## 3. Model Training (with Hyperparameter Tuning)
Four classifiers were trained on 80% training / 20% test split:

| Model | Accuracy | F1 Score | Key Hyperparameters |
|-------|----------|----------|----------------------|
| Logistic Regression | 0.869 | 0.85 | C=1.0 |
| Random Forest | 0.944 | 0.935 | n_estimators=100, max_depth=10 |
| **Gradient Boosting** ⭐ | **0.963** | **0.957** | n_est=100, lr=0.1, depth=5 |
| XGBoost | 0.962 | 0.956 | n_est=100, lr=0.1, depth=6 |

> ✅ **Best Model: Gradient Boosting Classifier** – highest accuracy (96.3%) and F1 (0.957)

## 4. Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling) was tested
- Class weight = 'balanced' for Logistic Regression / Random Forest
- XGBoost used `scale_pos_weight`

---

# 📈 Results & Insights

## Gradient Boosting – Classification Report
precision recall f1-score support
0 0.96 0.98 0.97 14573
1 0.97 0.95 0.96 11403
accuracy 0.96 25976

- **ROC-AUC:** 0.99 (excellent discrimination)
- Most important features:  
  `Online boarding`, `Inflight entertainment`, `Seat comfort`, `Class`, `Type of Travel`

> 💡 **Key Insight:** Business travelers and passengers who rate online boarding 4+ are **3x more likely** to be satisfied.

---

# 🖥️ Streamlit Dashboard Features

The app (`app.py`) provides a full interactive experience:

| Tab | Functionality |
|-----|----------------|
| 📊 Data Overview | View raw train/test data, shapes, missing values |
| 📈 EDA | Stacked bars, scatter plots, correlation heatmap, histograms |
| 🔧 Preprocessing | Apply feature engineering, handle missing/outliers |
| 🤖 Modeling | Choose model + feature selection + imbalance handling + train & evaluate |
| 🔮 Predict | Real-time prediction on new passenger data |

# How to Run Locally

```bash
git clone https://github.com/yourusername/airline-satisfaction.git
cd airline-satisfaction

pip install -r requirements.txt

streamlit run app.py
📁 Place train.csv and test.csv in the same folder (or upload via sidebar).

📁 Repository Structure
text

.
├── app.py                 
├── requirements.txt       
├── README.md              
├── train.csv              
└── test.csv    

📦 Requirements
Create requirements.txt with:

txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
xgboost==2.0.0
```
# 🔧 Detailed Steps to Run the Project

```bash
Step 1: Download the Dataset
Go to Kaggle Dataset Link
Download train.csv and test.csv

Step 2: Set Up Environment
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt

Step 3: Prepare Files
Place app.py in your project folder
Place train.csv and test.csv in the same folder
Create requirements.txt with the packages above

Step 4: Run the Application
streamlit run app.py

Step 5: Use the Dashboard
Upload Data: Use sidebar to upload train.csv and test.csv
Explore EDA: Check various plots to understand data patterns
Preprocess: Apply feature engineering (check the box)
Train Model: Select Gradient Boosting for best results
Predict: Enter passenger details and get satisfaction prediction
```

# 📊 Model Performance Comparison (Detailed)
```bash
1. Logistic Regression
text
Accuracy: 0.869
Classification Report:
               precision    recall  f1-score   support
           0       0.87      0.90      0.88     14573
           1       0.86      0.84      0.85     11403
    accuracy                           0.87     25976
2. Random Forest
text
Accuracy: 0.944
F1 Score: 0.935
Classification Report:
               precision    recall  f1-score   support
           0       0.94      0.96      0.95     14573
           1       0.94      0.93      0.94     11403
    accuracy                           0.94     25976
3. Gradient Boosting (BEST MODEL ⭐)
text
Accuracy: 0.963
F1 Score: 0.957
Classification Report:
               precision    recall  f1-score   support
           0       0.96      0.98      0.97     14573
           1       0.97      0.95      0.96     11403
    accuracy                           0.96     25976
4. XGBoost
text
Accuracy: 0.962
F1 Score: 0.956
Classification Report:
               precision    recall  f1-score   support
           0       0.96      0.98      0.97     14573
           1       0.97      0.94      0.96     11403
    accuracy                           0.96     25976
```

# 🚀 Future Improvements

Deploy on Streamlit Cloud / Hugging Face Spaces
Add SHAP explanations for predictions
Include deep learning (MLP) comparison
Add more feature selection methods (RFE, L1)
Create API endpoint for predictions

# 👩‍💻 Author
Your Name – Data Science Project
📧 your.email@example.com
🔗 GitHub
