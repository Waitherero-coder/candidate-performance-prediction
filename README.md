# 📊 Candidate Performance Prediction Simulation

## 📌 Project Overview

This project simulates a real-world **recruitment analytics problem**:  

> Can we predict a candidate’s future job performance based on pre-employment evaluation metrics?

Using synthetic data, we **build, train, and evaluate a machine learning model** to classify candidates into performance categories (Low, Average, High).  

This project demonstrates:

- Data simulation & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Machine learning model training & evaluation  
- Analytical interpretation of results  

The **main notebook** (`exploration.ipynb`) is the centerpiece, showing the **entire workflow from raw data to actionable insights**.  

---

## 🎯 Problem Statement

Organizations invest significant resources in recruitment. Predicting candidate performance **before hiring** can improve decision-making and reduce turnover costs.

This project aims to:

> Predict whether a candidate will be a **High**, **Average**, or **Low** performer based on structured evaluation data.

---

## 🧾 Dataset (Simulated)

Since real hiring data is confidential, this project generates **synthetic candidate data** with the following features:

| Feature | Description |
|----------|-------------|
| education_level | Candidate’s highest level of education |
| years_experience | Number of years in relevant field |
| technical_test_score | Technical assessment score (0–100) |
| interview_score | Interview evaluation score (0–100) |
| personality_score | Behavioral/personality assessment score |
| previous_rating | Performance rating from past role |
| performance_label | Target variable (High / Average / Low) |

**Dataset size:** 800 simulated candidates  

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## 📊 Machine Learning Approach

1. Data Generation  
2. Data Cleaning & Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Model Training  
   - Logistic Regression (Multiclass)  
5. Model Evaluation  
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - Feature Influence Analysis (Coefficients & Heatmap)  
6. Example Prediction for New Candidates  

---

## 📁 Project Structure

```text
Candidate-Performance-Prediction/
│
├── data/
│   └── simulated_candidates.csv        # Synthetic candidate dataset
│
├── notebooks/
│   └── exploration.ipynb               # Main analysis & ML notebook
│   └── .vscode/                        # VSCode settings
│   └── .ipynb_checkpoints/             # Jupyter checkpoints
│
├── src/
│   └── data_generation.py              # Code for generating synthetic dataset
│
└── README.md                           # Project documentation
```
--- 

Notes:
- data/ contains the synthetic dataset.
- notebooks/ contains the main notebook, showcasing the complete workflow.
- src/ includes supporting scripts, like data generation.
  
---

📈 Expected Outcomes
- Identify which candidate attributes most strongly predict performance
- Understand model predictions and feature influence
- Simulate a recruitment analytics workflow end-to-end

---

## 🚀 Future Improvements

- Add cross-validation  
- Hyperparameter tuning  
- Feature importance visualization  
- Deployment as a simple web app  

---

## 👩🏽‍💻 Author

**Elizabeth Waithereru Kalondu**  
Data Science Student
