---
title: "Logistic Regression, Random Forest and XGBoost performance comparison on a Bank Marketing Dataset"
date: 2025-04-15 14:11:00 +0100
categories: [machine learning]
tags: [statistical modelling, logistic regression, random forest, xgboost]
render_with_liquid: false
---

---

## 📝 Project Overview

This project analyzes a Portuguese bank's telemarketing campaign dataset to predict whether a client will subscribe to a long-term deposit. Using real-world customer data and multiple statistical models—including Logistic Regression, Random Forest, and XGBoost—the project showcases data preprocessing, exploratory data analysis, and predictive modeling to optimize campaign strategies.

---

## 📁 Dataset

- 📌 Source: [UCI Machine Learning Repository – Bank Marketing Dataset](https://doi.org/10.24432/C5K306)  
- 📅 Period: 2008–2013  
- 🔢 Observations: 4,100  
- 🎯 Target: Subscription to term deposit (`y`)

---

## 🔧 Tech Stack

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Matplotlib & Seaborn for visualizations
- Jupyter Notebook
- Markdown for documentation

---

## 🔍 Key Insights from EDA

- **Call Duration** is the strongest predictor of conversion, but not practical for real-time predictions.
- **Previous Contact (`pdays`)** and **number of contacts (`campaign`)** influence conversion—multiple contacts can lead to fatigue.
- **Economic Indicators** (employment rate, confidence index) correlate positively with success rates.
- **Box plots and correlation heatmaps** were used to understand distributions, outliers, and variable importance.

---

## 📈 Modeling Approach

Three classifiers were tested to solve the binary classification problem:

| Model              | Accuracy | ROC AUC |
|--------------------|----------|---------|
| Logistic Regression | 92.4%    | 95%     |
| Random Forest       | ~91%     | ~94%    |
| XGBoost             | ~91%     | 95%+    |

- Logistic Regression showed the best balance of simplicity and performance.
- XGBoost had slightly better AUC, making it a strong alternative for production.
- Low recall for positive cases (~45%) suggested the need for **threshold tuning or resampling** to better capture potential subscribers.

---

## ⚠️ Challenges

- **Imbalanced Dataset**: Only a minority of clients subscribe → ROC AUC and recall were prioritized over raw accuracy.
- **"Unknown" Categories**: Instead of imputation, these were treated as their own category for interpretability and better recall.

---

## ✅ Conclusion & Recommendations

- Use **Logistic Regression** as a baseline for campaign prediction.
- Consider economic indicators and call length in strategic planning.
- Future improvements: Use **cost-sensitive learning** or **SMOTE** for better recall on minority class.

---

## 📚 References

Key sources include peer-reviewed papers, technical blogs (e.g., Atlassian, Medium, Forbes), and project-specific research:

- Moro et al. (2014). *Decision Support Systems*
- Saito & Rehmsmeier (2015). *PLOS ONE*
- Yi (2024). *Atlassian Guides on Data Visualization*
- Ugenti (2024). *Forbes Council Post*

*See full reference list in the original report.*
"""

# Save to a markdown file
file_path = "/mnt/data/Bank_Marketing_Prediction.md"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)

file_path
