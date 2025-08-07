# 🤖 Human Activity Recognition via Time Series Classification

A full pipeline for classifying physical activities using multivariate time series data from wearable sensors.  
This project applies **statistical feature engineering**, **logistic regression**, and **penalized classifiers** to accurately predict human motion types from raw sensor data.

---

## 🧠 Overview

This project explores supervised time-series classification using a publicly available human activity dataset.  
I built a feature extraction framework, implemented binary and multi-class classifiers, and conducted robust model evaluation using real-world sensor signals.

---

## 📂 Dataset

- **Source:** [UCI AReM Dataset](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+(AReM))
- **Description:** Sensor readings (6 time series per sample) from human subjects performing 7 activities
- **Each instance:** 480-length values × 6 sensor channels (avg/var rss12, rss13, rss23)

---

## 🔨 Project Highlights

### 1️⃣ Time-Domain Feature Engineering
Extracted statistical features from each signal:
- `min`, `max`, `mean`, `median`, `std`, `Q1`, `Q3`
- Created a feature matrix with 88 samples × 42 features
- Computed standard deviations & 90% bootstrap confidence intervals for each feature

### 2️⃣ Binary Classification – Bending vs Other Activities
- Implemented logistic regression models to separate "bending" from other activities
- Visualized decision boundaries and class separation using scatter plots
- Introduced feature selection via:
  - Manual pruning (domain-based)
  - Recursive Feature Elimination (RFE)
- Addressed **class imbalance** via case-control sampling
- Evaluated with:
  - Confusion matrix
  - ROC curve & AUC
  - Cross-validation accuracy

### 3️⃣ Penalized Classification with L1 Regularization
- Built sparse logistic regression models via L1 penalty
- Automatically pruned irrelevant features
- Tuned both `l` (segment granularity) and `λ` (penalty strength) via cross-validation

### 4️⃣ Multiclass Activity Prediction
- Used multinomial logistic regression with L1 penalty
- Compared performance against Gaussian and Multinomial Naïve Bayes
- Analyzed multiclass confusion matrix and ROC macro/micro AUC

---

## 🚀 Key Results

| Task                        | Best Model                     | Insights                                  |
|-----------------------------|--------------------------------|-------------------------------------------|
| Binary Classification       | Logistic + RFE                 | High separation with 3 selected features  |
| Penalized Logistic Regression | L1-LogReg + Grid Search     | Auto feature selection + solid accuracy   |
| Multiclass Classification   | L1-Multinomial Regression      | Outperformed Naïve Bayes                  |

---

## 📈 Tools & Tech

- **Languages:** Python (NumPy, Pandas)
- **Modeling:** Scikit-learn (LogReg, RFE, Naïve Bayes)
- **Visualization:** Matplotlib, Seaborn
- **Stats:** Bootstrapping, Stratified CV, ROC/AUC
- **Engineering:** Custom time-series parsing & feature pipeline

---

## 💡 Learnings

- Extracting time-domain features is a practical approach for time-series problems
- Penalized models like **L1-logistic regression** simplify feature sets effectively
- Addressing **class imbalance** is crucial for reliable model performance
- Stratified cross-validation prevents skewed evaluations in sparse datasets

---

## 🧪 Future Improvements

- Try CNN or LSTM architectures on raw sensor sequences  
- Add domain-specific frequency features (FFT, spectral entropy)  
- Build a real-time mobile prototype for activity detection  

---

## 📬 Let's Connect

If this project resonates with your interests or you’d like to collaborate:

- 📧 [ngovinda@usc.edu](mailto:ngovinda@usc.edu)
- 🔗 [LinkedIn](https://linkedin.com/in/nikhil-govindaraju)

---

> _"Time series tells a story — your job is to listen, extract, and predict."_  
