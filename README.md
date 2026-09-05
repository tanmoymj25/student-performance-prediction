# 🎓 Student Performance Prediction

A Machine Learning based web application that predicts a student's expected examination score using academic, personal, family-related, learning, and environmental factors.

The project demonstrates an end-to-end Machine Learning workflow including data analysis, data preprocessing, categorical encoding, feature scaling, model training, model evaluation, model selection, and deployment using Streamlit.

---

## 📌 Introduction

Student academic performance can be influenced by several factors such as study hours, attendance, previous scores, parental involvement, access to resources, motivation, tutoring, family income, teacher quality, and other personal and environmental factors.

This project uses Machine Learning techniques to estimate a student's expected examination score based on these input factors.

The trained Machine Learning model is integrated with an interactive Streamlit web application where users can enter student information and receive a predicted exam score along with visual analysis.

> **Note:** This project is developed for educational and demonstration purposes. The predictions are based on the available dataset and should not be considered definitive real-world academic assessments.

---

## 🚀 Features

- 🎓 Predicts expected student examination score
- 📝 Interactive student information form
- 📊 Displays predicted score visually
- 📈 Interactive score gauge
- 📉 Benchmark comparison chart
- 📋 Prediction profile visualization
- 🤖 Machine Learning model comparison
- ⚙️ Numerical feature preprocessing
- 🔤 Categorical feature encoding
- 📏 Feature scaling using StandardScaler
- 🧹 Missing value handling
- 💾 Trained model saved using Joblib
- 🌐 Interactive Streamlit web application
- ☁️ Deployed using Streamlit Community Cloud

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Plotly**
- **Streamlit**
- **Joblib**

---

## 📊 Dataset

The project uses the:

`StudentPerformanceFactors.csv`

dataset.

The dataset contains different academic, personal, family-related, and environmental factors related to student performance.

### Input Features

The model uses the following features:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Sleep Hours
- Previous Scores
- Motivation Level
- Internet Access
- Tutoring Sessions
- Family Income
- Teacher Quality
- School Type
- Peer Influence
- Physical Activity
- Learning Disabilities
- Parental Education Level
- Distance from Home
- Gender

### Target Variable

The target variable is:

`Exam_Score`

The model predicts the expected examination score based on the above input features.

---

## 👨‍👩‍👧 Family-Related Factors

The dataset does not contain a separate `Family_Support` feature.

However, family-related information is represented through features such as:

- Parental Involvement
- Family Income
- Parental Education Level

These features provide useful information about the student's family and educational environment.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Analysis
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Missing Value Handling
   ↓
Feature Scaling / Normalization
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving using Joblib
   ↓
Streamlit Application
   ↓
User Input
   ↓
Prediction
   ↓
Visualization
