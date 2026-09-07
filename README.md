🎓 Student Performance Prediction
---
A Machine Learning based web application that predicts a student's expected examination score using academic, personal, family-related, learning, and environmental factors.
The project demonstrates an end-to-end Machine Learning workflow including data analysis, data preprocessing, categorical encoding, feature scaling, model training, model evaluation, model selection, and deployment using Streamlit.
---
📌 Introduction
Student academic performance can be influenced by several factors such as study hours, attendance, previous scores, parental involvement, access to resources, motivation, tutoring, family income, teacher quality, and other personal and environmental factors.
This project uses Machine Learning techniques to estimate a student's expected examination score based on these input factors.
The trained Machine Learning model is integrated with an interactive Streamlit web application where users can enter student information and receive a predicted exam score along with visual analysis.
> **Note:** This project is developed for educational and demonstration purposes. The predictions are based on the available dataset and should not be considered definitive real-world academic assessments.
---
🚀 Features
🎓 Predicts expected student examination score
📝 Interactive student information form
📊 Displays predicted score visually
📈 Interactive score gauge
📉 Benchmark comparison chart
📋 Prediction profile visualization
🤖 Machine Learning model comparison
⚙️ Numerical feature preprocessing
🔤 Categorical feature encoding
📏 Feature scaling using StandardScaler
🧹 Missing value handling
💾 Trained model saved using Joblib
🌐 Interactive Streamlit web application
☁️ Deployed using Streamlit Community Cloud
---
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Plotly
Streamlit
Joblib
---
📊 Dataset
The project uses the `StudentPerformanceFactors.csv` dataset.
The dataset contains different academic, personal, family-related, learning, and environmental factors related to student performance.
Input Features
Hours Studied
Attendance
Parental Involvement
Access to Resources
Extracurricular Activities
Sleep Hours
Previous Scores
Motivation Level
Internet Access
Tutoring Sessions
Family Income
Teacher Quality
School Type
Peer Influence
Physical Activity
Learning Disabilities
Parental Education Level
Distance from Home
Gender
Target Variable
`Exam_Score`
---
👨‍👩‍👧 Family-Related Factors
The dataset does not contain a separate `Family_Support` feature.
Family-related information is represented through:
Parental Involvement
Family Income
Parental Education Level
---
🔄 Machine Learning Workflow
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
```
---
🧹 Data Preprocessing
Median imputation for numerical features
Most-frequent imputation for categorical features
StandardScaler for numerical features
OneHotEncoder for categorical features
ColumnTransformer for combining preprocessing steps
---
🤖 Machine Learning Models
1. Linear Regression
Linear Regression was used to predict the examination score.
2. Decision Tree Regression
Decision Tree Regressor was trained with a maximum depth of 5.
The model with the highest R² Score was selected as the best-performing model.
---
📈 Model Evaluation
The models were evaluated using MAE, RMSE, and R² Score.
Model	R² Score	MAE	RMSE
Linear Regression	76.96%	0.45	1.80
Decision Tree Regression	54.44%	-	-
🏆 Best Model
Linear Regression
---
🌐 Streamlit Web Application
The trained model is integrated into an interactive Streamlit web application.
The application displays:
🎯 Predicted Exam Score
📊 Performance Category
📈 Interactive Score Gauge
📉 Benchmark Comparison
📋 Prediction Profile
🤖 Model Performance Comparison
Performance Categories
Predicted Score	Performance
Below 60	Needs Improvement
60–74	Average Performance
75–89	Good Performance
90–100	Excellent
---
💻 Installation
1. Clone the Repository
```bash
git clone https://github.com/tanmoymj25/student-performance-prediction.git
```
2. Navigate to the Project Folder
```bash
cd student-performance-prediction
```
3. Install Required Libraries
```bash
pip install -r requirements.txt
```
---
▶️ Run the Application
```bash
streamlit run app.py
```
---
📁 Project Structure
```text
student-performance-prediction/
│
├── app.py
├── model_training.py
├── preprocessing.py
├── data_analysis.py
├── StudentPerformanceFactors.csv
├── best_model.joblib
├── preprocessor.joblib
├── requirements.txt
└── README.md
```
File Description
`app.py` — Streamlit web application
`model_training.py` — Model training and evaluation
`preprocessing.py` — Data preprocessing pipeline
`data_analysis.py` — Exploratory Data Analysis
`StudentPerformanceFactors.csv` — Dataset
`best_model.joblib` — Saved best Machine Learning model
`preprocessor.joblib` — Saved preprocessing pipeline
`requirements.txt` — Required Python libraries
`README.md` — Project documentation
---
☁️ Deployment
The Streamlit application is deployed using Streamlit Community Cloud.
🔗 Live Demo
https://student-performance-prediction-7gmv5duershmnwjqinfyo.streamlit.app/
🔗 GitHub Repository
https://github.com/tanmoymj25/student-performance-prediction
---
🔮 Future Improvements
Add more Machine Learning algorithms
Improve prediction accuracy
Perform advanced hyperparameter tuning
Add more student performance features
Add additional interactive visualizations
Improve model interpretability
Add feature importance analysis
---
👨‍💻 Author
Tanmoy Majumder
GitHub: @tanmoymj25
---
📄 License
This project is created for educational and learning purposes.
