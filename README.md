# ❤️ Heart Disease Prediction App

## 📌 Overview

This project is a Machine Learning-based web application that predicts the likelihood of heart disease based on patient medical data. It combines data science, model building, and deployment using Streamlit to provide real-time predictions in a user-friendly interface.

---

## 🎯 Objectives

* Predict heart disease using classification algorithms
* Compare multiple machine learning models
* Select the best model based on performance
* Deploy the model using an interactive web app

---

## 🧠 Machine Learning Models Used

The following models were trained and evaluated:

* Logistic Regression
* Random Forest Classifier ⭐ (Best Model)
* Support Vector Machine (SVM)
* Gradient Boosting
* XGBoost

---

## 🏆 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | ~71%     |
| Logistic Regression | ~68%     |
| SVM                 | ~68%     |
| XGBoost             | ~68%     |
| Gradient Boosting   | ~66%     |

👉 **Final Model Selected: Random Forest**

✔ Balanced precision and recall
✔ Stable cross-validation performance

---

## 📊 Features Used

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Max Heart Rate
* Exercise Induced Angina
* ST Depression
* ST Slope
* Number of Major Vessels
* Thalassemia

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Selection (optional)
4. Model Training & Comparison
5. Hyperparameter Tuning (GridSearchCV)
6. Model Evaluation
7. Model Selection
8. Model Saving (`.pkl`)
9. Deployment using Streamlit

---

## 🖥️ Streamlit Application

The application allows users to:

✔ Enter patient medical details
✔ Get instant prediction (High Risk / Low Risk)
✔ View probability scores
✔ See confidence level
✔ Understand risk factors through explanation

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Srinidhisn5/heart-disease-prediction.git
cd heart-disease-prediction
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py
├── heart_model.pkl
├── columns.pkl
├── heart_disease_dataset.csv
├── Heart disease prediction(classification).ipynb
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

## 💡 Key Highlights

✔ End-to-end ML project
✔ Multiple model comparison
✔ Hyperparameter tuning
✔ Real-time prediction system
✔ User-friendly interface
✔ Model interpretability (confidence + explanation)

---

## ⚠️ Disclaimer

This application is intended for educational purposes only.
It should not be used as a substitute for professional medical advice.

---

## 📌 Future Improvements

* Improve accuracy using advanced techniques
* Add more datasets for better generalization
* Deploy on cloud (Streamlit Cloud / AWS)
* Add explainable AI (SHAP)
* Improve UI/UX

---

## 👨‍💻 Author

**Srinidhi SN**
GitHub: https://github.com/Srinidhisn5

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
