# ❄️ HVAC Cooling Load Prediction

### Machine Learning for Mechanical Engineering

An interactive machine learning dashboard designed to predict building cooling loads based on architectural and geometric characteristics.

## 📌 Project Overview

This project explores the application of Machine Learning in HVAC engineering.

The objective is to estimate building cooling loads using a trained regression model and visualize prediction performance through an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Predict building cooling loads using Machine Learning.
- Analyze the influence of building characteristics.
- Evaluate prediction accuracy.
- Compare multiple regression algorithms.
- Develop an interactive engineering dashboard.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Random Forest Regressor
- Linear Regression
- Decision Tree Regressor
- Pandas and NumPy
- Matplotlib
- Streamlit
- Google Colab
- GitHub

## 🏢 Input Parameters

The model uses eight building characteristics:

1. Relative Compactness
2. Surface Area
3. Wall Area
4. Roof Area
5. Overall Height
6. Orientation
7. Glazing Area
8. Glazing Area Distribution

## 🤖 Machine Learning Model

The primary prediction model is a Random Forest Regressor trained on building energy-efficiency data.

The dataset was divided into training and testing sets using an 80/20 split.

### Model Performance

| Metric | Result |
|---|---|
| R² Score | 0.968 |
| Mean Absolute Error (MAE) | 1.06 |

These metrics represent performance on the held-out test dataset, not guaranteed accuracy on real-world buildings.

## 📊 Dashboard Features

- Interactive building parameter inputs
- Cooling load prediction
- Model performance indicators
- Feature importance visualization
- Actual vs. predicted cooling load comparison
- Prediction error analysis
- Machine learning model comparison
- Professional dark-themed interface

## 🚀 Run the Application

Clone the repository:

```bash
git clone https://github.com/hasanalomari707-pixel/HVAC-Cooling-Load-Prediction.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
HVAC-Cooling-Load-Prediction/
├── app.py
├── hvac_model.pkl
├── comparison.csv
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## ⚠️ Engineering Limitations

This project is intended for educational and preliminary predictive analysis.

The machine learning model does not replace detailed HVAC load calculations or professional building energy simulation software.

Its predictions are limited by the dataset and input variables used during training.

## 🔮 Future Improvements

- Additional machine learning algorithms
- Hyperparameter optimization
- Cross-validation
- Climate and weather data integration
- Expanded engineering input parameters
- Advanced building energy analysis

## 👨‍💻 Project Category

Mechanical Engineering | HVAC | Machine Learning | Data Analytics

## 📸 Project Screenshots

The following screenshots demonstrate the HVAC Cooling Load Prediction dashboard, model predictions, and data visualizations.

## 📸 Project Screenshots

The following screenshots demonstrate the HVAC cooling load prediction dashboard, model predictions, and data visualizations.

### 1. HVAC Prediction Dashboard

![HVAC Dashboard](images/WhatsApp%20Image%202026-11-15%20at%207.20.07%20PM.jpeg)

### 2. Model Visualization

![Model Visualization](images/WhatsApp%20Image%202026-11-15%20at%207.20.10%20PM.jpeg)

### 3. Prediction Results

![Prediction Results](images/WhatsApp%20Image%202026-11-15%20at%207.20.49%20PM.jpeg)

### 4. Additional Analysis

![Additional Analysis](images/WhatsApp%20Image%202026-11-15%20at%207.21.03%20PM.jpeg)