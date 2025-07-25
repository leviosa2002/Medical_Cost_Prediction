# 🏥 Medical Insurance Cost Predictor

This project predicts medical insurance costs based on user input features such as age, gender, BMI, number of children, smoking status, and region.

## 📌 Features

- Performs EDA and data preprocessing on the insurance dataset
- Trains a regression model to predict insurance charges
- Saves the trained model (`insurance_prediction_model.pkl`)
- Provides an interactive Streamlit dashboard for visualizing data and making predictions
- Includes the original notebook analysis (`medical_cost_prediction.ipynb`)

## 💻 How to Run the Streamlit Dashboard

1. Clone the repository or download the files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Open your browser and navigate to the URL provided (typically http://localhost:8501)

## 🧠 Model Info

The models are trained using `scikit-learn`. The dashboard includes two models:

1. **Ridge Regression**: More stable, linear model with L2 regularization
2. **Random Forest**: More complex model that can capture non-linear relationships

Both models achieve around 85% accuracy (R² score) in predicting insurance charges.

Input features include:
- Age
- Gender
- BMI
- Number of children
- Smoking status
- Region
- Plus several engineered features like age-BMI interaction, etc.

## 📊 Dashboard Pages

The Streamlit dashboard includes the following pages:

1. **Home**: Overview of the project and dataset
2. **Data Exploration**: Interactive visualizations of the data
3. **Prediction**: Input your information to get a cost prediction
4. **Model Info**: Details about the machine learning models used

## 📂 Files

- `app.py`: Streamlit dashboard application
- `medical_cost_prediction.ipynb`: Detailed analysis notebook
- `insurance.csv`: Dataset with insurance information
- `models/insurance_prediction_model.pkl`: Saved trained model (generated on first run)
- `requirements.txt`: All necessary libraries
- `README.md`: You're reading it!

## ✨ Example Prediction

Using the Prediction tab in the dashboard, you can input values like:

```
Age: 30
Gender: male
BMI: 25.0
Children: 1
Smoker: no
Region: northeast

Predicted Insurance Cost: $4,258.67
```

## 🧪 Model Performance

| Model                     | R² Score | MAE         | RMSE        | CV R² Score | CV R² Std |
|--------------------------|----------|-------------|-------------|-------------|-----------|
| Ridge Regression         | 0.8497   | 4597.65     | 2708.53     | 0.8442      | 0.0379    |
| Random Forest            | 0.8497   | 4597.42     | 2426.33     | 0.8394      | 0.0465    |
| Linear Regression        | 0.8490   | 4609.40     | 2728.71     | 0.8441      | 0.0380    |
| Gradient Boosting        | 0.8474   | 4633.29     | 2463.84     | 0.8510      | 0.0371    |
| Lasso Regression         | 0.8491   | 4606.98     | 2723.60     | 0.8441      | 0.0379    |
| Support Vector Regression| 0.7127   | 6357.65     | 2755.18     | 0.7048      | 0.0486    |

✅ **Best Models Used**: `Ridge Regression` and `Random Forest` — They offered the best combination of R² score, MAE, and RMSE.

