# 🏥 Medical Insurance Cost Predictor

This project predicts medical insurance costs based on user input features such as age, gender, BMI, number of children, smoking status, and region.

## 📌 Features

- Performs EDA and data preprocessing on the insurance dataset
- Trains a regression model to predict insurance charges
- Saves the trained model (`medical_bill.pkl`)
- Provides an interactive UI inside Jupyter Notebook using `ipywidgets` for live predictions

## 💻 How to Run

1. Clone the repository or download the files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```
   jupyter notebook prediction.ipynb
   ```
4. Run all cells and use the interactive section at the end to get predictions.

## 🧠 Model Info

The model is trained using `scikit-learn`. The input features include:
- Age
- Gender
- BMI
- Number of children
- Smoking status
- Region (one-hot encoded)

## 📂 Files

- `prediction.ipynb`: Main notebook with all steps from EDA to prediction
- `medical_bill.pkl`: Saved trained model
- `requirements.txt`: All necessary libraries
- `README.md`: You're reading it!

## ✨ Example

After running the notebook, you’ll see sliders and dropdowns like this:

```python
# Example:
Age: 30
Gender: male
BMI: 25.0
Children: 1
Smoker: no
Region: northeast

# Output:
Predicted Insurance Cost: $4,258.67
```

## 🧪 Model Comparison

| Model                     | R² Score | MAE         | RMSE        | CV R² Score | CV R² Std |
|--------------------------|----------|-------------|-------------|-------------|-----------|
| Ridge Regression         | 0.8497   | 4597.65     | 2708.53     | 0.8442      | 0.0379    |
| Lasso Regression         | 0.8491   | 4606.98     | 2723.60     | 0.8441      | 0.0379    |
| Linear Regression        | 0.8490   | 4609.40     | 2728.71     | 0.8441      | 0.0380    |
| Gradient Boosting        | 0.8474   | 4633.29     | 2463.84     | 0.8510      | 0.0371    |
| Random Forest            | 0.8386   | 4764.85     | 2581.11     | 0.8394      | 0.0465    |
| Support Vector Regression| 0.7127   | 6357.65     | 2755.18     | 0.7048      | 0.0486    |

✅ **Best Model Chosen**: `Ridge Regression` — It offered the best combination of R² score, MAE, and RMSE.  
💾 Saved as: `medical_bill.pkl`

