# 🏠 House Price Prediction using Linear Regression

This project predicts **house prices** based on various features such as square footage, number of bedrooms, bathrooms, year built, garage size, and neighborhood quality.  
It uses **Linear Regression** from scikit-learn and evaluates model performance using standard regression metrics.

---

## 📌 Features Used
- `Square_Footage`
- `Num_Bedrooms`
- `Num_Bathrooms`
- `Year_Built`
- `Garage_Size`
- `Neighborhood_Quality`

Target column:
- `House_Price`

---

## ⚙️ Workflow
1. **Import Libraries** → pandas, numpy, matplotlib, seaborn, sklearn  
2. **Data Preprocessing**
   - Load dataset into a DataFrame  
   - Standardize features using **StandardScaler** (scaled between `-1` and `1`)  
3. **Train-Test Split**
   - 80% training, 20% testing  
4. **Model Training**
   - Apply **Linear Regression**  
   - Fit model on training data  
5. **Prediction**
   - Predict on test data (`X_test`)  
6. **Model Evaluation**
   - MAE (Mean Absolute Error)  
   - MSE (Mean Squared Error)  
   - RMSE (Root Mean Squared Error)  
   - R² Score (Coefficient of Determination)  
7. **Visualization**
   - Histogram → Distribution of House Prices  
   - Scatter Plot → Predicted Price vs Actual Price  
8. **User Input Prediction**
   - Accepts user-defined feature values  
   - Predicts house price instantly  

---

## 📊 Visualizations
- **Distribution of House Prices**  
  - X-axis → House Price  
  - Y-axis → Number of Houses  
  - `bins = 30`  

- **Predicted vs Actual Price**  
  - X-axis → Actual Price  
  - Y-axis → Predicted Price  

---

## 🧮 Example User Input
```python
# Example input
X_new = [[2500, 4, 3, 2015, 2, 8]]
predicted_price = model.predict(X_new)
print("Predicted House Price:", predicted_price)
