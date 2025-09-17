import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import locale

data = pd.read_csv('Machine Learning/house_price_regression_dataset.csv')

print(data.head(5))

house_details = ['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Garage_Size','Neighborhood_Quality']
scaler = StandardScaler()
df_scaled = data.copy()

df_scaled[house_details] = scaler.fit_transform(data[house_details])

X = df_scaled[house_details]
Y = df_scaled['House_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2 , random_state=42)

model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

print('Classification report')
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2  = r2_score(Y_test, y_pred)

# Print results
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# histogram
plt.figure(figsize = (10, 6))
plt.hist(df_scaled[['House_Price']], bins=30, color='skyblue')
plt.title('Distribution of House prices')
plt.xlabel('House Price')
plt.ylabel('Number of Houses')
plt.grid(True)
plt.show()

# Scatter 

plt.figure(figsize = (10, 6))
plt.scatter(Y_test, y_pred, color='blue', label='Actual Price')
plt.title('Predicted Price Vs Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid(True)
plt.show()

print('----Predict your result----')
try:
    Square_Footage = int(input('Enter total Area in sqrt: '))
    Num_Bedrooms = int(input('Enter No of Bedrooms: '))
    Num_Bathrooms = int(input('Enter No of Bathrooms: '))
    Year_Built = int(input('Enter house building year: '))
    Garage_Size = int(input('Enter No of Garage: '))
    Neighborhood_Quality = int(input('Enter Neighbourhood Quality (1 to 10): '))


    user_input_df = pd.DataFrame([{
        'Square_Footage': Square_Footage,
        'Num_Bedrooms': Num_Bedrooms,
        'Num_Bathrooms': Num_Bathrooms,
        'Year_Built': Year_Built,
        'Garage_Size': Garage_Size,
        'Neighborhood_Quality': Neighborhood_Quality
    }])

    user_input_scaled = scaler.fit_transform(user_input_df)
    prediction = model.predict(user_input_scaled)[0]

    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
    formatted_prediction = locale.format_string("%d", prediction, grouping=True)
    print(f"You House Price is around {formatted_prediction}")
except Exception as e:
    print("An Error Occour")