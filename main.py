import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# This code expects macroeconomic data containing Tax Revenue in % GDP, Social Expenditure in % GDP, Growth in % GDP, and a dummy for two countries you want to compare

file_path = r"C:\Users\Borgsman\Desktop\project\combined_greece_spain.csv"
df  = pd.read_csv(file_path)
df = df.dropna()

# Splitting the data at the crash date and ensuring it's int
df['Post Crash'] = (df['TIME_PERIOD'] >= 2011).astype(int)
df['Treat'] = df['dummy']
df['Post_Treat'] = df['Treat'] * df['Post Crash']
df['Fiscal Balance'] = df['Social Expenditure in % of GDP'] * df['Tax Revenue in % GDP']

y = df['Interest in % GDP']
X = df[['Growth', 'Fiscal Balance', 'Post_Treat', 'Treat', 'Post Crash']]
#Scaling to ensure that all variables have the same impact
scaler_X = StandardScaler()
scaler_y = StandardScaler()


X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_clean = X_scaled.replace([np.inf, -np.inf, np.nan], np.nan).dropna()
y_scaled_clean = y_scaled[X_scaled_clean.index]
#Differencing to Remove Trends
y_transformed = pd.Series(y_scaled_clean).diff().dropna()
X_transformed = X_scaled_clean.diff().dropna()
#Adding Constant
X_transformed = sm.add_constant(X_transformed)

Did_Model = sm.GLS(y_transformed, X_transformed).fit()
print(Did_Model.summary())



