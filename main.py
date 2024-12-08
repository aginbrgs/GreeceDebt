import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

#This code expects macroeconomic data containing Tax Revenue in % GDP, Social Expenditure in % GDP, Growth in % GDP and a dummy for two countries you want to compare

file_path = r"YourDataPath.csv" # Enter your Path here
df  = pd.read_csv(file_path)


                                                                #Here you should introduce your crash date
df_before2010=df[df['TIME_PERIOD']<2010]
df_after2010=df[df['TIME_PERIOD']>=2010]
df['Pre_Crisis'] = (df['TIME_PERIOD'] <= 2011).astype(int)      #Splitting the data at the crashdate and ensuring its int

y = df_before2010['Interest in % GDP']
X = df_before2010[['Growth','Tax Revenue in % GDP','Social Expenditure in % of GDP','dummy']]
X = sm.add_constant(X)
                                                                #Defining target variable and predictors
X_clean = X.replace([np.inf,-np.inf,np.nan])
X_clean = X_clean.dropna()
y_clean = y[X_clean.index]


ols_model = sm.OLS(y_clean, X_clean).fit()                      #Fitiing OLS

Autocor = sm.tsa.acf(ols_model.resid,fft=False,nlags=1)[1] #Calculate the autocorrelation in time series with a lag of 1
y_transformed = y_clean.diff().dropna() - Autocor * y_clean.shift(1).dropna() #De autocorrelated version of original time series 
X_transformed = X_clean.diff().dropna() - Autocor * X_clean.shift(1).dropna()

gls = sm.GLS(y_transformed,X_transformed).fit() #Fitting GLS
print(gls.summary())

