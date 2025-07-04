import warnings
import os

# Suppress TensorFlow warnings only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import dash
from dash import dcc
from dash import html
import pandas as pd
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))
df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv") 
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']

data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(len(data)):
    new_data.loc[i, "Date"] = data.iloc[i]["Date"]
    new_data.loc[i, "Close"] = data.iloc[i]["Close"]

new_data = new_data.set_index('Date')
dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model=load_model("saved_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values 
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test=[]
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_closing_price=model.predict(x_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

train=new_data[:987]
valid=new_data[987:].copy()  # Create a proper copy to avoid the warning
valid['Predictions']=predicted_closing_price

df= pd.read_csv("./stock_data.csv")
    