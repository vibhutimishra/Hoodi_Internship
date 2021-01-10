# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:19:24 2020

@author: Vibuthi mishra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

weather=pd.read_excel("weather for years.xlsx")
production=pd.read_excel("PDC_Db_latest.xlsx")

production=production.drop(["Plot Id"],axis=1)

plantDate=pd.DataFrame(production["Plant Date"])
harvestStart=pd.DataFrame(production["St Date"])
harvestEnd=pd.DataFrame(production["To Date"])

months=[]

#including time series to production data
for i in range(1874):
   plantDate["Plant Date"][i]=str(plantDate["Plant Date"][i])[0:11]
   harvestStart["St Date"][i]=str(harvestStart["St Date"][i])[0:11]
   py=str(plantDate["Plant Date"][i])[0:4]
   pm=str(plantDate["Plant Date"][i])[5:7]
   hy=str(harvestStart["St Date"][i])[0:4]
   hm=str(harvestStart["St Date"][i])[5:7]
   time=(int(hy)-int(py))*12+(int(hm)-int(pm))
   months.append(time)
   
months=pd.Series(months)
production['Time']=months


#finding the standard deviation
standardDeviation=pd.DataFrame(columns=['Max temp','Min temp','Rain','Sun hour','humidity','wind speed','cloud','Pressure'])
for i in range(1874):
    py=int(str(plantDate["Plant Date"][i])[0:4])
    pm=int(str(plantDate["Plant Date"][i])[5:7])
    hy=int(str(harvestStart["St Date"][i])[0:4])
    hm=int(str(harvestStart["St Date"][i])[5:7])

    start = weather[(weather['Year']==py) & (weather['Month']==pm)].index[0]
    end =  weather[(weather['Year']==hy) & (weather['Month']==hm)].index[0]
    if end>start:
        newdf=pd.DataFrame(weather[start:end+1])
        x=newdf.std()
        arr=[]
        for j in range(2,10):
            arr.append(x[j].round(5))
        
        newx=pd.DataFrame([arr],
                          columns=['Max temp','Min temp','Rain','Sun hour','humidity','wind speed','cloud','Pressure'])
        standardDeviation=standardDeviation.append(newx)
    else:
        arr=[0,0,0,0,0,0,0,0]
        newx=pd.DataFrame([arr],
                          columns=['Max temp','Min temp','Rain','Sun hour','humidity','wind speed','cloud','Pressure'])
        standardDeviation=standardDeviation.append(newx)


production.reset_index()
sd=standardDeviation
sd=sd.reset_index()
production=pd.concat([production,sd],axis=1)

#building model
production=production.drop(["Plant Date"],axis=1)
production=production.drop(["Ag Type"],axis=1)
production=production.drop(["St Date"],axis=1)
production=production.drop(["To Date"],axis=1)
production=production.drop(["index"],axis=1)
production=production.drop(["Disposal Name"],axis=1)

#droping those rows whose time is invalid
#production=production.drop(index)

#X=production.iloc[:,[True,True,True,True,True,False,True,True,True,True,True,True,True,True,True]].values
#y=production.iloc[:,5].values

#Labelencoder_X=LabelEncoder()
#X[:,4]=Labelencoder_X.fit_transform(X[:,4])

 
#Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

# Add new columns for soil data
ph=[]
Electric_conductivity=[]
Organic_carbon=[]
Nitrogen=[]
Phosphorus=[]
Potassium=[]
Sulphur=[]
Zn=[]
Fe=[]
Cu=[]
Mn=[]
Boron=[]

for i in range(1874):
    ph.append(7)
    Electric_conductivity.append(0)
    Organic_carbon.append(15)
    # 240- 480kg/ha
    Nitrogen.append(360)
    # 11 – 22 Kg/ha
    Phosphorus.append(16.5)
    # 110-280Kg/ha
    Potassium.append(195)
    Sulphur.append(0)
     # 1‑200 mg/kg
    Zn.append(100)
    Fe.append(0)
    Cu.append(0)
    Mn.append(0)
    Boron.append(50)
production["ph"]=ph;
production["Electric_conductivity(EC)"]=Electric_conductivity;
production["Organic_carbon(OC)"]=Organic_carbon;
production[" Nitrogen(N)"]= Nitrogen;
production["Phosphorus(P)"]=Phosphorus;
production["Potassium(K)"]=Potassium;
production["Sulphur(S)"]=Sulphur;
production["Zn"]=Zn;
production["Fe"]=Fe;
production["Cu"]=Cu;
production["Mn"]=Mn;
production["Boron"]=Boron;

production["yield"]=production["Tonns"]
X=production.iloc[:,1:27].values
y=production.iloc[:,27]

Labelencoder_X=LabelEncoder()
X[:,3]=Labelencoder_X.fit_transform(X[:,3])

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

forest_model = RandomForestRegressor(n_estimators=1000,random_state=1)
forest_model.fit(Xtrain, ytrain)
preds = forest_model.predict(Xtest)
print(mean_absolute_error(ytest, preds))


regr = LinearRegression()
regr.fit(Xtrain,ytrain)
pred2=regr.predict(Xtest)
print(mean_absolute_error(ytest, pred2))










