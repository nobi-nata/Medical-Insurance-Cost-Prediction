import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import numpy as np 
import time
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib


data = pd.read_csv("insurance.csv")
data['sex']=data['sex'].map({'female':0,'male':1})
data['medical']=data['medical'].map({'yes':1,'no':0})
data['region']=data['region'].map({'southeast':2,'southwest':1,'northeast':4,'northwest':3})


X= data.drop(['charges'],axis=1)
Y=data['charges']


st.title("Medical Insurance Cost Predictor")
st.image("https://miro.medium.com/max/700/1*WoVYSG5nZbErqzgy47tWBQ.jpeg",width = 450)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(data)
    
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    val = st.slider("Filter data using years",0,20)
    data = data.loc[data["children"]>= val]
    if graph == "Non-Interactive":
        plt.figure(figsize = (10,5))
        plt.scatter(data["children"],data["charges"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("charges")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout =go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range =[0,210000])
        )
        fig = go.Figure(data=go.Scatter(x=data["children"], y=data["charges"], mode='markers'),layout = layout)
        st.plotly_chart(fig)
    
if nav == "Prediction":
    st.header("Know Medical Insurance Cost")
    age = st.number_input("Enter your Age",0.0,100.0,step=0.25)
    gen = st.radio("Select your Gender",["male","female"])
    bmi = st.number_input("Enter your BMI",0.0,50.0)
    child = st.number_input("Enter number of Children",0.0,20.0)
    med = st.radio("Are you a Smoker?",["yes","no"])
    reg = st.radio("Select your Region",["northwest","northeast","southeast","southwest"])    
    
    

    
    if gen == 'female':
        ling = 0
    else:
        ling = 1

    if med == 'yes':
        medi = 1
    else:
        medi = 0

    if reg == 'southeast':
        area = 2
    elif reg == 'southwest':
        area = 1
    elif reg == 'northeast':
        area = 4
    else:
        area = 3

    data = {'age':age,'sex':ling,'bmi':bmi,'children':child,'medical':medi,'region':area}
    df = pd.DataFrame(data,index=[0])

    gr = GradientBoostingRegressor()
    gr.fit(X,Y)

    joblib.dump(gr,'model_joblib_gr')
    model = joblib.load('model_joblib_gr')
    pred = model.predict(df)

    # val = np.array(val).reshape(1,-1)
    # pred =model.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Predicted Medical Insuurance Cost is : {pred} ")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    with st.form(key="my_form" , clear_on_submit=True):
        age = st.number_input("Enter your Age",0.0,100.0,step=0.25)
        gen = st.radio("Select your Gender",["male","female"])
        bmi = st.number_input("Enter your BMI",0.0,50.0)
        child = st.number_input("Enter number of Children",0.0,20.0)
        med = st.radio("Are you a Smoker?",["yes","no"])
        reg = st.radio("Select your Region",["northwest","northeast","southeast","southwest"])
        sal = st.number_input("Enter Insurance Cost",0.0,1000000.0,step = 1000.0)
        sub = st.form_submit_button("Submit")

        if sub:
            prg = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                prg.progress(i+1)
            to_add = {"age":[age],"sex":[gen],"bmi":[bmi],"children":[child],"medical":[med],"region":[reg],"charges":[sal]}
            to_add = pd.DataFrame(to_add)
            to_add.to_csv("insurance.csv",mode='a',header = False,index= False)
            st.success("Submitted")
            
           





# x = np.array(data['children']).reshape(-1,1)
# lr = LinearRegression()
# lr.fit(x,np.array(data['charges']))

# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# lr = LinearRegression()
# lr.fit(X_train,Y_train)
# svm = SVR()
# svm.fit(X_train,Y_train)
# rf = RandomForestRegressor()
# rf.fit(X_train,Y_train)
# gr = GradientBoostingRegressor()
# gr.fit(X_train,Y_train)
# Y_pred1 = lr.predict(X_test)
# Y_pred2 = svm.predict(X_test)
# Y_pred3 = rf.predict(X_test)
# Y_pred4 = gr.predict(X_test)

# df1 = pd.DataFrame({'Actual':Y_test,'Lr':Y_pred1,'svm':Y_pred2,'rf':Y_pred3,'gr':Y_pred4})

# plt.subplot(221)
# plt.plot(df1['Actual'].iloc[0:11],label='Actual')
# plt.plot(df1['Lr'].iloc[0:11],label='Lr')
# plt.legend()

# plt.subplot(222)
# plt.plot(df1['Actual'].iloc[0:11],label='Actual')
# plt.plot(df1['svm'].iloc[0:11],label='svr')
# plt.legend()

# plt.subplot(223)
# plt.plot(df1['Actual'].iloc[0:11],label='Actual')
# plt.plot(df1['rf'].iloc[0:11],label='rf')
# plt.legend()

# plt.subplot(224)
# plt.plot(df1['Actual'].iloc[0:11],label='Actual')
# plt.plot(df1['gr'].iloc[0:11],label='gr')
# plt.legend()

# plt.tight_layout()
