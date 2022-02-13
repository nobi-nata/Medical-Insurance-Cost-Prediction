from black import color_diff
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import numpy as np 
import time
# import seaborn as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib


data = pd.read_csv("insurance.csv")


st.title("Medical Insurance Cost Predictor")
st.image("https://miro.medium.com/max/700/1*WoVYSG5nZbErqzgy47tWBQ.jpeg",width = 450)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    st.header("Know Our Dataset")
    if st.checkbox("Show Dataset"):
        st.dataframe(data)
    
    graph = st.selectbox("Select",["Numeric","Graphic"])
    value = st.selectbox("Select Column",["Age","Gender","BMI","Children","Smoker","Region","Charges"])
    if graph == "Numeric":
        if value == "Age":
            a = data['age'].value_counts()
            st.table(a)
        elif value == "Gender":
            b = data['gender'].value_counts()
            st.table(b)
        elif value == "BMI":
            c = data['bmi'].value_counts()
            st.table(c)
        elif value == "Children":
            d = data['children'].value_counts()
            st.table(d)
        elif value == "Smoker":
            e = data['smoker'].value_counts()
            st.table(e)
        elif value == "Region":
            f = data['region'].value_counts()
            st.table(f)
        else:
            g = data['charges']
            st.table(g)
    if graph == "Graphic":
        if value == "Age":
            # Distribution of age value
            plt.figure(figsize=(6,6))
            arr = data['age']
            fig, ax = plt.subplots()
            ax.hist(arr, bins=20)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Density')
            st.pyplot(fig)        
        elif value == "Gender":
            # Gender column
            plt.figure(figsize=(6,6))
            arr = data['gender']
            fig, ax = plt.subplots()
            ax.hist(arr, bins='scott',color='green')
            plt.title('Gender Distribution')
            plt.xlabel('Gender')
            plt.ylabel('Density')
            st.pyplot(fig) 
            
        elif value == "BMI":
            # BMI distribution
            # Normal BMI range --> 18.5 to 24.9
            plt.figure(figsize=(6,6))
            arr = data['bmi']
            fig, ax = plt.subplots()
            ax.hist(arr, bins=20,color='grey')
            plt.title('BMI Distribution')
            plt.xlabel('BMI')
            plt.ylabel('Density')
            st.pyplot(fig) 
        elif value == "Children":
            # Childern column
            plt.figure(figsize=(6,6))
            arr = data['children']
            fig, ax = plt.subplots()
            ax.hist(arr, bins='sturges',color='orange')
            plt.title('Children Distribution')
            plt.xlabel('Children')
            plt.ylabel('Density')
            st.pyplot(fig) 
        elif value == "Smoker":
            # Smoker column
            plt.figure(figsize=(6,6))
            arr = data['smoker']
            fig, ax = plt.subplots()
            ax.hist(arr, bins='scott',color='purple')
            plt.title('Smoker Distribution')
            plt.xlabel('Smoker')
            plt.ylabel('Density')
            st.pyplot(fig) 
        elif value == "Region":
            # Region distribution
            plt.figure(figsize=(6,6))
            arr = data['region']
            fig, ax = plt.subplots()
            ax.hist(arr, bins='scott')
            plt.title('Region Distribution')
            plt.xlabel('Region')
            plt.ylabel('Density')
            st.pyplot(fig) 
        else:
            # Distribution of charge value
            plt.figure(figsize=(6,6))
            arr = data['charges']
            fig, ax = plt.subplots()
            ax.hist(arr,bins=[2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500,50000])
            plt.title('Price Distribution')
            plt.xlabel('Charges')
            plt.ylabel('Density')
            st.pyplot(fig) 

data['gender']=data['gender'].map({'female':0,'male':1})
data['smoker']=data['smoker'].map({'yes':1,'no':0})
data['region']=data['region'].map({'southeast':2,'southwest':1,'northeast':4,'northwest':3})


X= data.drop(['charges'],axis=1)
Y=data['charges']

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

    data = {'age':age,'gender':ling,'bmi':bmi,'children':child,'smoker':medi,'region':area}
    df = pd.DataFrame(data,index=[0])

    gr = GradientBoostingRegressor()
    gr.fit(X,Y)

    joblib.dump(gr,'model_joblib_gr')
    model = joblib.load('model_joblib_gr')
    pred = model.predict(df)

    # val = np.array(val).reshape(1,-1)
    # pred =model.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Predicted Medical Insurance Cost is : {pred} ")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    with st.form(key="my_form" , clear_on_submit=True):
        age = st.number_input("Enter your Age",0.0,100.0,step=0.25)
        gen = st.radio("Select your Gender",["male","female"])
        bmi = st.number_input("Enter your BMI",0.0,50.0,step=0.25)
        child = st.number_input("Enter number of Children",0,5,step= 1)
        med = st.radio("Are you a Smoker?",["yes","no"])
        reg = st.radio("Select your Region",["northwest","northeast","southeast","southwest"])
        sal = st.number_input("Enter Insurance Cost",0.0,1000000.0,step = 1000.0)
        sub = st.form_submit_button("Submit")

        if sub:
            prg = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                prg.progress(i+1)
            to_add = {"age":[age],"gender":[gen],"bmi":[bmi],"children":[child],"smoker":[med],"region":[reg],"charges":[sal]}
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




# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <script><div id="accordion">
#        function setalert(){
#             <div class="alert alert-success" role="alert">
#                 A simple success alert with <a href="#" class="alert-link">an example link</a>. Give it a click if you like.
#             </div>;
#         setTimeout(() => {
#             setAlert(null);
#         }, 1500);
#         }
#     </div>
#     </script>
#     """,
#     height=600,
# )