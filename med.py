import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv("insurance.csv")
md_df =pd.read_csv("medical report.csv")


st.title("Medical Insurance Cost Predictor")
st.image("https://miro.medium.com/max/700/1*WoVYSG5nZbErqzgy47tWBQ.jpeg",width = 450)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])

from sklearn.preprocessing import LabelEncoder, binarize
le = LabelEncoder()
label2 = le.fit_transform(md_df['blood sugar'])
md_df.drop("blood sugar",axis=1,inplace=True)
md_df["blood sugar"]=label2


if nav == "Home":
    st.header("Know Our Dataset")
    
    datset = st.selectbox("Select the Dataset",["Medical-Insurance","Medical-Report"])

    if datset == "Medical-Insurance":
        if st.checkbox("Show Dataset"):
            st.dataframe(data)
        graph = st.selectbox("Select Type",["Numeric","Graphic"])
        value = st.selectbox("Select Column",["Age","Gender","BMI","Children","Medical","Region","Charges"])
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
            elif value == "Medical":
                e = data['medical'].value_counts()
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
                ax.hist(arr,bins=[10,20,30,40,50,60,70],rwidth=0.8)
                plt.title('Age Distribution')
                plt.xlabel('Age')
                plt.ylabel('Density')
                st.pyplot(fig)        
            elif value == "Gender":
                # Gender column
                plt.figure(figsize=(6,6))
                arr = data['gender']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',color='green',align="mid",rwidth=0.8)
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
                ax.hist(arr, bins=20,color='grey',rwidth=0.8)
                plt.title('BMI Distribution')
                plt.xlabel('BMI')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Children":
                # Childern column
                plt.figure(figsize=(6,6))
                arr = data['children']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='sturges',color='orange',rwidth=0.8)
                plt.title('Children Distribution')
                plt.xlabel('Children')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Medical":
                # Medical-Report column
                plt.figure(figsize=(6,6))
                arr = data['medical']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',color='purple',rwidth=0.8)
                plt.title('Medical Distribution')
                plt.xlabel('Medical')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Region":
                # Region distribution
                plt.figure(figsize=(6,6))
                arr = data['region']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',rwidth=0.8)
                plt.title('Region Distribution')
                plt.xlabel('Region')
                plt.ylabel('Density')
                st.pyplot(fig) 
            else:
                # Distribution of charge value
                plt.figure(figsize=(6,6))
                arr = data['charges']
                fig, ax = plt.subplots()
                ax.hist(arr,bins=[2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500,50000],rwidth=0.8)
                plt.title('Price Distribution')
                plt.xlabel('Charges')
                plt.ylabel('Density')
                st.pyplot(fig)


    if datset == "Medical-Report":
        if st.checkbox("Show Dataset"):
            st.dataframe(md_df)
        graph = st.selectbox("Select Type",["Numeric","Graphic"])
        value = st.selectbox("Select Column",["Age","Gender","Number of Surgeries","Blood Pressure","Colestrol","Blood Sugar","ECG","Heart Rate","Bone Fracture","Target"])
        if graph == "Numeric":
            if value == "Age":
                h = md_df['age'].value_counts()
                st.table(h)
            elif value == "Gender":
                i = md_df['gender'].value_counts()
                st.table(i)
            elif value == "Number of Surgeries":
                j = md_df['no. of surgeries'].value_counts()
                st.table(j)
            elif value == "Blood Pressure":
                k = md_df['blood pressure'].value_counts()
                st.table(k)
            elif value == "Colestrol":
                l = md_df['cholestrol'].value_counts()
                st.table(l)
            elif value == "Blood Sugar":
                m = md_df['blood sugar'].value_counts()
                st.table(m)
            elif value == "ECG":
                n = md_df['ecg'].value_counts()
                st.table(n)
            elif value == "Heart Rate":
                o = md_df['max. heart rate'].value_counts()
                st.table(o)
            elif value == "Bone Fracture":
                p = md_df['bone fracture'].value_counts()
                st.table(p)
            else:
                q = md_df['Target'].value_counts()
                st.table(q)
        if graph == "Graphic":
            if value == "Age":
                # Distribution of age value
                plt.figure(figsize=(6,6))
                arr = md_df['age']
                fig, ax = plt.subplots()
                ax.hist(arr, bins="fd",rwidth=0.8)
                plt.title('Age Distribution')
                plt.xlabel('Age')
                plt.ylabel('Density')
                st.pyplot(fig)        
            elif value == "Gender":
                # Gender column
                plt.figure(figsize=(6,6))
                arr = md_df['gender']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',color='green',rwidth=0.8)
                plt.title('Gender Distribution')
                plt.xlabel('Gender')
                plt.ylabel('Density')
                st.pyplot(fig) 
                
            elif value =="Number of Surgeries":
                # surgeries distribution
                # Normal BMI range --> 18.5 to 24.9
                plt.figure(figsize=(6,6))
                arr = md_df['no. of surgeries']
                fig, ax = plt.subplots()
                ax.hist(arr, bins="auto",color='grey',rwidth=1.2,align="mid")
                plt.title('Surgeries Distribution')
                plt.xlabel('Surgeries')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Blood Pressure":
                # Blood Pressure column
                plt.figure(figsize=(6,6))
                arr = md_df['blood pressure']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='sturges',color='orange',rwidth=0.8)
                plt.title('Blood Pressure Distribution')
                plt.xlabel('Blood Pressure')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Cholestrol":
                # Colestrol column
                plt.figure(figsize=(6,6))
                arr = md_df['cholestrol']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',color='purple',rwidth=0.8)
                plt.title('Cholestrol Distribution')
                plt.xlabel('Cholestrol')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Blood Sugar":
                # Blood Sugar distribution
                plt.figure(figsize=(6,6))
                arr = md_df['blood sugar']
                fig, ax = plt.subplots()
                ax.hist(arr, range=(0,1),rwidth=1.2)
                plt.title('Blood Sugar Distribution')
                plt.xlabel('Blood Sugar')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "ECG":
                # ECG distribution
                plt.figure(figsize=(6,6))
                arr = md_df['ecg']
                fig, ax = plt.subplots()
                ax.hist(arr,range=(0,1),rwidth=1.2)
                plt.title('ECG Distribution')
                plt.xlabel('ECG')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Heart Rate":
                # Heart Rate distribution
                plt.figure(figsize=(6,6))
                arr = md_df['max. heart rate']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',rwidth=0.8)
                plt.title('Heart Rate Distribution')
                plt.xlabel('Heart Rate')
                plt.ylabel('Density')
                st.pyplot(fig) 
            elif value == "Bone Fracture":
                # Region distribution
                plt.figure(figsize=(6,6))
                arr = md_df['bone fracture']
                fig, ax = plt.subplots()
                ax.hist(arr, bins='scott',rwidth=0.8)
                plt.title('Bone Fracture Distribution')
                plt.xlabel('Bone Fracture')
                plt.ylabel('Density')
                st.pyplot(fig) 
            else:
                # Distribution of charge value
                plt.figure(figsize=(6,6))
                arr = md_df['Target']
                fig, ax = plt.subplots()
                ax.hist(arr,range=(0,1),rwidth=1.2)
                plt.title('Target Distribution')
                plt.xlabel('Target')
                plt.ylabel('Density')
                st.pyplot(fig)

data['gender']=data['gender'].map({'female':0,'male':1})
data['medical']=data['medical'].map({'yes':1,'no':0})
data['region']=data['region'].map({'southeast':2,'southwest':1,'northeast':4,'northwest':3})



from sklearn.preprocessing import LabelEncoder
# Creating a instance of label Encoder.
le = LabelEncoder()
label1 = le.fit_transform(md_df['gender'])
md_df.drop("gender",axis=1,inplace=True)
md_df["gender"]=label1
label2 = le.fit_transform(md_df['blood sugar'])
md_df.drop("blood sugar",axis=1,inplace=True)
md_df["blood sugar"]=label2

label3 = le.fit_transform(md_df['bone fracture'])
md_df.drop("bone fracture",axis=1,inplace=True)
md_df["bone fracture"]=label3

md_df['bone fracture'] = md_df['bone fracture'].astype(pd.Int64Dtype()) 

X1 = md_df.drop(['Target'],axis=1)
y1 = md_df['Target']



X= data.drop(['charges'],axis=1)
Y=data['charges']

if nav == "Prediction":

    st.header("Know Medical Insurance Cost")
    age = st.number_input("Enter your Age",1.0,100.0,step=1.0)
    gen = st.radio("Select your Gender",["male","female"])
    bmi = st.number_input("Enter your BMI",0.0,50.0)
    child = st.number_input("Enter number of Children",0.0,4.0)
    reg = st.radio("Select your Region",["northwest","northeast","southeast","southwest"]) 
    

    st.subheader("Enter your Medical-details")
    sur = st.number_input("Number of Surgeries",0.0,10.0,step=1.0)
    bp = st.number_input("Blood Pressure",90.0,200.0,step=0.5)
    col = st.number_input("Colestrol",100.0,350.0,step=1.0)
    bs = st.radio("Blood Sugar",["no","yes"])
    ecg = st.radio("ECG",["no","yes"])
    mhr = st.number_input("Max. Heart Rate",30.0,200.0,step=1.0)
    bf = st.radio("Bone Fracture",["no","yes"])
    
    if gen == 'male':
        ling = 1
    else:
        ling = 0

    

    if reg == 'southeast':
        area = 2
    elif reg == 'southwest':
        area = 1
    elif reg == 'northeast':
        area = 4
    else:
        area = 3

    if bs == 'yes':
        blsr = 1
    else:
        blsr = 0
    if ecg == 'yes':
        eecg = 1
    else:
        eecg = 0
    if bf == 'yes':
        bofr = 1
    else:
        bofr = 0
    
    data1 = (age,ling,sur,bp,col,blsr,eecg,mhr,bofr)

    from sklearn.linear_model import LogisticRegression
    LR =LogisticRegression(C=0.01,solver='liblinear').fit(X1,y1)
    #changing input_data into numpy array
    input_data_as_numpyArray1 = np.asarray(data1)
    # reshape the array
    input_data_reshaped1 = input_data_as_numpyArray1.reshape(1,-1)
    med = LR.predict(input_data_reshaped1)
    

    if med == 1:
        medi = 1
    else:
        medi = 0
    
    
    data = {'age':age,'gender':ling,'bmi':bmi,'children':child,'medical':medi,'region':area}
    df = pd.DataFrame(data,index=[0])
    gr = GradientBoostingRegressor()
    gr.fit(X,Y)
    import joblib
    joblib.dump(gr,'model_joblib_gr')
    model = joblib.load('model_joblib_gr')
    prediction = model.predict(df)


    if st.button("Predict"):
        st.success(f"Predicted Medical Insurance Cost is : {prediction[0]} ")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    dataset = st.selectbox("Select the dataset",["Medical-Report","Medical-Insurance"])
    if dataset == "Medical-Report":
        with st.form(key="my_form" , clear_on_submit=True):
            age = st.number_input("Enter your Age",1.0,100.0,step=1.0)
            gen = st.radio("Select your Gender",["male","female"])
            sur = st.number_input("Number of Surgeries",0.0,10.0,step=1.0)
            bp = st.number_input("Blood Pressure",50.0,200.0,step=0.5)
            col = st.number_input("Colestrol",100.0,200.0,step=1.0)
            bs = st.radio("Blood Sugar",["no","yes"])
            ecg = st.radio("ECG",["no","yes"])
            mhr = st.number_input("Max. Heart Rate",100.0,200.0,step=1.0)
            bf = st.radio("Bone Fracture",["no","yes"])
            sal = st.radio("Healthy",["yes","no"])
            if sal == "yes":
                tar = 1
            else:
                tar = 0

            sub = st.form_submit_button("Submit")

            if sub:
                prg = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    prg.progress(i+1)
                to_add = {"age":[age],"gender":[gen],"no. of surgeries":[sur],"blood pressure":[bp],"cholestrol":[col],"blood sugar":[bs],"ecg":[ecg],"max. heart rate":[mhr],"bone fracture":[bf],"Target":[tar]}
                to_add = pd.DataFrame(to_add)
                to_add.to_csv("medical report.csv",mode='a',header = False,index= False)
                st.success("Submitted")
    
    if dataset == "Medical-Insurance":
    
        with st.form(key="my_form" , clear_on_submit=True):
            age = st.number_input("Enter your Age",0.0,100.0,step=1.0)
            gen = st.radio("Select your Gender",["male","female"])
            bmi = st.number_input("Enter your BMI",0.0,50.0,step=0.25)
            child = st.number_input("Enter number of Children",0,5,step= 1)
            med = st.radio("Are you a Medically-Fit?",["yes","no"])
            reg = st.radio("Select your Region",["northwest","northeast","southeast","southwest"])
            sal = st.number_input("Enter Insurance Cost",0.0,1000000.0,step = 1000.0)
            sub = st.form_submit_button("Submit")

            if sub:
                prg = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    prg.progress(i+1)
                to_add = {"age":[age],"gender":[gen],"bmi":[bmi],"children":[child],"medical":[med],"region":[reg],"charges":[sal]}
                to_add = pd.DataFrame(to_add)
                to_add.to_csv("insurance.csv",mode='a',header = False,index= False)
                st.success("Submitted")
            
           




