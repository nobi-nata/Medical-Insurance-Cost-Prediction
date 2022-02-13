from time import time
import streamlit as st
import pandas as pd
import numpy as np


a = [1,2,3,4,5,6,7,8]
n = np.array(a)
nd = n.reshape((2,4))
d = {
    "name":"harsh",
    "age":21,
    "city":"belgaum",
}

data = pd.read_csv('insurance.csv')


st.dataframe(data)
# st.table(nd)

st.json(d)

st.write(a)

# @st.cache
# def ret_time():
#     time.sleep(4)
#     return time.time()
    
# if st.checkbox("1"):
#     st.write(ret_time())

    
# if st.checkbox("2"):
#     st.write(ret_time())

