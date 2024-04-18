import streamlit as st
import pandas as pd 
import numpy as np
#from Hyuntae_Roh_proj2 import test_set_1a, test_set_1b, train_set_1a, train_set_1b 
from pprint import pprint


'''
# DSCI-510 Final Project  
designed by Hyuntae Roh 
'''

# Table of Content
st.sidebar.title('Table of Content')

# 페이지 선택
page = st.sidebar.radio('Choose the page to read.',
                        ('Home', 'Data and Preprocessing', 'Analysis', 'Code'))

# 페이지 컨텐츠
if page == 'Home':
    st.title('홈 페이지')
    st.write('이곳은 홈 페이지입니다.')


elif page == 'Data and Preprocessing':
    st.title('프로필 페이지')
    st.write('이곳은 프로필 페이지입니다.')
    my_pandas = pd.DataFrame({"col_1":[1,2,3,4], "col_2":[10,20,30,40]} )

    #mapping_1 = pd.read_csv('./data/mapping_wage_level.csv') 
    st.write(my_pandas) 


elif page == 'Analysis':
    st.title('1.Table analysis and Visualization')
    st.write('이곳은 데이터 시각화 페이지입니다.')


    st.title('2.Salary Prediction Model')
    '''
    Here is the training set that machine will learn. 
    ''' 
    #train_set_df = pd.read_csv('./data/train_set_1a.csv')
    #st.dataframe(train_set_df)

    '''
    Now, let's estimate the salary of positions; these are positions you can look at job board(themuse.com).
    '''
    #test_set_df = pd.read_csv('./data/test_set_1a.csv')
    #st.dataframe(test_set_df) 


elif page == 'Code': 
    st.title('Entire code behind this app')
    with open('Hyuntae_Roh_proj2.py','r') as file: 
        my_code = file.read() 

    st.code(my_code,language='python')


