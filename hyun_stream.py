pip freeze > requirements.txt
# Preset ---------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd 
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn import linear_model
#from Hyuntae_Roh_proj2 import test_set_1a, test_set_1b, train_set_1a, train_set_1b 
import time 

# Read CSV
train_set_1b = pd.read_csv('./data/train_set_1b.csv') 
train_set_1a = pd.read_csv('./data/train_set_1a.csv') 
test_set_1b = pd.read_csv('./data/test_set_1b.csv')
test_set_1a = pd.read_csv('./data/test_set_1a.csv') 


# Table of Content
st.sidebar.title('Table of Content')

# 페이지 선택
page = st.sidebar.radio('Choose the page to read.',
                        ('Home', 'Data and Preprocessing', 'Analysis', 'Code'))

# 페이지 컨텐츠
if page == 'Home':
    '''
    # HOME: DSCI-510 Final Project  
    designed by Hyuntae Roh 
    '''

    st.title('Collect your raw data')
    '''
    To save your precious time, this application is using pre-runned outcome of raw data collecting process.\n
    If you want to run the collecting process again, please click this button to collect raw data again\n
    '''
    button_status = False
    if st.button('Re-collect the raw data',help='It will take 10-15 minutes') == True :
        button_status = True
            # 스피너 표시
        with st.spinner('Running the raw data collecting process...'):
            # 파일 import하는 함수 호출
            import Hyuntae_Roh_proj2
            time.sleep(2)  # 예시로 2초 대기

        # 스피너 종료
        st.success('Thank you for waiting. Process Completed!')
    if button_status == True: 
        st.button('Re-collect the raw data',help='It will take 10-15 minutes',disabled=False) 

    st.write('The process may take 10-20 mintues.') 

elif page == 'Data and Preprocessing':
    ## Step1----------------------------------------
    st.markdown('# Step1: Data collection and Pre-processing :ear_of_rice:')
    st.write('Here is a brief summary of collected data') 
    data_summary = {"Name": ['Data_1', 'Data_2', 'Data_3', 'Data_4'], 
                    "Content": ['Job posts','Cost of Living','H-1b LCA','Company Info'],
                    "Collecting Method": ['API','Web-Scrap','Download','API'],
                    "Link": ['https://www.themuse.com/api/public/jobs','https://www.numbeo.com/cost-of-living/',
                             'https://www.dol.gov/agencies/eta/foreign-labor/performance','https://www.themuse.com/api/public/companies']
                    }
    data_sum = pd.DataFrame(data_summary) 
    st.write(data_sum) 

    st.write('For example, Data_1 is collected and pre-processed by the following codes')
    with open('./Hyuntae_Roh_proj2.py','r') as file: 
        lines = file.readlines()[9:165] 
    data1_code = ''.join(lines)

    with st.expander('Data_1 Code'): 
        st.code(data1_code) 

    ## Step2 --------------------------------------
    st.markdown('---') 
    st.markdown('# Step2: Generate Tables :rice:')  
    '''
    Based on the collected raw data, here I made some tables for data modeling, extra preprocessing, and analysis.

    MAPPING_TABLE: 
    All mapping tables are required to maintain data integrity of TEST_SET_TABLE
    1) mapping_company_names (COMPANY: DATA3 & DATA4) 
    - (DATA3)EMPLOYER NAME  ---key \n
    - (DATA4)COMPANY NAME  \n
    - (DATA4)COMPANY ID \n
    \n
    2) mapping_wage_level (Level: DATA1 & DATA3) 
    - (DATA3)PW_WAGE_LEVEL ---key \n
    - (DATA1)level  \n
    '''
    
    with st.expander('Mapping Tables'):
        st.write('Mapping Table of wage level.')    
        mapping_1 = pd.read_csv('./data/mapping_wage_level.csv').iloc[:,1:] 
        st.write(mapping_1) 

        st.write('Mapping Table of company names.')
        mapping_2 = pd.read_csv('./data/mapping_company_names.csv').iloc[:,1:] 
        st.write(mapping_2) 

    '''
    TRAIN_SET_TABLE (For Training data for Salary Estimate Model.) \n
    Data3 Inner join Data2 on= [YEAR, LOCATION]
    - ENGINEERING_TITLE ; 1 if Engineering or Engineer, 0 else. 
    - YEAR ; Positive Integer variable 
    - Cost of Living ; Continuous variable that implicitly include WORKSITE information(NY=100)   
    - LOCATION ; Binary variables(One-hot-encoding for SF or LA ; Base = New York)
    - PW_WAGE_LEVEL ; Positive Integer variable(Scale 1 to 4. Higer value means higher prevailing wage level) 
    - ADJUSTED_P_WAGE ; Response(dependent) feature(Adjusted by Cost-of-living index)

    Applied samely to TEST_SET_TABLE. 
    '''

    with st.expander('Tables for (linear/non-linear) machine learning'):
        st.write('Table of training set') 
        train = pd.read_csv('./data/train_set_1a.csv').iloc[:,1:]
        st.write(train)

        st.write('Table of test set') 
        test = pd.read_csv('./data/test_set_1a.csv').iloc[:,1:]
        st.write(test)   



elif page == 'Analysis':
    st.markdown('## Questions for analysis')
    '''
        The main goal of this project is to get a nice picture for international students to have better understanding in data-related job market.
    I anticipate my data and analysis could answer the following questions and directions: \n
     '''
    with st.expander('DISCLAIMER', expanded=True):
        st.markdown('1. Raw data were collected only for three cities: New York, Los Angeles, San Francisco') 
        st.markdown('2. Time window of training data is 2019 ~ 2023. Test data is 2024.') 
        st.markdown('3. The size of company was randomly assigned when the company name has no match with DATA_4(company_info)')
    # Part 1====================================================
    st.markdown('## Part 1. Exploring training data(H-1b LCA)') 
    ## A1_1. 
    st.markdown('(Q1) Which company recorded the highest base salary for data-specific roles such as data analyst, data engineer or data scientist? Now, group the table by company name and calculate the number of LCA and the average base salary. Which company is a top sponsor of H1-B in terms of the number of LCA? How about in terms of average salary? ')
    year = st.multiselect('You can multi-select YEAR',set(train_set_1b['YEAR'].values),set(train_set_1b['YEAR'].values)) 
    df_A1_1 = train_set_1b[ train_set_1b['YEAR'].isin(year) ].iloc[:,1:]
    df_A1_1 = df_A1_1.sort_values(by=['ADJUSTED_P_WAGE'], ascending=False).reset_index(drop=True) 
    st.write(df_A1_1)

    st.write("Let's see the ranking of the number of H-1b sponse and average wage. To sort by the column name, click the column name")
    h1b_count = df_A1_1.groupby('EMPLOYER_NAME').agg({'JOB_TITLE':'count','ADJUSTED_P_WAGE':'mean'}).reset_index().sort_values('JOB_TITLE', ascending=False) 
    st.write(h1b_count) 

    ## A1_2.
    st.markdown("(Q2) Now, view the data in location-wise. Apply location filter as ‘Los Angeles’ and report the number of certified LCA, sponsors and their average salary. Then change the location filter to New York, San Fransisco, etc. and compare the result. Which city seems to have an wider opportunity for H1-B? Any significant wage difference can you tell?") 
    city = st.multiselect('City filter',set(train_set_1b['WORKSITE_CITY'].values),'LOS ANGELES')
    df_A1_2 = train_set_1b[ train_set_1b['WORKSITE_CITY'].isin(city) ].iloc[:,1:]
    h1b_count_2 = df_A1_2.groupby(['YEAR']).agg({'JOB_TITLE':'count','ADJUSTED_P_WAGE':'mean'}).reset_index()
    h1b_count_2 = h1b_count_2.round(1).sort_values('JOB_TITLE',ascending=False) 
    st.write(h1b_count_2) 


    #Part 2=====================================================  
    st.markdown('## Part 2. Job posts and wage prediction')
    ## A2_1
    st.markdown('(Q1) One of the most curious information that job seekers want to find from job posts is the salary figure, indeed. Although the employers are reluctant to clarify the amount of the salary would be, you can try prediction using linear regression, for instance')
    '''
    → Now, let's estimate the salary of positions; these are positions you can look at job board(themuse.com).
    Here, "ADJUSTED_P_WAGE", which states the prevailing wage adjusted by cost-of-living index, is our target of prediction.
    Let's take a look of the result.
    '''
    with st.expander('Table of Training Data'):
        train = pd.read_csv('./data/train_set_1a.csv').iloc[:,1:]
        st.write(f"Shape of data : {train.shape}")
        st.write(train)     
    

    # Split train/test data 
    X_train = train_set_1a.iloc[:,:-1]
    Y_train = train_set_1a.iloc[:,-1]
    X_test = test_set_1a.iloc[:,:-1]
    Y_test = test_set_1a.iloc[:,-1]

    # Model Training  
    model= linear_model.LinearRegression() 
    model.fit(X_train, Y_train) 

    # Prediction 
    y_pred = model.predict(X_test)   

    # Update to the table of test data 
    test_set_1a['ADJUSTED_P_WAGE'] = np.round(y_pred,1) 
    test_set_1b['ADJUSTED_P_WAGE'] = np.round(y_pred,1)  

    st.write('Prediction Result(LinearRegression)')
    st.write(test_set_1b.iloc[:,1:] ) 


    ## A2_2 
    st.markdown('(Q2) In terms of job location, can you describe the regional distribution of data-related positions?') 
    ### Table (df_A2_1)
    df_A2_1 = test_set_1b.iloc[:,1:]
    job_count = df_A2_1.groupby('WORKSITE_CITY').agg({'JOB_TITLE':'count','ADJUSTED_P_WAGE':'mean','PW_WAGE_LEVEL':'mean'}).reset_index().round(2) 
    coords = pd.DataFrame({"City":['LOS ANGELES', 'SAN FRANCISCO', 'NEW YORK'], 
                           "latitude":[34.0549,37.7749, 40.7128],
                           "longitude":[-118.2426, -122.4194, -74.0060] 
                           })
    job_count = pd.merge(job_count,coords,how='left',left_on='WORKSITE_CITY',right_on='City')
    st.write(job_count.drop('City',axis=1))

    ### Display on map 
    
    fig = px.scatter_mapbox(job_count, lat="latitude", lon="longitude", hover_name="WORKSITE_CITY",
                            size="JOB_TITLE", color="ADJUSTED_P_WAGE",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            size_max=40, zoom=2)
    
    fig.update_layout(mapbox_style="open-street-map") # You can change map style
    st.write(fig)

    ### Draw Stacked boxplot  
    st.write('The segment table below tells the distribution of wage_level and engineer title by each city.')
    df_A2_2 = df_A2_1[['WORKSITE_CITY','PW_WAGE_LEVEL','ENGINEER_TITLE']]
    city2 = st.select_slider('Choose the city to view',set(df_A2_2['WORKSITE_CITY'].values))
    df_A2_2 = pd.DataFrame(df_A2_2[ df_A2_2['WORKSITE_CITY'] == city2 ].value_counts().reset_index()) 
    #st.write(df_A2_2)  

    color_mapping = {0: 'blue', 1: 'orange'} #--- Engineer title 에 따른 색상 매핑

    # 데이터프레임을 E 값에 따라 그룹화하여 각각의 그룹에 대한 Count 합계 계산해야한다.

    fig = go.Figure() #--- Plotly의 Bar 객체 생성
    for e_value, color in color_mapping.items(): #--- 각각의 그룹에 대한 stacked bar 생성
        grouped_data = df_A2_2[df_A2_2['ENGINEER_TITLE'] == e_value]
        fig.add_trace(go.Bar(x=grouped_data['PW_WAGE_LEVEL'], y=grouped_data['count'], name=f'ENGINNER={e_value}',
                            marker_color=color))

    # 그래프 레이아웃 설정
    fig.update_layout(barmode='stack', xaxis_title='PW_WAGE_LEVEL', yaxis_title='Count',
                    title='Stacked Bar Chart with Color') 
    
    st.plotly_chart(fig) 

        
    ## A2_3
    st.markdown('(Q3) If you can get old records from the API, for example job posting published in 2020, can you describe change in regional distribution from every March from 2020 to 2024?')
    st.markdown(' → Unfortunately, the API provider seems to remove old job posts, so I should leave this question for future analysis')  





elif page == 'Code': 
    st.title('Entire code behind this app')
    with open('Hyuntae_Roh_proj2.py','r') as file: 
        my_code = file.read() 

    st.code(my_code,language='python')


