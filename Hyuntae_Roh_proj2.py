import requests
import pandas as pd  
import re 
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup

# ==============================================================================================================
# Data 1(API data) collect =====================================================================================
# 1.Loading API data
# Data 1(API data) ----- custom functions
def jobposts_collector(url:str,category:list, location:list):
    all_page_data = {} #--- Store job posts of all pages
    all_page_data_df = pd.DataFrame()
    current_page = 1   #--- Preset 
    print("Bringing job posts from API...")
    while 1 :    
        get_params = {'page':current_page,
                'category':category,
                'location':location
                
                }
        temp = requests.get(url, params= get_params)
        max_page = temp.json()['page_count'] 

        if temp.status_code == 200: 
            #parsing JSON data 
            page_data, page_data_df = jobposts_per_page(temp)

            print(f"Page {current_page} out of {max_page} complete.")
            if max_page >1 :
                current_page += 1


            all_page_data[current_page] = page_data 
            all_page_data_df = pd.concat([all_page_data_df,page_data_df], axis=0) 

            if current_page > max_page:
                break 

        else:
            print("Failed to retrieve data. Status code:", temp.status_code) 
        #result_ = temp.json()

    jobpost_columns = ['company_id','company_name','posted_date',
                    'remoteflex','locations',
                    'category','position','level',
                    'content_salary','content_year',
                    'content_prog_lang','content_line']
    all_page_data_df.columns = jobpost_columns

    return all_page_data, all_page_data_df


def jobposts_per_page(temp:requests.Response):
    a_page_results_ = temp.json()['results']
    job_posts = []
    job_posts_df = pd.DataFrame()
    
    for i in a_page_results_: 
        content = content_parser(i['contents'] ) #--- custom function
        category = i['categories'][0]['name']  
        position = i['name']
        level = i['levels'][0]['short_name']
        pub_date = i['publication_date']  
        location = [] 
        for k in i['locations']: 
            location.append( k['name'].upper() )
        company_name = i['company']['name']
        company_id = i['company']['id']
        remoteflex = 1 if 'FLEXIBLE / REMOTE' in location else 0 

        a_post = {'company_id':company_id,
                'company_name':company_name,
                'Posted date':pub_date,
                'remoteflex':remoteflex,
                'locations':",".join(location),
                'category':category,
                'position':position,
                'level':level,
                'content':content
                }
        
        a_post_df = pd.DataFrame([[company_id, company_name, pub_date[:10],
                                  remoteflex, ",".join(location), 
                                  category, position, level,
                                    content['salary'],content['years'], 
                                    content['prog_lang'], content['line']
                                    ]])

        job_posts.append(a_post)
        job_posts_df = pd.concat([job_posts_df,a_post_df],axis=0)
    #job_posts_df.columns = jobpost_columns
    return job_posts, job_posts_df 

def content_parser(content):
    #content_in_a_line ="".join(content) #--when content is list type.
    content_in_a_line = content 
    soup_ = BeautifulSoup(content_in_a_line, 'html.parser')
    parsed = soup_.find_all('p') 
    
    exp_ = {'line':"", 'years':"", 'prog_lang':"", 'salary':""}
    list_prog_lang = ['SQL','Python','Spark','Azure','Java','Scala',
                      'NoSQL','MySQL','AWS', 'Google Cloud',
                      'PowerBI','Tableau','Snowflake','BigQuery','Databricks']
    
    for i in parsed :
        #print(i)
        temp = str(i)   
        temp = tag_remover(temp) #--- custom function
        splited = temp.split() 
        if temp.find("experience")>-1 or temp.find("proficient")>-1:
            indicies = rough_idx_finder("experience", splited)  #--- custom function
            exp_['line'] = [ " ".join(splited[idx-5:idx+6]) for idx in indicies] #-- Bring left&right 5 word blocks from center(idx)    
            if temp.find("year")>-1:
                #print(temp)
                #print(splited)
                indicies = rough_idx_finder("year",splited) 
                #print(indicies)
                idx = indicies[0]
                exp_['years'] = splited[idx-1] 
            
            intersection = set(splited) & set(list_prog_lang)
            flag = list(intersection)           
            if len(flag) >0 : 
                exp_['prog_lang'] = list(intersection)

        if temp.find("salary")>-1 and temp.find("$")>-1:
            indicies = rough_idx_finder("$",splited)
            if len(indicies)>1: 
                exp_['salary'] = splited[indicies[0]:indicies[-1]+1]
            else:
                exp_['salary'] = splited[indicies[0]]  


    return exp_

def rough_idx_finder(seek:str,data:list):
    indices = []
    for idx, ele in enumerate(data): 
        if seek in ele :
            indices.append(idx) 
    return indices 

def tag_remover(temp:str): 
    tag_lists = ['<p>','</p>','<li>','</li>','<br>','<br/>','</br>','<b>','</b>','<ul>','</ul>','<span>','\\n',]
    removed = temp
    for i in tag_lists: 
        removed = removed.replace(i,' ') 
    return removed


## Presets 
url_base= 'https://www.themuse.com/api/public/'  #--- API docs: https://www.themuse.com/developers/api/v2 
jobs = 'jobs' #for Data_1
url = url_base + jobs
companies = 'companies' #for Data_4

## Filters 
category = ['Data Science','Data and Analytics']
locations = ['Los Angeles, CA','San Francisco, CA', 'New York, NY']

## Outcome 
all_page, all_page_df = jobposts_collector(url, category, locations)



# Data_2(Scrapable data) collect ===============================================================================================
# Data 2  ---custom function 
def col_scrapper(year_filter, soup_cost):
    #pprint(soup_cost.tbody.find_all('td'))
    html_list = soup_cost.tbody.find_all('td')
    cost_of_living = pd.DataFrame()
    row_idx = 0 
    city_name = ''
    state_name = ''
    location = ''
    empty_row_checker = 0 
    
    for i in html_list :
        #print(i.attrs.values())

        if str(i.attrs.values()).find('city') > -1 :
            #print(i.string)
            city_name = i.string 

            if (city_name).count(',')>1 and (city_name).count('United')>0:
                state_name = city_name.split(',')[1].strip().upper()
                city_name = city_name.split(',')[0].upper()
                location = city_name+", "+state_name
                empty_row_checker = 0.5
            else:
                empty_row_checker = 0
        
        if empty_row_checker==0.5 and "text-align: right" in i.attrs.values() : 
            #print(i.string)
            col_index = i.string 
            empty_row_checker = 1


        if empty_row_checker == 1 :
            new_row = pd.DataFrame([[year_filter,location,city_name,state_name, col_index]])
            cost_of_living = pd.concat([cost_of_living, new_row], axis=0)
            empty_row_checker = 0 
            

    cost_of_living.columns = ['YEAR','LOCATION','City','State','Cost of Living Plus Rent Index']
    return cost_of_living

# Data 2(scrapable) collect 
cost_df = pd.DataFrame() 
print("Scraping Cost of Living Index...")
for year in range(2019, 2025): 
    year_filter = str(year) 
    cost_url = f'https://www.numbeo.com/cost-of-living/region_rankings.jsp?title={year_filter}&region=021&displayColumn=2'
    response_cost = requests.get(cost_url)
    print(response_cost.status_code) 
    soup_cost = BeautifulSoup(response_cost.content, 'html.parser') 

    scrapped_df = col_scrapper(year_filter, soup_cost) 
    cost_df = pd.concat([cost_df, scrapped_df], axis=0)


# Data_3(Downloadable data) collect  ======================================================================================
# Data3(downloadable data) --- custom function
def custom_filter(csvfile:pd.DataFrame, time_window:list, location_keyword:dict, job_keyword:str):
    csvfile= csvfile.dropna() 
    start_date = time_window[0]
    end_date = time_window[1]

    # h-1b filter
    filter_h1b = csvfile['VISA_CLASS']=='H-1B'
    
    # Time window
    filter_time = (csvfile['DECISION_DATE'] >= start_date) & (csvfile['DECISION_DATE'] <= end_date)

    # Location kewords
    filter_states = 1 if location_keyword.get('state', -1) == -1 else (csvfile['WORKSITE_STATE'].isin(location_keyword['state']))
    filter_city = 1 if location_keyword.get('city', -1) == -1 else (csvfile['WORKSITE_CITY'].isin(location_keyword['city']))

    # Job keword
    filter_job = csvfile['JOB_TITLE'].str.contains(job_keyword)

    # PW_UNIT filter 
    filter_pwu = csvfile['PW_UNIT_OF_PAY'] == 'Year'

    # Comine conditions 
    tempcsv = csvfile[filter_h1b & filter_time & filter_states & filter_city & filter_pwu & filter_job]
    # Index reset
    tempcsv.reset_index(drop=True, inplace=True)

    return tempcsv

# Data 3(downloadable data) - Collect
## Preset:
file_path = './data/' 
years = [str(i) for i in range(2019,2024)]

## read csv 
for i in years:
    globals()[f'LCA_{i}'] = pd.read_csv(file_path+"LCA_fy"+i+".csv")

## Preprocessing for LCA_2019 -- This is because the U.S.A government changed the format after 2019.      
LCA_2019.columns = list(LCA_2020.columns) 

def convert_state_name_to_acronym(state_name):
    state_mapping = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY' }

    return state_mapping.get(state_name, state_name)

LCA_2019['WORKSITE_STATE'] = LCA_2019['WORKSITE_STATE'].apply(convert_state_name_to_acronym)
LCA_2019['PW_WAGE_LEVEL'] = LCA_2019['PW_WAGE_LEVEL'].str.replace('Level ','')


## Concatenate All LCA csv files.
LCA_2019to2023 = pd.concat([LCA_2019,LCA_2020,LCA_2021,LCA_2022,LCA_2023], axis=0) 
LCA_2019to2023['LOCATION'] = LCA_2019to2023['WORKSITE_CITY']+", "+LCA_2019to2023["WORKSITE_STATE"] 
LCA_2019to2023['YEAR'] = LCA_2019to2023['DECISION_DATE'].str[:4]
LCA_2019to2023['CASE_STATUS'] = LCA_2019to2023['CASE_STATUS'].str.replace('Certified - Withdrawn','CERTIFIED-WITHDRAWN')
LCA_2019to2023['CASE_STATUS'] = LCA_2019to2023['CASE_STATUS'].str.upper() 

## Project the filtered data 
start_date = '2019-01-01'
end_date = '2023-12-31'
time_window = [start_date, end_date]

job_keyword = 'DATA'
location_keyword = {'city': ['LOS ANGELES','SAN FRANCISCO','NEW YORK']} #Please write in capital letter. State name should be inputed as betwo alhabet acronym(ex: New York -> NY )
LCA_df = custom_filter(LCA_2019to2023,time_window, location_keyword, job_keyword) 
LCA_df



# Data 4(API data) collect ==========================================================================================
## Custom functions 
def companies_per_page(json_:requests.Response):
    #pprint(temp.json())
    a_page_results_ = json_['results']
    comp_info_df = pd.DataFrame()

    for a_company in a_page_results_:
        comp_id = a_company['id']
        comp_name = a_company['name']
        comp_shortname = a_company['short_name']
        comp_industry= []
        for i in a_company['industries']: 
            comp_industry.append(i['name'] ) 
        comp_size = a_company['size']['short_name']

        new_row = pd.DataFrame([[comp_size, comp_id, comp_name, comp_shortname, ",".join(comp_industry)]])
        comp_info_df = pd.concat([comp_info_df, new_row], axis=0 ) 
    return comp_info_df 

def comp_info_collector(url, size, locations ):
    current_page = 1
    all_comp_info = pd.DataFrame()

    print("Bringing company profiles from API...")
    while 1 :
        get_params = {'page': current_page,
            'size':size,
            'location':locations
            }
        temp = requests.get(url, params = get_params )
        json_ = temp.json() 
        max_page = json_['page_count']


        if temp.status_code==200:
            a_page_result_ = companies_per_page(json_) 
            all_comp_info = pd.concat([all_comp_info, a_page_result_],axis=0)
            print(f'Page {current_page} out of {max_page} completed')
            if max_page >1:
                current_page += 1 
        else : 
            print("Failed to retrieve data. Status code:", temp.status_code) 

        if current_page > max_page:
            break 


    all_comp_info.columns = ['company_size','company_id','company_name','short_name','industry'] 
    
    return all_comp_info

## Loading API data(company info)
## Presets 
url_base= 'https://www.themuse.com/api/public/'  #--- API docs: https://www.themuse.com/developers/api/v2 
companies = 'companies' #for Data_4
url = url_base + companies

## Filters 
inudstry = []
size = [] #-- ex) 'Lage Size', 'Small Size', 'Medium Size'
locations = [] #-- ex) 'New York, NY'

## Outcome 
comp_info_df = comp_info_collector(url, size, locations)
comp_info_df


# =============================================================================================================
# MAPPING_TABLE_COMPANY (EMPLOYER_NAME -> short_name : Data3 & Data4 )
## Custom functions 
def match_company_names(str1, str2):
    # Similarity score
    score = fuzz.ratio( str1.lower(), str2.lower())
    
    return score 

def map_companies(df_A, df_B):
    company_info = []

    for name_a in set(df_A['EMPLOYER_NAME']):
        #name_a_re = re.sub(r'INTERNATIONAL|TECHNOLOG|FINANCIAL|INC|LLC|, CORP|CORPORATION|ADVERTISING|COMMUNICATION|DEVELOPMENT|CONSULTING|SOLUTION|SERVICES|CENTER|ASSOCIATES', '', name_a)
        name_a_re = name_a.split(" ")[0] 
        name_a_re2 = None if len(name_a_re.split(" "))<2 else (name_a_re.split(" ")[0] + name_a_re.split(" ")[1]).replace(" ","")

        matched_name_a = None
        matched_name_b = None
        max_score = 0 
        for name_b in df_B['short_name']:
            name_b_re = name_b.upper()
            
            score = match_company_names(name_a_re, name_b_re) 
            score2 = 0 if name_a_re2 is None else match_company_names(name_a_re2, name_b_re)
            if max(score, score2) > max_score :
                matched_name_a = name_a
                matched_name_b = name_b
                max_score = max(score,score2)
                size_b = list(df_B[ df_B['short_name'] == name_b ]['company_size'])[0]
                
        if max_score > 90 :
            company_info.append({'h1b_name': matched_name_a, 'on_job_board': matched_name_b,"company_size":size_b})
        else:
            company_info.append({'h1b_name': name_a, 'on_job_board': None, "company_size": np.random.choice(['large','medium','small']) })


    return pd.DataFrame(company_info) 

## Generate mapping_table 
mapping_company_df = map_companies(LCA_df, comp_info_df).dropna()

#export to csv 
mapping_company_df.to_csv('./data/mapping_company_names.csv')
mapping_company_df



# MAPPING_TABLE_WAGE (Level->PW_WAGE_LEVEL: Data1 & Data3) ====================================================
level_mapping = {'internship':1,'entry':1,'mid':2,'senior':3,'management':4} 

def convert_level_to_PWWL(level_in_Data3):
    level_mapping = {'internship':1,'entry':2,'mid':3,'senior':3,'management':4} 
    return level_mapping.get(level_in_Data3,'-') 

mapping_wage_level = pd.DataFrame(level_mapping.items()) 
mapping_wage_level.columns = ['Before','After']

## Export to csv
mapping_wage_level.to_csv('./data/mapping_wage_level.csv')   


# ======================================================================================================================
# TRAIN_SET_TABLEs (For Training data for Salary Estimate Model.)
## inner join : Data_2(cost of living) and Data_3(h-1b)
train_set_1b = pd.merge(LCA_df, cost_df, on=['YEAR','LOCATION'], how='left')
train_set_1b['ADJUSTED_P_WAGE'] = train_set_1b['PREVAILING_WAGE'].astype(float)*(100/train_set_1b['Cost of Living Plus Rent Index'].astype(float))
train_set_1b['ADJUSTED_P_WAGE'] = train_set_1b['ADJUSTED_P_WAGE'].round(1)

## Preprocessing 
def convert_PWWL(pwwl):
    mapping = {'I':1,'II':2,'III':3,'IV':4}
    return mapping.get(pwwl)

### Adding new columns and Typecasting columns 
train_set_1b['PW_WAGE_LEVEL'] = train_set_1b['PW_WAGE_LEVEL'].apply(convert_PWWL)
train_set_1b['ENGINEER_TITLE'] = train_set_1b['JOB_TITLE'].str.contains('ENGIN').astype(int)
train_set_1b['FULL_TIME_POSITION'] = train_set_1b['FULL_TIME_POSITION'].replace({'Y':1,'N':0})
train_set_1b['YEAR'] = train_set_1b['YEAR'].astype(int)
comp_list = set(train_set_1b[ 'EMPLOYER_NAME' ].values)
train_set_1b['EMPLOYER_SIZE'] = pd.merge(train_set_1b, mapping_company_df, left_on=['EMPLOYER_NAME'], right_on=['h1b_name'],how='inner')['company_size']
train_set_1b['SF'] = train_set_1b['WORKSITE_CITY'].str.contains('SAN FRANCISCO').astype(int)
train_set_1b['LA'] = train_set_1b['WORKSITE_CITY'].str.contains('LOS ANGELES').astype(int)
## drop redundant columns
dropped_cols = ['CASE_STATUS','FULL_TIME_POSITION','DECISION_DATE','VISA_CLASS',
                'City','State','LOCATION','WORKSITE_STATE','PW_UNIT_OF_PAY','PREVAILING_WAGE']
train_set_1b = train_set_1b.drop(dropped_cols,axis=1)


## Drop columns -- generate INNER_JOIN_TABLE_1a 
train_set_1a = train_set_1b.drop(['JOB_TITLE','EMPLOYER_NAME','WORKSITE_POSTAL_CODE'],axis=1)
new_col_order = ['ENGINEER_TITLE','EMPLOYER_SIZE','SF','LA','PW_WAGE_LEVEL','YEAR','Cost of Living Plus Rent Index','ADJUSTED_P_WAGE']
train_set_1a = train_set_1a[ new_col_order ].dropna() 
train_set_1a['level'] = train_set_1a['level'].replace({'small':1,'medium':2,'large':3})

## Export to csv 
train_set_1a.reset_index.to_csv('./data/train_set_1a.csv')
train_set_1a



# =======================================================================================================================
# TEST_SET_TABLE(For Testing Salary Estimate Model) ------ inner join : Data_1(job board) and Data_3(h-1b)  
##  Preprocessing - part1
import numpy as np
drop_cols = ['content_salary','content_year','content_prog_lang','content_line','category'] 
job_board = all_page_df.drop(drop_cols,axis=1)
temp_df = pd.merge(job_board, comp_info_df.drop(['company_name','short_name','industry'],axis=1), on='company_id', how='left')

temp_df['posted_year'] = temp_df['posted_date'].str[:4]
temp_df['ENGINEER_TITLE'] = temp_df['position'].str.upper().str.contains('ENGINEER').astype(int)
temp_df = temp_df[ temp_df['level'].isin(['internship']) == False ] 
temp_df['ADJUSTED_P_WAGE'] = 0

temp_df['level'] = temp_df['level'].replace({'entry':1,'senior':2,'mid':3,'management':4})
# Lead, President, Director에 해당되는 행을 필터링하여 별도의 데이터프레임으로 저장
filtered_df = temp_df[(temp_df['position'].str.contains('Lead')) |
                      (temp_df['position'].str.contains('President')) |
                      (temp_df['position'].str.contains('Director'))]

# 별도로 저장된 데이터프레임의 level 값을 4로 변경
filtered_df.loc[:, 'level'] = 4

# 원본 데이터프레임에 변경된 부분을 반영
temp_df.update(filtered_df)

filter_NY = temp_df['locations'].str.contains('NEW YORK') 
filter_SF = temp_df['locations'].str.contains('SAN FRANCISCO') 
filter_LA = temp_df['locations'].str.contains('LOS ANGELES')
temp_df[ 'NY' ] = filter_NY.astype(int)
temp_df[ 'SF' ] = filter_SF.astype(int)
temp_df[ 'LA' ] = filter_LA.astype(int)

## Preprocessing -part 2
## Endow random worksite label for remoteflex==1 without city label.  
cities = []
for index, row in temp_df[['NY','LA','SF']].iterrows():
    count = row.sum()
    indicies_with_1 = row[row == 1].index
    if count == 1:
        city = row[row == 1].index[0]
    elif count >1:
        city = np.random.choice(indicies_with_1)
    else: 
        city = np.random.choice(['NY','SF','LA'])
    cities.append(city)

temp_df['WORKSITE_CITY'] = cities
city_map = {'NY':'NEW YORK','LA':'LOS ANGELES','SF':'SAN FRANCISCO'} 
temp_df['WORKSITE_CITY'] = temp_df['WORKSITE_CITY'].map(city_map)

## Merge with Cost_of_living data 
temp_df = pd.merge(temp_df,cost_df, how='left',left_on=['WORKSITE_CITY','posted_year'],right_on=['City','YEAR'])

## Reorder Columns
new_col_order = ['position','ENGINEER_TITLE','company_name','company_size','WORKSITE_CITY','SF','LA','level','YEAR','Cost of Living Plus Rent Index','ADJUSTED_P_WAGE']
test_set_1b = temp_df[ new_col_order ]
new_col_name = ['JOB_TITLE','ENGINEER_TITLE','EMPLOYER_NAME','EMPLOYER_SIZE','WORKSITE_CITY','SF','LA','PW_WAGE_LEVEL','YEAR','Cost of Living Plus Rent Index','ADJUSTED_P_WAGE']
test_set_1b.columns = new_col_name

## Final output 
test_set_1b = test_set_1b.dropna() 
test_set_1a = test_set_1b.drop(['JOB_TITLE','EMPLOYER_NAME','WORKSITE_CITY'],axis=1)
test_set_1a['level'] = test_set_1a['level'].replace({'small':1,'medium':2,'large':3})
test_set_1a

## Export to csv 
test_set_1a.reset_index(drop=True).to_csv('./data/test_set_1a.csv')
test_set_1a

