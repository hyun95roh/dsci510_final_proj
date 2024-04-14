import streamlit as st
import pandas as pd 
import numpy as np
import requests
import re 
import time 
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from pprint import pprint


'''
Table of Content 
1. Visualization
'''

'''
#1. Visualization
## 1.1 Code itself
'''
st.write("Example - st.code():")
my_code = "pd.DataFrame( {'first column': [1,2,3,4], 'second column':[10,20,30,40]})"
st.code(my_code, language='python')



