import streamlit as st
import time
import json
import pandas as pd
import numpy as np

st.set_page_config(page_title="Create Expectations")

@st.cache_data
def load_data(uploaded_file_original):
	try:
		data = pd.read_csv(uploaded_file_original)
	except:
		data = pd.read_excel(uploaded_file_original)
	return data

st.markdown("# Create Expectations")

uploaded_file_original = st.file_uploader("Upload your raw data", type=['csv', 'xlsx'], help='Only .csv or .xlsx file is supported.')
if uploaded_file_original is not None:
        data = load_data(uploaded_file_original)
        st.write('# Solution using input widgets')
        # a selection for the user to specify the number of rows
        num_rows = st.slider('Number of expectations', min_value=1, max_value=10)
        # columns to lay out the inputs
        grid = st.columns(3)
        # Function to create a row of widgets (with row number input to assure unique keys)
        def add_row(row):
            with grid[0]:
                st.selectbox('Expectations', ('Column values must not be null', 'Column values must be in a list', 'Column values must be of a certain type'))
            with grid[1]:
                st.multiselect('Columns', list(data.columns))
            with grid[2]:
                st.number_input('col3', step=1, key=f'input_col3{row}')
            #with grid[3]:
            #    st.number_input('col4', step=1, key=f'input_col4{row}',
            #                    value = st.session_state[f'input_col2{row}'] \
            #                        -st.session_state[f'input_col3{row}'],
            #                    disabled=True)
    
        # Loop to create rows of input widgets
        for r in range(num_rows):
            add_row(r)
