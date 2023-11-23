import streamlit as st
import time
import json
import pandas as pd
import numpy as np
from PIL import Image

# set the page configuration
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="Generate Expectations",  # String or None. Strings get appended with "• Streamlit". 
    page_icon=None,  # String, anything supported by st.image, or None.
)

image = Image.open('assets/danone_ds_logo.png')
col1, col2 = st.columns([8, 1])
with col1:
	st.title('Generate Expectations')
with col2:
	st.write('Powered by:')
	st.image(image, width=90, output_format='PNG')

@st.cache_data
def load_data(uploaded_file_original):
    try:
        data = pd.read_csv(uploaded_file_original)
    except:
        data = pd.read_excel(uploaded_file_original)
    return data

st.subheader('Upload your raw data')
uploaded_file_original = st.file_uploader("Upload your raw data", type=['csv', 'xlsx'], help='Only .csv or .xlsx file is supported.', label_visibility="collapsed")
if uploaded_file_original is not None:
    data = load_data(uploaded_file_original)

    st.subheader('List of expectations')
    st.write('The expectations you have submitted will be reflected in the table below.')
    
    # Create an empty dataframe on first page load, will skip on page reloads
    if 'input' not in st.session_state:
        input = pd.DataFrame({'Expectations':[],'Columns':[],'Values':[]})
        st.session_state.input = input

    # Show current data
    st.dataframe(st.session_state.input, hide_index=False, use_container_width=True)
    
    def delete_expectation(expectation_number):
        if expectation_number in st.session_state.input.index:
            st.session_state.input.drop(index=expectation_number, inplace=True)
            st.session_state.input.reset_index(inplace=True, drop=True)
    
    if not st.session_state.input.empty:
        expectation_number = st.number_input('Input the row number of the expectation you want to delete', value=None, min_value=0, max_value=st.session_state.input.shape[0]-1)
        if expectation_number not None:
            st.button(f'Delete Expectation No.{expectation_number}', on_click=delete_expectation(expectation_number))

    def clear_cache():
        keys = list(st.session_state.keys())
        for key in keys:
            st.session_state.pop(key)
    st.button('Clear all expectations', on_click=clear_cache)

    config = {
    "expectation_suite_name": "my_expectation_suite",
    "datasource_name": "my_datasource",
    "project_name": "data_quality_check",
    "sheet_name": "in",
    "data_connector_name": "data_connector",
    "data_asset_name": "data_quality_check_data",
    "checkpoint_name": "my_checkpoint",
    "run_name_template": "data_quality_check",
    "rules": []
    }

    df = st.session_state.input
    for index, row in df.iterrows():
        if row['Expectations'] == 'Column values must not be null':
            config['rules'].append(
                {"expectation": "expect_column_values_to_not_be_null",
                 "kwargs": {"column": row['Columns']}
                }
            )
        elif row['Expectations'] == 'Column values must be in a list':
            config['rules'].append(
                {"expectation": "expect_column_values_to_be_in_list",
                 "kwargs": {"column": row['Columns'][0],
                           "value_list": [x.strip() for x in row['Values'].split(',')]
                           }
                }
            )
        elif row['Expectations'] == 'Column values must be numeric (integer or float)':
            config['rules'].append(
                {"expectation": "test_column_values_to_be_of_type_numeric",
                 "kwargs": {"column": row['Columns']}
                }
            )
        elif row['Expectations'] == 'Column values must be null':
            config['rules'].append(
                {"expectation": "expect_column_values_to_be_null",
                 "kwargs": {"column": row['Columns']}
                }
            )
        elif row['Expectations'] == 'Column values must match a pattern in text':
            config['rules'].append(
                {"expectation": "expect_column_values_to_match_regex",
                 "kwargs": {"column": row['Columns'][0],
                           "regex": row['Values']
                           }
                }
            )
    
    json_string = json.dumps(config)

    if not df.empty:
        st.download_button(
            label="Download your json file",
            file_name="expectations.json",
            mime="application/json",
            data=json_string,
        )	
	
    # Function to append non-form inputs into dataframe
    def add_df():
        row = pd.DataFrame({'Expectations':[st.session_state.input_df_col1],
                'Columns':[st.session_state.input_df_col2],
                'Values':[st.session_state.input_df_col3]})
        st.session_state.input = pd.concat([st.session_state.input, row])
        st.session_state.input.reset_index(inplace=True, drop=True)
	
    
    st.subheader('Input and submit your expectations')
    # Inputs created outside of a form
    select_box = st.selectbox('Expectations (required)', ('Column values must not be null', 'Column values must be null', 'Column values must be in a list', 'Column values must be numeric (integer or float)', 'Column values must match a pattern in text'), key='input_df_col1')
    if select_box == 'Column values must be in a list' or select_box == 'Column values must match a pattern in text':
        column_select = st.multiselect('Columns (required)', list(data.columns), key='input_df_col2', placeholder='Select only 1 column', max_selections=1)
    else:
        column_select = st.multiselect('Columns (required)', list(data.columns), key='input_df_col2', placeholder='Select 1 or more columns')
    if select_box == 'Column values must be in a list':
        text_input = st.text_input('Values (input values should be separated by a comma)', key='input_df_col3')
        if text_input:
            st.write("You entered: ", text_input)
    elif select_box == 'Column values must match a pattern in text':
        text_input = st.text_input('Values (only a regular expression should be input)', key='input_df_col3')
        if text_input:
            st.write("You entered: ", text_input)	    
    else:
        st.number_input('Values (not required)', value=None, key='input_df_col3', disabled=True)

    if column_select:
        st.button('Submit', on_click=add_df)


