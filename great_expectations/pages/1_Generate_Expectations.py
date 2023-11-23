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
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write('The expectations you have submitted will be reflected in the table below.')
    with col2:
        def clear_cache():
            keys = list(st.session_state.keys())
            for key in keys:
                st.session_state.pop(key)
        st.button('Delete all expectations', on_click=clear_cache)
        
    # Create an empty dataframe on first page load, will skip on page reloads
    if 'input' not in st.session_state:
        input = pd.DataFrame({'Expectations':[],'Columns':[],'Values':[]})
        st.session_state.input = input

    # Show current dataframe
    st.dataframe(st.session_state.input, hide_index=False, use_container_width=True)

    # Download current dataframe as a json file
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

    df = st.session_state.input.loc[st.session_state.input.astype(str).drop_duplicates().index]
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
        elif row['Expectations'] == 'Column values must be between 2 numbers':
            config['rules'].append(
                {"expectation": "expect_column_values_to_be_between",
                 "kwargs": {"column": row['Columns'][0],
                            "min_value": row['Values'].split(' - ')[0],
			    "max_value": row['Values'].split(' - ')[1]
                           }
                }
            )

    json_string = json.dumps(config)

    if not df.empty:
        st.download_button(
            label="Download your list as a json file",
            file_name="expectations.json",
            mime="application/json",
            data=json_string,
        )

    # Add options to delete expectations
    def delete_expectation(expectation_number):
        if expectation_number in st.session_state.input.index:
            st.session_state.input.drop(index=expectation_number, inplace=True)
            st.session_state.input.reset_index(inplace=True, drop=True)

    if not st.session_state.input.empty:
        st.write('If you want to delete an expectation, please input the row number of the expectation you want to delete below:')
        delete_input, delete_empty = st.columns([5, 5])
        with delete_input:
            expectation_number = st.number_input(label='Input the row number of the expectation you want to delete', value=None, min_value=0, max_value=st.session_state.input.shape[0]-1, label_visibility="collapsed")
        if expectation_number is not None:
            st.button(f'Delete Expectation No.{expectation_number}', on_click=delete_expectation, args=(expectation_number,), kwargs=None)
	
    # Function to append non-form inputs into dataframe
    def add_df():
        row = pd.DataFrame({'Expectations':[st.session_state.input_df_col1],
                'Columns':[st.session_state.input_df_col2],
                'Values':[st.session_state.input_df_col3]})
        if row['Expectations'][0] == 'Column values must not be null':
            row['Values'][0] = 'Not null'
        elif row['Expectations'][0] == 'Column values must be null':
            row['Values'][0] = 'Null'
        elif row['Expectations'][0] == 'Column values must be numeric (integer or float)':
            row['Values'][0] = 'Numeric'
        elif row['Expectations'][0] == 'Column values must be between 2 numbers':
            row['Values'][0] = str(st.session_state.input_df_col3) + ' - ' + str(st.session_state.input_df_col4)
        st.session_state.input = pd.concat([st.session_state.input, row])
        st.session_state.input.reset_index(inplace=True, drop=True)
	
    
    st.subheader('Input and submit your expectations')
    # Inputs created outside of a form
    select_box = st.selectbox('Expectations (required)', ('Column values must not be null', 'Column values must be null', 'Column values must be in a list', 'Column values must be numeric (integer or float)', 'Column values must match a pattern in text', 'Column values must be between 2 numbers'), key='input_df_col1')
    if select_box == 'Column values must be in a list' or select_box == 'Column values must match a pattern in text' or select_box == 'Column values must be between 2 numbers':
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
    elif select_box == 'Column values must be between 2 numbers':
        min_num, max_num = st.columns(2)
        with min_num:
            st.number_input('Min value (can be empty)', value=None, key='input_df_col3')
        with max_num:
            st.number_input('Max value (can be empty)', value=None, key='input_df_col4')
    else:
        st.number_input('Values (not required)', value=None, key='input_df_col3', disabled=True)

    if column_select:
        st.button('Submit', on_click=add_df)
