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

    st.subheader('List of expectations')
    st.write('The expectations you have input are reflected in the table below.')
    
    # Create an empty dataframe on first page load, will skip on page reloads
    if 'input' not in st.session_state:
        input = pd.DataFrame({'Expectations':[],'Columns':[],'Values':[]})
        st.session_state.input = input

    # Show current data
    st.dataframe(st.session_state.input, hide_index=True)
    st.write('To clear all expectations, please click the button in the top right corner and choose "Rerun".')
    
    # Function to append non-form inputs into dataframe
    def add_df():
        row = pd.DataFrame({'Expectations':[st.session_state.input_df_col1],
                'Columns':[st.session_state.input_df_col2],
                'Values':[st.session_state.input_df_col3]})
        st.session_state.input = pd.concat([st.session_state.input, row])
    
    st.subheader('Input and submit your expectations')
    # Inputs created outside of a form
    select_box = st.selectbox('Expectations', ('Column values must not be null', 'Column values must be in a list', 'Column values must be numeric (integer or float)'), key='input_df_col1')
    if select_box == 'Column values must be in a list':
        st.selectbox('Columns', list(data.columns), key='input_df_col2', placeholder='Select only 1 column')
    else:
        st.multiselect('Columns', list(data.columns), key='input_df_col2', placeholder='Select 1 or more columns')
    if select_box == 'Column values must not be null':
        st.text_input('Values', key='input_df_col3', disabled=True)
    elif select_box == 'Column values must be in a list':
        text_input = st.text_input('Values', key='input_df_col3')
        if text_input:
            st.write("You entered: ", text_input)
    elif select_box == 'Column values must be numeric (integer or float)':
        st.text_input('Values', key='input_df_col3', disabled=True)
    st.button('Submit', on_click=add_df)

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
                 "kwargs": {"column":row['Columns'][0],
                           "value_list":[x.strip() for x in row['Values'].split(',')]
                           }
                }
            )
        elif row['Expectations'] == 'Column values must be numeric (integer or float)':
            config['rules'].append(
                {"expectation": "test_column_values_to_be_of_type_numeric",
                 "kwargs": {"column": row['Columns']}
                }
            )
    
    json_string = json.dumps(config)
    st.download_button(
        label="Download your json file",
        file_name="expectations.json",
        mime="application/json",
        data=json_string,
    )
