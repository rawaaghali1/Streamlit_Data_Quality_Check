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

    st.write('# Solution using a dataframe')
    
        # Create an empty dataframe on first page load, will skip on page reloads
    if 'input' not in st.session_state:
        input = pd.DataFrame({'Expectations':[],'Columns':[],'Values':[]})
        st.session_state.input = input

        # Show current data
    st.dataframe(st.session_state.input)
    
    st.write('#### Using form submission')

        # Function to append inputs from form into dataframe
    def add_dfForm():
        row = pd.DataFrame({'Expectations':[st.session_state.input_df_form_col1],
                            'Columns':[st.session_state.input_df_form_col2],
                            'Values':[st.session_state.input_df_form_col3]})
        st.session_state.input = pd.concat([st.session_state.input, row])

    # Inputs listed within a form
    dfForm = st.form(key='dfForm', clear_on_submit=True)
    with dfForm:
        a = st.selectbox('Expectations', ('Column values must not be null', 'Column values must be in a list', 'Column values must be of a certain type'), key='input_df_form_col1')
        st.multiselect('Columns', list(data.columns), key='input_df_form_col2')
        if a == 'Column values must not be null':
            st.text_input('Values', key='input_df_form_col3')
        elif a == 'Column values must be in a list':
            st.number_input('Values', step=1, key='input_df_form_col3')
        elif a == 'Column values must be of a certain type':
            st.selectbox('Values', ('Text', 'Numbers'), key='input_df_form_col3')
        st.form_submit_button(on_click=add_dfForm)

    st.write('#### Not using form submission')

    # Function to append non-form inputs into dataframe
    def add_df():
        row = pd.DataFrame({'Expectations':[st.session_state.input_df_col1],
                'Columns':[st.session_state.input_df_col2],
                'Values':[st.session_state.input_df_col3]})
        st.session_state.input = pd.concat([st.session_state.input, row])

    # Inputs created outside of a form
    select_box = st.selectbox('Expectations', ('Column values must not be null', 'Column values must be in a list', 'Column values must be of a certain type'), key='input_df_col1')
    st.multiselect('Columns', list(data.columns), key='input_df_col2')
    if select_box == 'Column values must not be null':
        st.text_input('Values', key='input_df_col3', disabled=True)
    elif select_box == 'Column values must be in a list':
        st.text_input('Values', key='input_df_col3')
    elif select_box == 'Column values must be of a certain type':
        st.selectbox('Values', ('Text', 'Numbers'), key='input_df_col3')
    st.button('Submit', on_click=add_df)
    
    st.write('## Solution using input widgets')
    # a selection for the user to specify the number of rows
    num_rows = st.slider('Number of expectations', min_value=1, max_value=10)
    # columns to lay out the inputs
    grid = st.columns(3)
    # Function to create a row of widgets (with row number input to assure unique keys)
    def add_row(row):
        with grid[0]:
            st.selectbox('Expectations', ('Column values must not be null', 'Column values must be in a list', 'Column values must be of a certain type'), key=f'input_col1{row}')
        with grid[1]:
            st.multiselect('Columns', list(data.columns), key=f'input_col2{row}')
        with grid[2]:
            st.number_input('Col3', step=1, key=f'input_col3{row}')
            #with grid[3]:
            #    st.number_input('col4', step=1, key=f'input_col4{row}',
            #                    value = st.session_state[f'input_col2{row}'] \
            #                        -st.session_state[f'input_col3{row}'],
            #                    disabled=True)
    
        # Loop to create rows of input widgets
    for r in range(num_rows):
        add_row(r)
