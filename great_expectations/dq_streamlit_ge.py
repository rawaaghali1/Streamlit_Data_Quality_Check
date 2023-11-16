import pandas as pd
import numpy as np
import streamlit as st
import json
import datetime
import os
import io
import plotly.express as px
import great_expectations as ge
import xlsxwriter
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from great_expectations.core.batch import BatchRequest
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.yaml_handler import YAMLHandler
from great_expectations.data_context import FileDataContext
from great_expectations.data_context import BaseDataContext
from great_expectations.util import get_context
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.validator.validator import Validator
from great_expectations.dataset import PandasDataset
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from expectations_func import Data_quality_check
from utile_functions import create_df_from_validation_result
from utile_functions import convert_dict_to_dataframe

# set the page configuration
st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title="Data Quality App",  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

# hide the blurb and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# define layout for plotly graphs
@st.cache_data
def layout():
    layout_plot = dict(paper_bgcolor = '#0E1117',
        plot_bgcolor = '#0E1117',
        width = 200,
        height = 200,
        margin = dict(t = 0, l = 0, r = 0, b = 0),
        hovermode = False,
        showlegend = False,
        xaxis_title = "",
        yaxis_title = "",
        font=dict(family="Arial",
                size=10,
            ),
        xaxis=dict(
            showline = True,
            showgrid = True,
            showticklabels=True,
            gridcolor = "#dfe3e8",
            linecolor = '#1E3246'), 
        yaxis=dict(
            showline = True,
            showgrid = True,
            showticklabels=True,
            gridcolor = "#dfe3e8",
            linecolor = '#1E3246'),
            shapes = [],
            bargap = 0,
            annotations=[dict(text='', x=0.5, y=0.5, showarrow=False)])
    layout_dist = layout_plot.copy()
    layout_dist['hovermode'] = 'x'
    return layout_plot, layout_dist

def great_expectations_configuration(config, df):
    yaml = YAMLHandler()
    # config properties
    expectation_suite_name = config['expectation_suite_name']
    project_name = config['project_name']
    datasource_name = config['datasource_name']
    checkpoint_name = config['checkpoint_name']
    data_asset_name = config['data_asset_name']
    run_name_template=config["run_name_template"]
    rules = config['rules']
    if 'sheet_name' in config:
        sheet_name = config['sheet_name']
    else:
        sheet_name = None

    context = get_context()
    
    datasource_config = {
        "name": datasource_name,
        "class_name": "Datasource",
        "module_name": "great_expectations.datasource",
        "execution_engine": {
            "module_name": "great_expectations.execution_engine",
            "class_name": "PandasExecutionEngine",
        },
        "data_connectors": {
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "module_name": "great_expectations.datasource.data_connector",
                "batch_identifiers": ["default_identifier_name"],
            },
        },
    }
    
    context.test_yaml_config(yaml.dump(datasource_config))
    context.add_datasource(**datasource_config)
    my_expectation_suite_name = expectation_suite_name
    try:
      context.add_expectation_suite(expectation_suite_name=my_expectation_suite_name)
    except:
      pass
    
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name=data_asset_name,  
        runtime_parameters={"batch_data": df},  
        batch_identifiers={"default_identifier_name": "default_identifier"},)
    validator = context.get_validator(
        batch_request=batch_request, expectation_suite_name=my_expectation_suite_name
    )
    expectation_suite = context.get_expectation_suite(expectation_suite_name)
    return context, batch_request, validator, expectation_suite

# basic metrics like null values, unique values
@st.cache_data
def compute_basic_metrics(data):
	basic_metrics_df = pd.DataFrame({"null_values" : data.isna().sum(), "unique_values" : data.nunique()})
	return basic_metrics_df

@st.cache_data
#def gen_profile_report(df, *report_args, **report_kwargs):
#    return ProfileReport(df, *report_args, **report_kwargs)
def gen_profile_report(df):
	return ProfileReport(df, progress_bar=True)

uploaded_file_original = st.sidebar.file_uploader("Upload your raw data", type=['csv', 'xlsx'], help='Only .csv or .xlsx file is supported.')
rules_yes_or_not = st.sidebar.radio(
    "Do you have a json file for column checks?",
    ["No", "Yes"],
    captions = ["You can perform column checks and download an Excel report", "You can check some basic visualizations of your data"])

if rules_yes_or_not == 'Yes':
	uploaded_file_rule = st.sidebar.file_uploader("Upload your json file", type='json', help='Only .json file for rules is supported.')
elif rules_yes_or_not == 'No' and uploaded_file_original is not None:
	try:
		df = pd.read_csv(uploaded_file_original)
	except:
		df = pd.read_excel(uploaded_file_original)
	###### ROW 6 #######
	# barplot and distribution 
	basic_metrics_df= compute_basic_metrics(df)
	st.subheader('Visualization')
	bar_plot, gap_bar_distribution, distribution_plot = st.columns([7, 1, 7])
	#bar_plot, distribution_plot = st.columns([1, 1])

	with bar_plot:
		bar_selectbox = st.selectbox(
			'Select an option',
			('null values', 'unique values'),
			key = 'bar_selectbox'
		)
		if bar_selectbox == "null values":
			variable = 'null_values'
		else:
			variable = 'unique_values'
		#st.bar_chart(basic_metrics_df[variable])
		fig = px.bar(basic_metrics_df[variable])
		fig.update_layout(showlegend=False)
		st.plotly_chart(fig, use_container_width=True)
		
	with gap_bar_distribution:
		st.write('')

	with distribution_plot:
		dist_selectbox = st.selectbox(
			'Select a column to get distribution',
			tuple(df.select_dtypes([np.number]).columns),
			key = 'dist_selectbox'
		)
		fig = px.histogram(df, x=dist_selectbox)
		#fig.update_layout(layout_dist)
		st.plotly_chart(fig, use_container_width=True)

	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

	###### ROW 7 #######
	st.subheader('Data Profiling', help='Data profiling is the process of examining, analyzing, and creating useful summaries of data.')
	if st.button('Generate a Data Profiling report', help='The process can take up to 1 minute. If you encounter an error message, please try to refresh the page.'):
		pr = gen_profile_report(df)
		with st.expander("Report", expanded=True):
			st_profile_report(pr)

	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if uploaded_file_original is not None and uploaded_file_rule is not None:
    try:
	    df = pd.read_csv(uploaded_file_original)
    except:
	    df = pd.read_excel(uploaded_file_original)
    config = json.load(uploaded_file_rule)
    context, batch_request, validator, expectation_suite = great_expectations_configuration(config, df)
    # Instantiate the Data_quality_check class
    dqc = Data_quality_check()
    merged_df_new = pd.DataFrame()
    print(merged_df_new.shape)
    for rule in config['rules']:
        expectation = rule['expectation']
        if expectation == 'columns_to_exist':
            column = rule['kwargs']['column']
            result = dqc.columns_to_exist(df, column)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")  
        elif expectation == 'column_values_to_not_be_null_if_column_a':
            column = rule['kwargs']['value_list']
            column_a = rule['kwargs']['column_a']
            value_a = rule['kwargs']['value_a']
            result = dqc.column_values_to_not_be_null_if_column_a(df,column_a, column, value_a)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")              
        elif expectation == 'expect_column_values_to_be_in_list':
            column=rule['kwargs']['column']
            value_list = rule['kwargs']['value_list']
            result = dqc.expect_column_values_to_be_in_list(df, column, value_list)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_datetime':
            result = dqc.test_column_values_to_be_of_type_datetime(df, column_A, column_B, or_equal=True)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")    
        elif expectation == 'test_column_values_to_be_positive_or_zero':
            column=rule['kwargs']['column']
            result = dqc.test_column_values_to_be_positive_or_zero(df, column)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_numeric':
            column=rule['kwargs']['column']
            result = dqc.test_column_values_to_be_of_type_numeric(df, column)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_string':
            column=rule['kwargs']['column']
            result = dqc.test_column_values_to_be_of_type_string(df, column)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'expect_column_values_to_match_regex':
            column=rule['kwargs']['column']
            regex_str = rule['kwargs']['regex']
            result = dqc.expect_column_values_to_match_regex(df, column,regex_str)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_in_list':
            column_b = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values = rule['kwargs']['list_of_values']
            set_of_values = rule['kwargs']['set_of_values']
            result = dqc.test_check_value_in_list(df, column_a, list_of_values, column_b, set_of_values)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_and_range':
            column_b = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values = rule['kwargs']['list_of_values']
            min_value = rule['kwargs']['min_value']
            max_value = rule['kwargs']['max_value']
            result = dqc.test_check_value_in_list(df, column_a, list_of_values, column_b, min_value,max_value)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_in_list_2_columns':
            column_c = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            column_b = rule['kwargs']['column_b']
            list_of_values_a = rule['kwargs']['list_of_values_a']
            list_of_values_b = rule['kwargs']['list_of_values_b']
            list_of_values_c = rule['kwargs']['set_of_values']
            result = dqc.test_check_value_in_list_2_columns(df, column_a, list_of_values_a, column_b, list_of_values_b, column_c, list_of_values_c)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_column_existence_if_a':
            column = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values_a = rule['kwargs']['list_of_values_a']
            result = dqc.test_check_column_existence_if_a(df, column_a, list_of_values_a, column)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'expect_column_values_to_be_between':
                column= rule['kwargs']['column']
                min_value = rule['kwargs']['min_value']
                max_value = rule['kwargs']['max_value']
                result = dqc.test_column_values_to_be_between(df, column, min_value, max_value)
                df_exists=convert_dict_to_dataframe(result)
                merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
                merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'column_pair_values_to_be_greater_than':
            column_A=rule['kwargs']['column_a']
            column_B=rule['kwargs']['column_b']
            or_equal=rule['kwargs']['or_equal']
            result = dqc.column_pair_values_to_be_greater_than(df, column_A, column_B,or_equal)
            df_exists=convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        else:
            continue
    
    st.write(merged_df_new.shape)
    def classify_problem(description):
        keywords = {
            "Classification": ["expected values", "must be equal to",'is in'],
            "Out-of-Range": ["must be between", "positive"],
            "Data-type": ["data type", "numeric", "string"],
            "Missing-value": ["must not be null"],
            "Pattern": ["must match"]
        }
        # Convert lists of strings to a single string and convert to lowercase
        if isinstance(description, list):
            description = ', '.join(description).lower()
        else:
            description = description.lower()
        for problem_type, words in keywords.items():
            if any(word in description for word in words):
                return problem_type
        return "unknown"

    # Apply the classification function to determine the problem type
    merged_df_new['Problem Type'] = merged_df_new['notes'].apply(classify_problem)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
	    # Write each dataframe to a different worksheet.
	    merged_df_new.to_excel(writer, sheet_name='Sheet1', index=False)
	
    st.sidebar.download_button(
       label = "Press to Download",
       data = buffer,
       file_name = "file.xlsx",
       mime = "application/vnd.ms-excel"
    )
