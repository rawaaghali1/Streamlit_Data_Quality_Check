import pandas as pd
import numpy as np
import streamlit as st
import json
import datetime
import os
import io
import time
import copy
import openpyxl
import xlsxwriter
import ast
from datetime import date
from PIL import Image
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import great_expectations as ge
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

def great_expectations_configuration(config, data):
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
        runtime_parameters={"batch_data": data},  
        batch_identifiers={"default_identifier_name": "default_identifier"},)
    validator = context.get_validator(
        batch_request=batch_request, expectation_suite_name=my_expectation_suite_name
    )
    expectation_suite = context.get_expectation_suite(expectation_suite_name)
    return context, batch_request, validator, expectation_suite

@st.cache_data
def load_data(uploaded_file_original, uploaded_file_rule):
	try:
		data = pd.read_csv(uploaded_file_original)
	except:
		data = pd.read_excel(uploaded_file_original)
	config = json.load(uploaded_file_rule)
	data.set_index('RES_NUM', drop = False, inplace = True)
	return data, config

def perform_dqc(config, dqc):
    merged_df_new = pd.DataFrame()
    for rule in config['rules']:
        expectation = rule['expectation']
        if expectation == 'columns_to_exist':
            column = rule['kwargs']['column']
            result = dqc.columns_to_exist(data, column)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")  
        elif expectation == 'column_values_to_not_be_null_if_column_a':
            column = rule['kwargs']['value_list']
            column_a = rule['kwargs']['column_a']
            value_a = rule['kwargs']['value_a']
            result = dqc.column_values_to_not_be_null_if_column_a(data,column_a, column, value_a)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")              
        elif expectation == 'expect_column_values_to_be_in_list':
            column=rule['kwargs']['column']
            value_list = rule['kwargs']['value_list']
            result = dqc.expect_column_values_to_be_in_list(data, column, value_list)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_datetime':
            result = dqc.test_column_values_to_be_of_type_datetime(data, column_A, column_B, or_equal=True)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")    
        elif expectation == 'test_column_values_to_be_positive_or_zero':
            column = rule['kwargs']['column']
            result = dqc.test_column_values_to_be_positive_or_zero(data, column)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_numeric':
            column=rule['kwargs']['column']
            result = dqc.test_column_values_to_be_of_type_numeric(data, column)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_column_values_to_be_of_type_string':
            column=rule['kwargs']['column']
            result = dqc.test_column_values_to_be_of_type_string(data, column)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'expect_column_values_to_match_regex':
            column=rule['kwargs']['column']
            regex_str = rule['kwargs']['regex']
            result = dqc.expect_column_values_to_match_regex(data, column,regex_str)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_in_list':
            column_b = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values = rule['kwargs']['list_of_values']
            set_of_values = rule['kwargs']['set_of_values']
            result = dqc.test_check_value_in_list(data, column_a, list_of_values, column_b, set_of_values)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_and_range':
            column_b = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values = rule['kwargs']['list_of_values']
            min_value = rule['kwargs']['min_value']
            max_value = rule['kwargs']['max_value']
            result = dqc.test_check_value_in_list(data, column_a, list_of_values, column_b, min_value,max_value)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_value_in_list_2_columns':
            column_c = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            column_b = rule['kwargs']['column_b']
            list_of_values_a = rule['kwargs']['list_of_values_a']
            list_of_values_b = rule['kwargs']['list_of_values_b']
            list_of_values_c = rule['kwargs']['set_of_values']
            result = dqc.test_check_value_in_list_2_columns(data, column_a, list_of_values_a, column_b, list_of_values_b, column_c, list_of_values_c)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'test_check_column_existence_if_a':
            column = rule['kwargs']['column']
            column_a = rule['kwargs']['column_a']
            list_of_values_a = rule['kwargs']['list_of_values_a']
            result = dqc.test_check_column_existence_if_a(data, column_a, list_of_values_a, column)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'expect_column_values_to_be_between':
            column= rule['kwargs']['column']
            min_value = rule['kwargs']['min_value']
            max_value = rule['kwargs']['max_value']
            result = dqc.test_column_values_to_be_between(data, column, min_value, max_value)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        elif expectation == 'column_pair_values_to_be_greater_than':
            column_A=rule['kwargs']['column_a']
            column_B=rule['kwargs']['column_b']
            or_equal=rule['kwargs']['or_equal']
            result = dqc.column_pair_values_to_be_greater_than(data, column_A, column_B,or_equal)
            df_exists = convert_dict_to_dataframe(result)
            merged_df_new = pd.concat([merged_df_new, df_exists], ignore_index=True)
            merged_df_new['run_dat'] =  datetime.datetime.now().strftime("%d/%m/%Y")
        else:
            continue
    return merged_df_new

@st.cache_data
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

@st.cache_data
def compute_dq_metrics_2(data,dq_json):
	# COMPLETENESS
	completeness = int(np.round((data.notna().to_numpy() == True).mean() * 100))

	# CONSISTENCY
	cols = ['SRC_SYS_COD','IDT_COD','PDS_COD','NPB_COD','DIL_VAL','HUM_VAL','TPR_VAL','TIM_VAL','RES_TXT']
	type_list = [str,str,numpy.int64,str,numpy.float64,numpy.float64,numpy.int64,numpy.float64,str]
	# create temporary df
	temp_data = data[cols]
	temp_type_list = []
	# get the type of columns
	for col in temp_data.columns:
		temp_type_list.append(type(temp_data[col].iloc[0]))
	con_df = pd.DataFrame({"columns" : cols, "type_actual" : type_list, "type_current" : temp_type_list})
	con_df['type_result'] = con_df['type_actual'] == con_df['type_current']
	consistency = round(con_df["type_result"].sum()/len(con_df) * 100)

	# ACCURACY
	a = 0
	b = 0
	if data['SRC_SYS_COD'].nunique() == 1:
		a = 95
	if len(list(data['PHY_STA_COD'].unique())) == 1:
		b = 95    
	c = 100 - len(data[~data['AGE_DSC'].isin(['Infant','Non-Infant'])])/len(data)
	d = 100 - len(data[~data['PHY_STA_COD'].isin(['Powder'])])/len(data)
	accuracy = round(((a+b+c+d)/400)*100)

	# RELEVANCY
	relevancy = round(((accuracy + consistency)/200)*100)

	# TIMELINESS
	today = datetime.datetime.now().date()
	#data_date = datetime.datetime.strptime(data['MNF_DAT'].iloc[0], "%Y-%m-%d %H:%M:%S.%f").date()
	#delta = today - data_date
	timeliness = 50
	#if delta.days > 150:
	#    timeliness = 30
	#elif delta.days > 120:
	#    timeliness = 50
	#elif delta.days > 90:
	#    timeliness = 60
	#elif delta.days > 60:
	#    timeliness = 80

	# create a score using checks passed and records dropped
	#checks_score = round((dq_json['checks_passed']/dq_json['checks_total'])*100)
	# get a score based on number of rows dropped
	#records_score = round((dq_json['total_records_dropped']/dq_json['total_records_actual'])*100)
	# final dq score
	total_score = round(((completeness + consistency + accuracy + relevancy + timeliness)/500) * 100)

	dq_metrics_df = pd.DataFrame({"metric" : ["completeness","completeness_l","consistency","consistency_l","accuracy","accuracy_l","relevancy","relevancy_l","timeliness","timeliness_l"], \
	"percentage" : [completeness,100-completeness,consistency,100-consistency,accuracy,100-accuracy,relevancy,100-relevancy,timeliness,100-timeliness]})

	return dq_metrics_df, total_score

# basic metrics like null values, unique values
@st.cache_data
def compute_basic_metrics(data):
	basic_metrics_df = pd.DataFrame({"null_values" : data.isna().sum(), "unique_values" : data.nunique()})
	return basic_metrics_df

# get column checks results
@st.cache_data
def compute_column_checks_results(dq_json):
	columns = []
	checks = []
	results = []
	for i in dq_json:
		columns.append(i['column'])
		checks.append(ast.literal_eval(i['notes'])[1])
		results.append(i['success'])
	column_results_df = pd.DataFrame({'columns' : columns, 'checks' : checks, 'results' : results})
	return column_results_df

@st.cache_data
#def gen_profile_report(df, *report_args, **report_kwargs):
#    return ProfileReport(df, *report_args, **report_kwargs)
def gen_profile_report(df):
	return ProfileReport(df, progress_bar=True)

uploaded_file_original = st.sidebar.file_uploader("Upload your raw data", type=['csv', 'xlsx'], help='Only .csv or .xlsx file is supported.')
uploaded_file_rule = st.sidebar.file_uploader("Upload your json file", type='json', help='Only .json file for rules is supported.')
if uploaded_file_original is not None and uploaded_file_rule is not None:
    	data, config = load_data(uploaded_file_original, uploaded_file_rule)
    	context, batch_request, validator, expectation_suite = great_expectations_configuration(config, data)
    	# Instantiate the Data_quality_check class
    	dqc = Data_quality_check()
    	merged_df_new = perform_dqc(config, dqc)
    
    	st.write(merged_df_new.shape)

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
	
	dq_json = merged_df_new.to_dict(orient ='records')
	
	dq_metrics_df_2, total_score_2 = compute_dq_metrics_2(data,dq_json)

	#table_results_df = compute_table_checks_results(dq_json)

	column_results_df = compute_column_checks_results(dq_json)

	basic_metrics_df= compute_basic_metrics(data)

	layout_plot, layout_dist = layout()

	data_for_profiling = data.reset_index(drop=True)
	if data.shape[0]>50000:
		data_for_profiling = data_for_profiling.sample(50000)
	else:
		data_for_profiling = data_for_profiling
	pr = gen_profile_report(data_for_profiling)

	# Heading
	# put logo image on the top right
	image = Image.open('assets/danone_ds_logo.png')
	col1, col2 = st.columns([8, 1])
	with col1:
		st.title('Data Quality Dashboard')
	with col2:
		st.write('Powered by:')
		st.image(image, width=90, output_format='PNG')

	st.subheader('Metrics')

	###### ROW 0 #######
	accuracy, relevancy, completeness, timeliness, consistency, overall_score = st.columns([1,1,1,1,1,1])

	with accuracy:
		st.markdown('Accuracy', help='Score based on the percentage of rows that have correct values for certain columns')		      
		fig = px.pie(dq_metrics_df_2[dq_metrics_df_2['metric'].str.contains('accuracy')], names = 'metric', values = 'percentage', color = 'metric', \
			hole = 0.5,color_discrete_map={"accuracy" : '#19AA6E',"accuracy_l" : '#0E1117'})
		fig.update_traces(textinfo='none')
		layout_plot['annotations'][0]['text'] = str(dq_metrics_df_2[dq_metrics_df_2['metric'] == "accuracy"]["percentage"].iloc[0])
		fig.update_layout(layout_plot)
		st.plotly_chart(fig, use_container_width=True)

	with relevancy:
		st.markdown('Relevancy', help='Score based on the performance of Accuracy and Consistency')		      
		fig = px.pie(dq_metrics_df_2[dq_metrics_df_2['metric'].str.contains('relevancy')], names = 'metric', values = 'percentage', color = 'metric', \
			hole = 0.5,color_discrete_map={"relevancy" : '#19AA6E',"relevancy_l" : '#0E1117'})
		fig.update_traces(textinfo='none')
		layout_plot['annotations'][0]['text'] = str(dq_metrics_df_2[dq_metrics_df_2['metric'] == "relevancy"]["percentage"].iloc[0])
		fig.update_layout(layout_plot)
		st.plotly_chart(fig, use_container_width=True)

	with completeness:
		st.markdown('Completeness', help='Score based on the percentage of rows that are non-missing for certain columns')	
		fig = px.pie(dq_metrics_df_2[dq_metrics_df_2['metric'].str.contains('completeness')], names = 'metric', values = 'percentage', color = 'metric', \
			hole = 0.5,color_discrete_map={"completeness" : '#19AA6E',"completeness_l" : '#0E1117'})
		fig.update_traces(textinfo='none')
		layout_plot['annotations'][0]['text'] = str(dq_metrics_df_2[dq_metrics_df_2['metric'] == "completeness"]["percentage"].iloc[0])
		fig.update_layout(layout_plot)
		st.plotly_chart(fig, use_container_width=True)

	with timeliness:
		st.markdown('Timeliness', help='Score based on how recently the data is ingested')	
		fig = px.pie(dq_metrics_df_2[dq_metrics_df_2['metric'].str.contains('timeliness')], names = 'metric', values = 'percentage', color = 'metric', \
			hole = 0.5,color_discrete_map={"timeliness" : '#19AA6E',"timeliness_l" : '#0E1117'})
		fig.update_traces(textinfo='none')
		layout_plot['annotations'][0]['text'] = str(dq_metrics_df_2[dq_metrics_df_2['metric'] == "timeliness"]["percentage"].iloc[0])
		fig.update_layout(layout_plot)
		st.plotly_chart(fig, use_container_width=True)

	with consistency:
		st.markdown('Consistency', help='Score based on the percentage of rows that have correct data types for certain columns')
		fig = px.pie(dq_metrics_df_2[dq_metrics_df_2['metric'].str.contains('consistency')], names = 'metric', values = 'percentage', color = 'metric', \
			hole = 0.5,color_discrete_map={"consistency" : '#19AA6E',"consistency_l" : '#0E1117'})
		fig.update_traces(textinfo='none')
		layout_plot['annotations'][0]['text'] = str(dq_metrics_df_2[dq_metrics_df_2['metric'] == "consistency"]["percentage"].iloc[0])
		fig.update_layout(layout_plot)
		st.plotly_chart(fig, use_container_width=True)

	with overall_score:
		st.metric(label="DQ Overall Score", value=f"{total_score_2}", delta = f"-{100 - total_score_2}", help='Calculated as the average score of the 5 metrics')
		st.metric(label="Total column checks", value=f"{len(dq_json)}", delta = f"-{sum(1 for element in dq_json if element['success']=='FALSE' or element['success']==False)} checks failed")

	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

	###### ROW 1 #######
	#PHY_STA_COD, TIM_VAL, TPR_VAL, UNT_COD, FAT_CNT_TXT, NIT_FLU_TXT = st.columns(6)
	#AGE_DSC, PRO_HDR_TXT, DEN_VAL, DIL_VAL, HUM_VAL, overall_score = st.columns(6)
	#with PHY_STA_COD:
	#    st.write('PHY_STA_COD')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('PHY_STA_COD')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"PHY_STA_COD" : '#19AA6E',"PHY_STA_COD_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "PHY_STA_COD"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with TIM_VAL:
	#    st.write('TIM_VAL')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('TIM_VAL')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"TIM_VAL" : '#19AA6E',"TIM_VAL_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "TIM_VAL"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with TPR_VAL:
	#    st.write('TPR_VAL')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('TPR_VAL')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"TPR_VAL" : '#19AA6E',"TPR_VAL_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "TPR_VAL"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with UNT_COD:
	#    st.write('UNT_COD')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('UNT_COD')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"UNT_COD" : '#19AA6E',"UNT_COD_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "UNT_COD"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with FAT_CNT_TXT:
	#    st.write('FAT_CNT_TXT')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('FAT_CNT_TXT')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"FAT_CNT_TXT" : '#19AA6E',"FAT_CNT_TXT_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "FAT_CNT_TXT"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with NIT_FLU_TXT:
	#    st.write('NIT_FLU_TXT')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('NIT_FLU_TXT')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"NIT_FLU_TXT" : '#19AA6E',"NIT_FLU_TXT_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "NIT_FLU_TXT"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with AGE_DSC:
	#    st.write('AGE_DSC')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('AGE_DSC')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"AGE_DSC" : '#19AA6E',"AGE_DSC_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "AGE_DSC"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with PRO_HDR_TXT:
	#    st.write('PRO_HDR_TXT')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('PRO_HDR_TXT')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"PRO_HDR_TXT" : '#19AA6E',"PRO_HDR_TXT_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "PRO_HDR_TXT"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with DEN_VAL:
	#    st.write('DEN_VAL')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('DEN_VAL')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"DEN_VAL" : '#19AA6E',"DEN_VAL_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "DEN_VAL"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with DIL_VAL:
	#    st.write('DIL_VAL')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('DIL_VAL')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"DIL_VAL" : '#19AA6E',"DIL_VAL_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "DIL_VAL"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with HUM_VAL:
	#    st.write('HUM_VAL')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('HUM_VAL')], names = 'metric', values = 'percentage', color = 'metric', \
	#        hole = 0.5,color_discrete_map={"HUM_VAL" : '#19AA6E',"HUM_VAL_l" : '#0E1117'})
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "HUM_VAL"]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	#with overall_score:
	#    st.metric(label="DQ score", value=f"{total_score}", delta = f"-{100 - total_score}")

	#st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

	###### ROW 2 #######
	# heading
	table_checks_heading, column_checks_heading = st.columns([1,1])
	with table_checks_heading:
		st.subheader('Column checks')

	#with column_checks_heading:
	#    st.subheader('Metrics')


	######## ROW 3 #######
	#table_checks_radio, table_checks_stats, column_checks_radio, column_checks_stats = st.columns([1,1,1,1])
	column_checks_radio, column_checks_stats = st.columns([1,1])

	#TABLE CHECKS
	#with table_checks_radio:
	#    table_checks_radio = st.radio(label = 'Checks status', options = ('Pass', 'Fail'), key = 'table_checks_radio')

	#with table_checks_stats:
	#    st.metric(label="Total checks", value=f"{len(data)}", delta = f"-{dq_json['checks_table_level']['checks_table_failed']} checks failed")


	# COLUMN CHECKS
	# radio button for checks passed and failed
	with column_checks_radio:
		column_checks_radio = st.radio(label = 'Checks status', options = ('Pass', 'Fail'), key = 'column_checks_radio')

	# overall checks passed and failed
	#with column_checks_stats:
	#    st.metric(label="Total checks", value=f"{len(dq_json)}", delta = f"-{sum(1 for element in dq_json if element['success']=='FALSE' or element['success']==False)} checks failed")


	###### ROW 4 #######
	#table_checks_select, column_checks_col_select, column_checks_select = st.columns([2,1,1])

	# table checks select
	#with table_checks_select:
	#    if table_checks_radio == 'Pass':
	#       table_checks_options = tuple(table_results_df[table_results_df['results']==True]['checks'])
	#    else:
	#        table_checks_options = tuple(table_results_df[table_results_df['results']==False]['checks'])
		
	#    table_checks_selectbox = st.selectbox(
	#    'Select a check',
	#    table_checks_options,
	#    key = 'table_checks_selectbox'
	#    )
	column_checks_col_select, column_checks_select = st.columns([1,1])
	# column select a column
	with column_checks_col_select:
		if column_checks_radio == 'Pass':
			column_checks_col_options = tuple(column_results_df[(column_results_df['results']==True)|(column_results_df['results']=='TRUE')]['columns'].unique())
		else:
			column_checks_col_options = tuple(column_results_df[(column_results_df['results']==False)|(column_results_df['results']=='FALSE')]['columns'].unique())
		
		column_checks_col_selectbox = st.selectbox(
		'Select a column',
		column_checks_col_options,
		key = 'column_checks_col_selectbox'
		)

	# column checks select 
	with column_checks_select:
		if column_checks_radio == 'Pass':
			column_checks_options = tuple(column_results_df[(column_results_df['columns'] == column_checks_col_selectbox) & ((column_results_df['results']==True)|(column_results_df['results']=='TRUE'))]['checks'])
		else:
			column_checks_options = tuple(column_results_df[(column_results_df['columns'] == column_checks_col_selectbox) & ((column_results_df['results']==False)|(column_results_df['results']=='FALSE'))]['checks'])
		column_checks_selectbox = st.selectbox(
		'Select a check',
		column_checks_options,
		key = 'column_checks_selectbox'
		)

	###### ROW 5 #######

	table_checks_json, column_checks_json = st.columns([5, 5])

	# table checks json
	#with table_checks_json:
	#    try:
	#        table_json = dq_json['checks_table_level'][table_checks_selectbox]
	#        st.json(table_json)
	#    except KeyError:
	#        st.json({'checks' : 'None'})

	# columns checks json

	with column_checks_json:
		try:
			for i in dq_json:
				if column_checks_selectbox[0:15] in i['notes']:
					i_subset = {}
					i_subset['Column'] = i['column']
					i_subset['Expectation'] = ast.literal_eval(i['notes'])[1]
					i_subset['Expectation type'] = i['Problem Type']
					i_subset['Success'] = i['success']
					i_subset['Partial unexpected value list'] = i['partial_unexpected_list']
					i_subset['Partial unexpected index list'] = i['partial_unexpected_index_list']
					i_subset['Unexpected value counts'] = i['partial_unexpected_counts']
					i_subset['Run date'] = i['run_dat']
					st.json(i_subset)
					#st.json(i)
		except KeyError:
			st.json({'checks' : 'None'})
			
	with table_checks_json:
		for i in dq_json:
			if column_checks_col_selectbox == i['column']:
				column_metrics_df = pd.DataFrame(
					{'Type': ['Expected', 'Missing', 'Unexpected'], 
					 'Count': [int(i['element_count'])-int(i['unexpected_count'])-int(i['missing_count']), int(i['missing_count']), int(i['unexpected_count'])]
					}
				)
				fig = px.pie(column_metrics_df, values='Count', names='Type', color = 'Type', hole = 0.5, \
						 color_discrete_map={"Expected": '#19AA6E', "Missing": '#A9DFC9', "Unexpected": '#FE0000'})
				st.plotly_chart(fig, use_container_width=True)
				break

	#    color_discrete_map = {}
	#    color_discrete_map[column_checks_col_selectbox] = '#19AA6E'
	#    color_discrete_map[column_checks_col_selectbox+'_l'] = '#0E1117'
	#    st.write('Data Quality Score')
	#    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains(column_checks_col_selectbox)], names = 'metric', values = 'percentage', color = 'metric',\
	#        hole = 0.5, color_discrete_map=color_discrete_map)
	#    fig.update_traces(textinfo='none')
	#    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == column_checks_col_selectbox]["percentage"].iloc[0])
	#    fig.update_layout(layout_plot)
	#    st.plotly_chart(fig, use_container_width=True)

	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


	###### ROW 6 #######
	# barplot and distribution 
	st.subheader('Visualization')
	bar_plot, gap_bar_distribution, distribution_plot = st.columns([6, 1, 6])
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
			tuple(data.select_dtypes([np.number]).columns),
			key = 'dist_selectbox'
		)
		fig = px.histogram(data, x=dist_selectbox)
		#fig.update_layout(layout_dist)
		st.plotly_chart(fig, use_container_width=True)

	#corr_plot, unknown_plot  = st.columns([1,1])
	#with corr_plot:
	#    st.write('Correlation heatmap')
	#    data_quantitative = data[["DEN_VAL", "DIL_VAL", "HUM_VAL", "TPR_VAL", "TIM_VAL"]]
	#    fig = px.imshow(data_quantitative.corr(numeric_only=True))
	#    st.write(fig)
	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

	###### ROW 7 #######
	st.subheader('Data Profiling', help='Data profiling is the process of examining, analyzing, and creating useful summaries of data.')
	if st.button('Generate a Data Profiling report', help='The process can take up to 1 minute. If you encounter an error message, please try to refresh the page.'):
		with st.expander("Report", expanded=True):
			st_profile_report(pr)

	st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
