# import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import numpy
import json
import time
import copy
import datetime
import openpyxl
import ast
from datetime import date
from PIL import Image
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

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


# Add a selectbox to the sidebar:
sb_selectbox = st.sidebar.selectbox(
    'Select a dataset',
    ('Batch 1', 'Batch 2'),
    key = 'sb_selectbox'
)

# loading the data
@st.cache_data
def load_data(sb_selectbox):
    if sb_selectbox == 'Batch 1':
        data = pd.read_excel("data/test_raw_file.xlsx")
        dq_json = json.load(open("result/test_original.json"))
    else:
        data = pd.read_csv("data/test_raw_file_2.csv")
        dq_json = json.load(open("result/test_original_2.json"))
#    lowercase = lambda x: str(x).lower()
#    data.rename(lowercase, axis='columns', inplace=True)
    data.set_index('RES_NUM', drop = False, inplace = True)
    return data, dq_json

# loading the data
#@st.cache_data
#def load_data():
#    data = pd.read_excel("data/test_raw_file.xlsx")
#    dq_json = json.load(open("result/test_original.json"))
#    lowercase = lambda x: str(x).lower()
#    data.rename(lowercase, axis='columns', inplace=True)
#    data.set_index('RES_NUM', drop = False, inplace = True)
#    return data, dq_json

# compute the measures of data quality based on project criteria
@st.cache_data
def compute_dq_metrics(data, dq_json):
    # PHY_STA_COD
    PHY_STA_COD = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'PHY_STA_COD'][0])
    # TIM_VAL
    TIM_VAL = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'TIM_VAL'][0])
    # TPR_VAL
    TPR_VAL = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'TPR_VAL'][0])
    # UNT_COD
    UNT_COD = int(100-[float(item['missing_percent'])+float(item['unexpected_percent_total']) for item in dq_json if item['column'] == 'UNT_COD'][0])
    # FAT_CNT_TXT
    FAT_CNT_TXT = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'FAT_CNT_TXT'][0])
    # NIT_FLU_TXT
    NIT_FLU_TXT = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'NIT_FLU_TXT'][0])
    # AGE_DSC
    AGE_DSC = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'AGE_DSC'][0])
    # PRO_HDR_TXT
    PRO_HDR_TXT = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'PRO_HDR_TXT'][0])
    # DEN_VAL
    DEN_VAL = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'DEN_VAL'][0])
    # DIL_VAL
    DIL_VAL = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'DIL_VAL'][0])
    # HUM_VAL
    HUM_VAL = int(100-[float(item['missing_percent']+item['unexpected_percent_total']) for item in dq_json if item['column'] == 'HUM_VAL'][0])
    
    # create a score using checks passed and records dropped
    # checks_score = round((dq_json['checks_passed']/dq_json['checks_total'])*100)
    # get a score based on number of rows dropped
    # records_score = round((dq_json['total_records_dropped']/dq_json['total_records_actual'])*100)
    # final dq score
    total_score = round(((PHY_STA_COD + TIM_VAL + TPR_VAL + UNT_COD + FAT_CNT_TXT + NIT_FLU_TXT + AGE_DSC + PRO_HDR_TXT + DEN_VAL + DIL_VAL + HUM_VAL)/1100) * 100)

    dq_metrics_df = pd.DataFrame({"metric" : ["PHY_STA_COD","PHY_STA_COD_l","TIM_VAL","TIM_VAL_l","TPR_VAL","TPR_VAL_l","UNT_COD","UNT_COD_l",\
					     "FAT_CNT_TXT","FAT_CNT_TXT_l","NIT_FLU_TXT","NIT_FLU_TXT_l","AGE_DSC","AGE_DSC_l","PRO_HDR_TXT","PRO_HDR_TXT_l",\
					     "DEN_VAL","DEN_VAL_l","DIL_VAL","DIL_VAL_l","HUM_VAL","HUM_VAL_l"], \
    "percentage" : [PHY_STA_COD,100-PHY_STA_COD,TIM_VAL,100-TIM_VAL,TPR_VAL,100-TPR_VAL,UNT_COD,100-UNT_COD,\
		   FAT_CNT_TXT,100-FAT_CNT_TXT,NIT_FLU_TXT,100-NIT_FLU_TXT,AGE_DSC,100-AGE_DSC,PRO_HDR_TXT,100-PRO_HDR_TXT,\
		   DEN_VAL,100-DEN_VAL,DIL_VAL,100-DIL_VAL,HUM_VAL,100-HUM_VAL]})

    return dq_metrics_df, total_score

# compute the measures of general data quality based on 3-5 criteria
@st.cache_data
def compute_dq_metrics_2(data,dq_json):
    # COMPLETENESS
    completeness = int(np.round((data.notna().to_numpy() == True).mean() * 100))

    # CONSISTENCY
    cols = ['SRC_SYS_COD','IDT_COD','PDS_COD','NPB_COD','DIL_VAL','HUM_VAL',
                        'TPR_VAL','TIM_VAL','RES_TXT']
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

# get table checks results
#@st.cache_data
#def compute_table_checks_results(dq_json):
#    table_checks = []
#    table_results = []
#    for i in  dq_json['checks_table_level'].keys():
#        if type(dq_json['checks_table_level'][i]) not in (int,bool):
#            table_checks.append(i)
#            table_results.append(dq_json['checks_table_level'][i]['result'])
#
#    table_results_df = pd.DataFrame({"checks" : table_checks, "results" : table_results})
#
#    return table_results_df

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
def gen_profile_report(df, *report_args, **report_kwargs):
    return ProfileReport(df, *report_args, **report_kwargs)

# run the functions
data, dq_json= load_data(sb_selectbox)

dq_metrics_df, total_score = compute_dq_metrics(data,dq_json)

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
    st.markdown('DQ Overall Score', help='Calculated as the average score of the 5 metrics')
    #st.metric(label="DQ Overall Score", value=f"{total_score_2}", delta = f"-{100 - total_score_2}")
    st.metric(label=None, value=f"{total_score_2}", delta = f"-{100 - total_score_2}")
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
bar_plot, distribution_plot  = st.columns([1,1])

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

    st.bar_chart(basic_metrics_df[variable])

with distribution_plot:
    dist_selectbox = st.selectbox(
    'Select a column to get distribution',
    tuple(data.select_dtypes([np.number]).columns),
    key = 'dist_selectbox'
    )
    fig = px.histogram(data, x=dist_selectbox)
    fig.update_layout(layout_dist)
    st.plotly_chart(fig, use_container_width=True)

#corr_plot, unknown_plot  = st.columns([1,1])
#with corr_plot:
#    st.write('Correlation heatmap')
#    data_quantitative = data[["DEN_VAL", "DIL_VAL", "HUM_VAL", "TPR_VAL", "TIM_VAL"]]
#    fig = px.imshow(data_quantitative.corr(numeric_only=True))
#    st.write(fig)
#st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

###### ROW 7 #######
st.subheader('Data Profiling')
with st.expander("Report", expanded=True):
	st_profile_report(pr)

###### ROW 8 #######
# signal 
# st.write('Signal')
# st.line_chart(data['crssi_dbm'])
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# raw dataset
st.subheader('View raw dataset')
if st.checkbox('Show dataframe'):
    data_for_profiling

