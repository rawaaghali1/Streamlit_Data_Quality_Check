# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import numpy
import json
import time
import copy
import datetime
from datetime import date

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
@st.cache(allow_output_mutation=True)
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
    ('Dataset 1', 'Dataset 2'),
    key = 'sb_selectbox'
)

# loading the data
@st.cache(allow_output_mutation=True)
def load_data(sb_selectbox):
    if sb_selectbox == 'Dataset 1':
        data = pd.read_csv("data/dataset_1.csv")
        dq_json = json.load(open("result/dq_result.json"))
    else:
        data = pd.read_csv("data/dataset_2.csv")
        dq_json = json.load(open("result/dq_result.json"))
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # data.set_index('kp_in_track', drop = False, inplace = True)
    return data, dq_json

# compute the measures of data quality based on project criteria
# deleted

# basic metrics like null values, unique values
@st.cache(allow_output_mutation=True)
def compute_basic_metrics(data):
    basic_metrics_df = pd.DataFrame({"null_values" : data.isna().sum(), "unique_values" : data.nunique()})
    return basic_metrics_df

# get table checks results
@st.cache(allow_output_mutation=True)
def compute_table_checks_results(dq_json):
    table_checks = []
    table_results = []
    for i in  dq_json['checks_table_level'].keys():
        if type(dq_json['checks_table_level'][i]) not in (int,bool):
            table_checks.append(i)
            table_results.append(dq_json['checks_table_level'][i]['result'])

    table_results_df = pd.DataFrame({"checks" : table_checks, "results" : table_results})

    return table_results_df

# get column checks results
@st.cache(allow_output_mutation=True)
def compute_column_checks_results(dq_json):
    columns = []
    checks = []
    results = []
    for i in dq_json['checks_column_level'].keys():
        if type(dq_json['checks_column_level'][i]) not in (int,bool):
            for j in dq_json['checks_column_level'][i].keys():
                columns.append(i)
                checks.append(j)
                results.append(dq_json['checks_column_level'][i][j]['result'])

    column_results_df = pd.DataFrame({'columns' : columns, 'checks' : checks, 'results' : results})

    return column_results_df

# run the functions
data, dq_json= load_data(sb_selectbox)

dq_metrics_df, total_score = compute_dq_metrics(data,dq_json)

table_results_df = compute_table_checks_results(dq_json)

column_results_df = compute_column_checks_results(dq_json)

basic_metrics_df= compute_basic_metrics(data)

layout_plot, layout_dist = layout()

# Heading
st.title('Data Quality')
st.subheader('metrics')

###### ROW 1 #######
accuracy, relevancy, completeness, timeliness, consistency, overall_score,stats = st.columns([1,1,1,1,1,1,1])

with accuracy:
    st.write('Accuracy')
    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('accuracy')], names = 'metric', values = 'percentage', color = 'metric', \
        hole = 0.5,color_discrete_map={"accuracy" : '#19AA6E',"accuracy_l" : '#0E1117'})
    fig.update_traces(textinfo='none')
    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "accuracy"]["percentage"].iloc[0])
    fig.update_layout(layout_plot)
    st.plotly_chart(fig, use_container_width=True)

with relevancy:
    st.write('relevancy')
    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('relevancy')], names = 'metric', values = 'percentage', color = 'metric', \
        hole = 0.5,color_discrete_map={"relevancy" : '#19AA6E',"relevancy_l" : '#0E1117'})
    fig.update_traces(textinfo='none')
    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "relevancy"]["percentage"].iloc[0])
    fig.update_layout(layout_plot)
    st.plotly_chart(fig, use_container_width=True)

with completeness:
    st.write('Completeness')
    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('completeness')], names = 'metric', values = 'percentage', color = 'metric', \
        hole = 0.5,color_discrete_map={"completeness" : '#19AA6E',"completeness_l" : '#0E1117'})
    fig.update_traces(textinfo='none')
    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "completeness"]["percentage"].iloc[0])
    fig.update_layout(layout_plot)
    st.plotly_chart(fig, use_container_width=True)

with timeliness:
    st.write('Timeliness')
    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('timeliness')], names = 'metric', values = 'percentage', color = 'metric', \
        hole = 0.5,color_discrete_map={"timeliness" : '#19AA6E',"timeliness_l" : '#0E1117'})
    fig.update_traces(textinfo='none')
    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "timeliness"]["percentage"].iloc[0])
    fig.update_layout(layout_plot)
    st.plotly_chart(fig, use_container_width=True)

with consistency:
    st.write('Consistency')
    fig = px.pie(dq_metrics_df[dq_metrics_df['metric'].str.contains('consistency')], names = 'metric', values = 'percentage', color = 'metric', \
        hole = 0.5,color_discrete_map={"consistency" : '#19AA6E',"consistency_l" : '#0E1117'})
    fig.update_traces(textinfo='none')
    layout_plot['annotations'][0]['text'] = str(dq_metrics_df[dq_metrics_df['metric'] == "consistency"]["percentage"].iloc[0])
    fig.update_layout(layout_plot)
    st.plotly_chart(fig, use_container_width=True)

with overall_score:
    st.metric(label="DQ score", value=f"{total_score}", delta = f"-{100 - total_score}")

with stats:
    st.metric(label="Total records", value=f"{dq_json['total_records_actual']}", delta = f"-{dq_json['total_records_dropped']} rows dropped")
    st.metric(label="Total checks", value=f"{dq_json['checks_total']}", delta = f"-{dq_json['checks_failed']} checks failed")

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

###### ROW 2 #######
# heading
table_checks_heading, column_checks_heading = st.columns([1,1])
with table_checks_heading:
    st.subheader('Table checks')

with column_checks_heading:
    st.subheader('Column checks')


######## ROW 3 #######
table_checks_radio, table_checks_stats, column_checks_radio, column_checks_stats = st.columns([1,1,1,1])

#TABLE CHECKS
with table_checks_radio:
    table_checks_radio = st.radio(label = 'Checks status', options = ('Pass', 'Fail'), key = 'table_checks_radio')

with table_checks_stats:
    st.metric(label="Total checks", value=f"{dq_json['checks_table_level']['checks_table_total']}", delta = f"-{dq_json['checks_table_level']['checks_table_failed']} checks failed")


# COLUMN CHECKS
# radio button for checks passed and failed
with column_checks_radio:
    column_checks_radio = st.radio(label = 'Checks status', options = ('Pass', 'Fail'), key = 'column_checks_radio')

# overall checks passed and failed
with column_checks_stats:
    st.metric(label="Total checks", value=f"{dq_json['checks_column_level']['checks_column_total']}", delta = f"-{dq_json['checks_column_level']['checks_column_failed']} checks failed")


###### ROW 4 #######
table_checks_select, column_checks_col_select, column_checks_select = st.columns([2,1,1])

# table checks select
with table_checks_select:
    if table_checks_radio == 'Pass':
        table_checks_options = tuple(table_results_df[table_results_df['results'] == True]['checks'])
    else:
        table_checks_options = tuple(table_results_df[table_results_df['results'] == False]['checks'])
    
    table_checks_selectbox = st.selectbox(
    'Select a check',
    table_checks_options,
    key = 'table_checks_selectbox'
    )

# column select a column
with column_checks_col_select:
    if column_checks_radio == 'Pass':
        column_checks_col_options = tuple(column_results_df[column_results_df['results'] == True]['columns'].unique())
    else:
        column_checks_col_options = tuple(column_results_df[column_results_df['results'] == False]['columns'].unique())
    
    column_checks_col_selectbox = st.selectbox(
    'Select a column',
    column_checks_col_options,
    key = 'column_checks_col_selectbox'
    )

# column checks select 
with column_checks_select:
    if column_checks_radio == 'Pass':
        column_checks_options = tuple(column_results_df[(column_results_df['columns'] == column_checks_col_selectbox) & (column_results_df['results'] == True)]['checks'])
    else:
        column_checks_options = tuple(column_results_df[(column_results_df['columns'] == column_checks_col_selectbox) & (column_results_df['results'] == False)]['checks'])
    column_checks_selectbox = st.selectbox(
    'Select a check',
    column_checks_options,
    key = 'column_checks_selectbox'
    )

###### ROW 5 #######

table_checks_json, column_checks_json = st.columns([1,1])

# table checks json
with table_checks_json:
    try:
        table_json = dq_json['checks_table_level'][table_checks_selectbox]
        st.json(table_json)
    except KeyError:
        st.json({'checks' : 'None'})

# columns checks json
with column_checks_json:
    try:
        columns_json = dq_json['checks_column_level'][column_checks_col_selectbox][column_checks_selectbox]
        st.json(columns_json)
    except KeyError:
        st.json({'checks' : 'None'})
        

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

###### ROW 6 #######
# barplot and distribution 
st.subheader('Viz')
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

###### ROW 7 #######
# signal 
st.write('Signal')
st.line_chart(data['crssi_dbm'])
# raw dataset
st.write('View raw dataset')
if st.checkbox('Show dataframe'):
    data

