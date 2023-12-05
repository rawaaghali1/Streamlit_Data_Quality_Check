import streamlit as st 

from PIL import Image

# set the page configuration
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="Interactive Data Quality Check",  # String or None. Strings get appended with "• Streamlit". 
    page_icon=None,  # String, anything supported by st.image, or None.
)

image_logo = Image.open('assets/danone_ds_logo.png')
col1, col2 = st.columns([8, 1])
with col1:
	st.title('Interactive Data Quality Check')
with col2:
	st.write('Powered by:')
	st.image(image_logo, width=90, output_format='PNG')
    
# copies 
home_title = "Interactive Data Quality Check"
home_introduction = """
Welcome to Interactive Data Quality Check! Data Quality Check, or DQC, refers to the process of validating and ensuring the accuracy, consistency, and reliability of data in your data. This process is crucial in data management and analytics because high-quality data is essential for making informed decisions and accurate analyses. 
\n
To help you ensure the quality of your data, we developed this interactive DQC Tool, a web app where you can upload your raw data, define your rules (or expectations) of what certain column values in your data should be like, and generate a dashboard to review your data quality and even explore your raw data.
"""

home_howitworks = """
Your journey towards impeccable data quality begins here! Interactive Data Quality Check streamlines the process of validating and enhancing the quality of your datasets. Here’s how you can harness the power of our tool:  

##### 1. Generate Your Data Quality Dashboard

- **Upload Your Data**: Begin by uploading your dataset in a supported format, such as CSV or Excel. Please make sure that your file only contains one sheet and one header row on top.
- **Upload Your JSON File for DQC Rules**: Upload a JSON file that outlines your specific rules for data quality check. This file should contain criteria such as data types, desired ranges, uniqueness constraints, and more. If you don't have such a file, you can generate one in the "Create Your DQC Rules" step.
- **Receive Insights**: Once you submit your data and your JSON file for DQC rules, you'll receive a comprehensive dashboard visualizing the quality of your data, highlighting areas that meet your criteria and those that need attention.
- **Download your Excel Report**: Once you submit your data and your JSON file for DQC rules, you can also choose to download an Excel report that includes the index of column values that don't match your rules.

##### 2. Create Your DQC Rules

- **Upload Your Data**: Upload your CSV or Excel dataset, and our tool will read the column names in your data for the next steps.
- **Define Your DQC Rules**: Our intuitive interface will guide you through setting up your DQC rules for any column. This could include specifications like acceptable value ranges, required formats, or uniqueness.
- **Generate and Download Your JSON File**: After setting your rules, hit 'Download the list as a json file'. Instantly, you'll have a tailor-made JSON file that you can use with our app in "DQC Dashboard".
"""

st.markdown("""\n""")
st.markdown("#### About")
st.write(home_introduction)

st.markdown("""\n""")

st.markdown("#### How It Works")
image_flowchart = Image.open('assets/flowchart.png')
st.image(image_flowchart, output_format='PNG')
st.markdown(home_howitworks)

st.markdown("""\n""")

st.info('''
If you encouter any technical issues or need any help, please contact [Zhengxiao YING](mailto:zhengxiao.ying@external.danone.com) or [Mahabub ALAM](mailto:mahabub.alam@danone.com).''', icon="ℹ️")

