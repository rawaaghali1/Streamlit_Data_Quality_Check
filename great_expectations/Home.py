import streamlit as st 

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="DQC Tool",  # String or None. Strings get appended with "• Streamlit". 
    page_icon=None,  # String, anything supported by st.image, or None.
)

# copies 
home_title = "DQC Tool"
home_introduction = """
Welcome to DQC Tool! DQC, or Data Quality Check, refers to the process of validating and ensuring the accuracy, consistency, and reliability of data in your data. This process is crucial in data management and analytics because high-quality data is essential for making informed decisions and accurate analyses. 
\n
To help you ensure the quality of your data, we developed this DQC Tool, a web app where you can upload your raw data, define your expectations of what certain column values in your data should be like, and generate a dashboard to review your data quality and even explore your raw data.
"""

home_howitworks = """
Your journey towards impeccable data quality begins here! Data Quality Check Web App streamlines the process of validating and enhancing the quality of your datasets. Here’s how you can harness the power of our tool:  

##### 1. Create Your Expectation JSON File

- **Upload Your Data and Define Expectations**: If you don't know about JSON files, no worries! Simply upload your CSV or Excel dataset, and our intuitive interface will guide you through setting up your expectations for each column. This could include specifications like acceptable value ranges, required formats, or uniqueness.
- **Generate and Download Your JSON File**: After setting your expectations, hit 'Download the list as a json file'. Instantly, you'll have a tailor-made JSON file that you can use with our app in the next step.

##### 2. Generate Your Data Quality Dashboard

- **Upload Your Data**: Begin by uploading your dataset in a supported format, such as CSV or Excel.
- **Upload Your Expectation JSON File**: Provide us with a JSON file (generated in the previous step) that outlines your specific expectations for data quality. This file should contain criteria such as data types, desired ranges, uniqueness constraints, and more.
- **Receive Insights**: Once you submit your data and expectation file, you'll receive a comprehensive dashboard visualizing the quality of your data, highlighting areas that meet your criteria and those that need attention.
- **Download your Excel Report**: Once you submit your data and expectation file, you can also choose to download an Excel report that includes the index of column values that don't match your expectations.

"""

#st.title(home_title)
st.markdown(f"""# {home_title}""",unsafe_allow_html=True)

st.markdown("""\n""")
st.markdown("#### About")
st.write(home_introduction)

st.markdown("""\n""")

st.markdown("#### How It Works")
st.markdown(home_howitworks)

st.markdown("""\n""")

st.info('''
If you encouter any technical issues or need any help, please contact [Zhengxiao YING](mailto:zhengxiao.ying@external.danone.com) or [Mahabub ALAM](mailto:mahabub.alam@danone.com).''', icon="ℹ️")

