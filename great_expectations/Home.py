import streamlit as st 
import app_utils as vutil 
import app_component as ac 
import app_user as uv 


st.set_page_config(
    page_title="Home",
)

ac.render_cta()

# copies 
home_title = "DQC Tool"
home_introduction = "Welcome to DQC Tool! DQC, or Data Quality Check, refers to the process of validating and ensuring the accuracy, consistency, and reliability of data in your data. This process is crucial in data management and analytics because high-quality data is essential for making informed decisions and accurate analyses. To help you ensure the quality of your data, we developed this DQC Tool, a web app where you can upload your raw data, define your expectations of what certain column values in your data should be like, and generate a dashboard to review your data quality and even explore your raw data."
home_howitworks = ""

#st.title(home_title)
st.markdown(f"""# {home_title}""",unsafe_allow_html=True)

st.markdown("""\n""")
st.markdown("#### About")
st.write(home_introduction)

st.markdown("#### How It Works")
st.write(home_privacy)

st.markdown("""\n""")
st.markdown("""\n""")

st.markdown("#### Get Started")

vu = uv.app_user()
if 'user' not in st.session_state or st.session_state.user['id'] is None:
    vu.view_get_info()
else:
    vu.view_success_confirmation()
    st.write("\n")
    col1, col2 = st.columns(2)
    with col1: 
        if st.button("Hang out with AI Assistants in the Lounge"):
            vutil.switch_page('lounge')
    with col2: 
        if st.button("Create your own AI Assistants in the Lab"):
            vutil.switch_page('lab')
