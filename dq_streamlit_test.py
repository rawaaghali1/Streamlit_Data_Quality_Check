import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("data/test_raw_file_2.csv")
pr = ProfileReport(df, title="Report")
st_profile_report(pr)
