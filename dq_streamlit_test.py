import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

df = pd.read_excel("data/test_raw_file.xlsx")
if df.shape[0]>10000:
  df = df.sample(10000)
pr = ProfileReport(df, title="Report")
st_profile_report(pr)
