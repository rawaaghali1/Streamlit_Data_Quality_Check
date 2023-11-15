import pandas as pd
import numpy as np
import streamlit as st
import json
import datetime
import os
import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.yaml_handler import YAMLHandler
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
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

yaml = YAMLHandler()
uploaded_file = st.file_uploader("Choose a file")
df = pd.read_csv(uploaded_file)
file_path = '/great_expectations/sl_batch2v1_rules_3.json'
with open(file_path, 'r') as file:
    config = json.load(file)

# config properties
data_path = config['data_path']
context_root_dir = config['context_root_dir']
expectation_suite_name = config['expectation_suite_name']
project_name = config['project_name']
data_path=config['project_name']
datasource_name = config['datasource_name']
checkpoint_name = config['checkpoint_name']
data_asset_name = config['data_asset_name']
root_directory = config["root_directory"]
run_name_template=config["run_name_template"]
rules = config['rules']
if 'sheet_name' in config:
    sheet_name = config['sheet_name']
else:
    sheet_name = None


data_context_config = DataContextConfig(
    store_backend_defaults=FilesystemStoreBackendDefaults(
        root_directory=root_directory
    ),
)
context = get_context(project_config=data_context_config)

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
# Instantiate the Data_quality_check class
dqc = Data_quality_check()
merged_df_new = pd.DataFrame()
expectation_suite = context.get_expectation_suite(expectation_suite_name)

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

st.dataframe()
