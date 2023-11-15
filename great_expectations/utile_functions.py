import pandas as pd
from great_expectations.core.expectation_validation_result import (
    ExpectationValidationResult,
)


def create_df_from_validation_result(validation_result: ExpectationValidationResult):
    result_dict = validation_result.result
    element_count = result_dict.get("element_count")
    unexpected_count = result_dict.get("unexpected_count")
    unexpected_percent = result_dict.get("unexpected_percent")
    partial_unexpected_list = result_dict.get("partial_unexpected_list")
    partial_unexpected_index_list = result_dict.get("partial_unexpected_index_list")
    partial_unexpected_counts = result_dict.get("partial_unexpected_counts")
    unexpected_list = result_dict.get("unexpected_list")
    unexpected_index_list = result_dict.get("unexpected_index_list")
    unexpected_index_query = result_dict.get("unexpected_index_query")
    raised_exception = validation_result.exception_info.get("raised_exception")
    exception_traceback = validation_result.exception_info.get("exception_traceback")
    exception_message = validation_result.exception_info.get("exception_message")
    meta = validation_result.meta
    success = validation_result.success

    data = {
        "element_count": [element_count],
        "unexpected_count": [unexpected_count],
        "unexpected_percent": [unexpected_percent],
        "partial_unexpected_list": [partial_unexpected_list],
        "partial_unexpected_index_list": [partial_unexpected_index_list],
        "partial_unexpected_counts": [partial_unexpected_counts],
        "unexpected_list": [unexpected_list],
        "unexpected_index_list": [unexpected_index_list],
        "unexpected_index_query": [unexpected_index_query],
        "raised_exception": [raised_exception],
        "exception_traceback": [exception_traceback],
        "exception_message": [exception_message],
        "meta": [meta],
        "success": [success],
    }

    return pd.DataFrame(data)


#%%

def convert_dict_to_dataframe(d: dict) -> pd.DataFrame:
    # Initialize empty list to hold the dictionaries for each column
    column_dicts = []

    # Loop through each column and extract its dictionary
    for column, column_data in d.items():
        # Extract the result dictionary for the column
        result_dict = column_data["result"]

        # Add the column name to the dictionary for this column's data
        result_dict["column"] = column

        # Check if the 'notes' key is present in the 'meta' dictionary
        if "notes" in column_data["meta"]:
            notes = column_data["meta"]["notes"]["content"]
        else:
            notes = []

        result_dict["notes"] = notes
        result_dict.update(column_data["exception_info"])
        result_dict["success"] = column_data["success"]

        # Reorder the keys of the dictionary so that 'column' is the first key
        result_dict = {
            "column": result_dict["column"],
            **{k: v for k, v in result_dict.items() if k != "column"},
        }

        # Append the dictionary to the list
        column_dicts.append(result_dict)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(column_dicts)

    return df


#%%

def convert_dict_to_dataframe_q(d: dict, i: str) -> pd.DataFrame:
    # Initialize empty list to hold the dictionaries for each column
    column_dicts = []

    # Loop through each column and extract its dictionary
    for idx, (column, column_data) in enumerate(d.items()):
        # Extract the result dictionary for the column
        result_dict = column_data['result']
        
        # Add the column name to the dictionary for this column's data
        result_dict['column'] = column
        
        # Check if the 'notes' key is present in the 'meta' dictionary
        if 'notes' in column_data['meta']:
            notes = column_data['meta']['notes']['content']
        else:
            notes = []
        
        result_dict['notes'] = notes
        result_dict.update(column_data['exception_info'])
        result_dict['success'] = column_data['success']
        
        # Reorder the keys of the dictionary so that 'column' is the first key
        result_dict = {'column': result_dict['column'], **{k: v for k, v in result_dict.items() if k != 'column'}}
        
        # Append the dictionary to the list
        column_dicts.append(result_dict)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(column_dicts)

    # Add the "Quality N°" column
    df.insert(0, 'Quality N°', "Q_" + i)

    return df

