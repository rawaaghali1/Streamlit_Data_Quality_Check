import pandas as pd
import numpy as np
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.validator.validator import Validator
import great_expectations as ge
from great_expectations.dataset import PandasDataset
from great_expectations.core.expectation_configuration import ExpectationConfiguration


#%%

class Data_quality_check:

    def test_column_values_to_be_positive_or_zero(
        self, df, column: str
    ):
        """
        Test if values in the specified column are either positive or zero.
        """
        results = {}
        
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the dataframe.")
            return results
        
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=my_expectation_suite_name,
        )
        df[column] = df[column].replace("", pd.NA)
        
        # Convert valid values in the column to float
        df[column] = pd.to_numeric(df[column], errors="coerce")
        result = validator.expect_column_values_to_be_greater_than_or_equal_to(
            column=column,
            value=0,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must be either positive or zero.",
                        "",
                        "OK",
                    ],
                }
            },
        )
        results[column] = {
            "expectation_name": "expect_column_values_to_be_positive_or_zero",
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results
    
    def columns_to_exist(self, df: pd.DataFrame, columns_to_validate: list):
        """
        Validates the existence of specified columns.

        :param df: The DataFrame to be validated.
        :param columns_to_validate: A list of column names to be validated for existence.
        :return: A dictionary containing validation results for each column.
        """
        # Convert the DataFrame to a Great Expectations DataContext
        validator = ge.from_pandas(df)
        results = {}  # Initialize a dictionary to store validation results

        # Iterate over each column to validate its existence
        for column in columns_to_validate:
            result = validator.expect_column_to_exist(
                column=column,
                result_format="COMPLETE",
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": ["Existence", f"{column} must exist.", "", "OK"],
                    }
                },
            )

            # Store validation results for the current column
            results[column] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        return results  # Return the dictionary of validation results

    def expect_column_values_to_be_unique(self, df, column):
        """
        Expects the values in the specified column to be unique (without duplicates).

        :param df: The DataFrame to be validated.
        :param column: The column name for which uniqueness is expected.
        :return: A dictionary containing validation results for the uniqueness expectation.
        """
        dataset = ge.dataset.PandasDataset(df)
        results = {}  # Initialize a dictionary to store validation results

        if column not in df.columns:
            return None  # Early return if the column is not present in the DataFrame

        result = dataset.expect_column_values_to_be_unique(
            column=column,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Uniqueness",
                        f"The values in {column} must be unique.",
                        "",
                        "OK",
                    ],
                }
            },
        )

        # Store validation results for the uniqueness expectation
        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results  # Return the dictionary of validation results

    def expect_column_values_to_be_in_list(self,df: pd.DataFrame, column: str, expected_values: list) -> pd.DataFrame:
        """
        Expects the values in the specified column to be within a list of expected values.

        :param df: The DataFrame to be validated.
        :param column: The column name to be validated.
        :param expected_values: A list of expected values.
        :return: A dictionary containing validation results for the expected values.
        """

        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: str(x).upper() if x is not None else None)

        results = {}
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the dataframe.")
            return results
        
        expected_values_upper = [str(value).upper() for value in expected_values]
        
        df_upper = df[column].str.upper()
        
        result = ge.dataset.PandasDataset(df_upper).expect_column_values_to_be_in_set(
            column=column,
            value_set=expected_values_upper,
            result_format="COMPLETE",
            include_config=False,
            catch_exceptions=False,
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must be equal to one of the expected values: {expected_values}.",
                        "",
                        "OK"
                    ]
                }
            }
        )
        
        if result is not None:
            results[column] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success
            }
            # Add expectation to the expectation suite
            expectation_config = ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": column, "value_set": expected_values_upper},
                meta={"notes": f"{column} must be equal to one of the expected values: {expected_values}."}
            )

        return results


    def columns_to_exist_if_condition_column(
        self,
        df: pd.DataFrame,
        columns_with_conditions: list,
        condition_column: str,
        condition_value: str,
    ):
        """
        Validates the existence of specified columns based on a condition column's value.

        :param df: The DataFrame to be validated.
        :param columns_with_conditions: A list of column names to be validated for existence.
        :param condition_column: The column name for the condition.
        :param condition_value: The value of the condition column to trigger validation.
        :return: A dictionary containing validation results for each column.
        """
        validator = ge.from_pandas(df)
        results = {}  # Initialize a dictionary to store validation results

        for column in columns_with_conditions:
            if column not in df.columns:
                continue  # Skip this iteration if the column is not present in the DataFrame

            # Check if the value of condition_column equals condition_value
            if df[condition_column].equals(condition_value):
                result = validator.expect_column_to_exist(
                    column=column,
                    result_format="COMPLETE",
                    meta={
                        "notes": {
                            "format": "markdown",
                            "content": [
                                "Existence",
                                f"{column} must exist if {condition_column} equals to {condition_value}.",
                                "",
                                "OK",
                            ],
                        }
                    },
                )

                results[column] = {
                    "result": result.result,
                    "exception_info": result.exception_info,
                    "meta": result.meta,
                    "success": result.success,
                }

        return results  # Return the dictionary of validation results

    def column_values_to_not_be_null(self, df: pd.DataFrame, columns_to_validate: list):
        """
        Validates that the values in specified columns are not null.

        :param df: The DataFrame to be validated.
        :param columns_to_validate: A list of column names to be validated for non-null values.
        :return: A dictionary containing validation results for each column.
        """
        validator = ge.from_pandas(df)
        results = {}  # Initialize a dictionary to store validation results

        for column in columns_to_validate:
            if column not in df.columns:
                continue  # Skip this iteration if the column is not present in the DataFrame

            result = validator.expect_column_values_to_not_be_null(
                column=column,
                result_format="COMPLETE",
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": [
                            "Completeness",
                            f"{column} must not be null.",
                            "",
                            "OK",
                        ],
                    }
                },
            )

            results[column] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        return results  # Return the dictionary of validation results

    def column_values_to_not_be_null_if_column_a(
        self, df: pd.DataFrame, column_a: str, column_to_validate: str, some_value: str
    ):
        """
        Validates that the values in a specified column are not null if a condition based on another column is met.

        :param df: The DataFrame to be validated.
        :param column_a: The column name to be used for the condition.
        :param column_to_validate: The column to be validated for non-null values.
        :param some_value: The value of the condition column to trigger the validation.
        :return: A dictionary containing validation results for the non-null values expectation.
        """
        validator = ge.from_pandas(df)
        results = {}  # Initialize a dictionary to store validation results

        if column_to_validate not in df.columns:
            print(f"Column '{column_to_validate}' does not exist in the dataframe.")
            return results

        # Check the condition based on column_a
        if df[column_a].equals(some_value):
            result = validator.expect_column_values_to_not_be_null(
                column=column_to_validate,
                result_format="COMPLETE",
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": [
                            "Completeness",
                            f"{column_to_validate} must not be null if {column_a} equals {some_value}.",
                            "",
                            "OK",
                        ],
                    }
                },
            )

            results[column_to_validate] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        return results

    def test_column_values_to_be_of_type_string(
        self, df: pd.DataFrame, columns_to_validate: list
    ):
        """
        Validates that the values in specified columns are of type string.

        :param df: The DataFrame to be validated.
        :param columns_to_validate: A list of column names to be validated for string type.
        :return: A dictionary containing validation results for the data type expectation.
        """
        validator = ge.from_pandas(df)
        results = {}  # Initialize a dictionary to store validation results

        for column in columns_to_validate:
            if column not in df.columns:
                continue

            try:
                result = validator.expect_column_values_to_be_of_type(
                    column=column,
                    type_="str",
                    result_format="COMPLETE",
                    meta={
                        "notes": {
                            "format": "markdown",
                            "content": [
                                "Data Type",
                                f"{column} must be of string type.",
                                "",
                                "OK",
                            ],
                        }
                    },
                )

                results[column] = {
                    "result": result.result,
                    "exception_info": result.exception_info,
                    "meta": result.meta,
                    "success": result.success,
                }
            except ge.exceptions.UnexpectedColumn:
                # If the column doesn't exist, skip the validation and continue to the next column
                continue

        return results

    def test_column_values_to_be_of_type_numeric(self, df, columns_to_validate):
        """
        Validates that the values in specified columns are of type numeric.

        :param df: The DataFrame to be validated.
        :param columns_to_validate: A list of column names to be validated for numeric type.
        :return: A dictionary containing validation results for the data type expectation.
        """
        validator = ge.from_pandas(df)
        results = {}  

        for column in columns_to_validate:
            if column not in df.columns:
                return results

        for column in columns_to_validate:
            try:
                result = validator.expect_column_values_to_match_regex(
                    column=column,
                    regex=r"^\d+(\.\d+)?$",
                    result_format="COMPLETE",
                    meta={
                        "notes": {
                            "format": "markdown",
                            "content": [
                                "Data Type",
                                f"{column} must be a numeric value (integer or float).",
                                "",
                                "OK",
                            ],
                        }
                    },
                )

                results[column] = {
                    "result": result.result,
                    "exception_info": result.exception_info,
                    "meta": result.meta,
                    "success": result.success,
                }

            except ge.exceptions.UnexpectedColumn:
                continue

        return results

    def expect_column_values_to_match_regex(self, df, column, regex_str):
        """
        Expect values in the specified column to match the given regex.
        """
        validator = ge.from_pandas(df)
        results = {}

        if column not in df.columns:
            results

        result = validator.expect_column_values_to_match_regex(
            column=column,
            regex=regex_str,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must match {regex_str}.",
                        "",
                        "OK",
                    ],
                }
            },
        )

        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def expect_column_values_to_be_in_set(
        self, df: pd.DataFrame, column_values_dict: dict
    ) -> pd.DataFrame:
        """
        Expect values in the specified columns to be equal to one of the expected values.
        the keys are the column names and the expected_values are the values
        """
        results = {}
        df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        if column not in df.columns:
            results
        for column, expected_values in column_values_dict.items():
            
            result = ge.dataset.PandasDataset(df).expect_column_values_to_be_in_set(
                column=column,
                value_set=expected_values,
                result_format="COMPLETE",
                include_config=False,
                catch_exceptions=False,
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": [
                            "Validity",
                            f"{column} must be equal to one of the expected values: {expected_values}.",
                            "",
                            "OK",
                        ],
                    }
                },
            )

            results[column] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        return results

    def test_column_values_to_be_of_type_datetime(self, df, column, date_format):
        """
        Test if values in the specified column match the specified datetime format.
        """
        dataset = ge.dataset.PandasDataset(df)
        results = {}

        if column not in df.columns:
            pass

        result = dataset.expect_column_values_to_match_strftime_format(
            column=column,
            date_format=date_format,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must match the specified datetime format {date_format}.",
                        "",
                        "OK",
                    ],
                }
            },
        )

        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def validate_column_values(self, df: pd.DataFrame, column_values_dict: dict):
        """
        Validate that values in specified columns match the expected values.
        """
        validator = ge.from_pandas(df)
        results = {}

        if any(key not in df.columns for key in column_values_dict):
            print(
                "One or more columns from column_values_dict do not exist in the dataframe."
            )
            return results
        for column, expected_values in column_values_dict.items():
            result = validator.expect_column_values_to_be_in_set(
                column=column,
                value_set=expected_values,
                result_format="COMPLETE",
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": [
                            "Validity",
                            f"{column} must be equal to one of the expected values: {expected_values}.",
                            "",
                            "OK",
                        ],
                    }
                },
            )

            results[column] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        return results
    
    def test_column_pair_value_and_existence(
        self, df: pd.DataFrame, column_A: str, value_A: str, column_B: str
    ):
        """
        Test if values in column A exist and match the expected value, and if column B exists.
        """
        results = {}
        
        if (column_A not in df.columns) or (column_B not in df.columns):
            print(f"Column '{column_A}' or '{column_B}' does not exist in the dataframe.")
            return results

        dataset = PandasDataset(df)
        
        # Check if column A exists and its values are in a set
        result_A_existence = dataset.expect_column_to_exist(column_A)
        if not result_A_existence.success:
            results[column_A] = {
                "result": result_A_existence.result,
                "exception_info": result_A_existence.exception_info,
                "meta": result_A_existence.meta,
                "success": result_A_existence.success,
            }
            return results

        result_A_values = dataset.expect_column_values_to_be_in_set(column_A, [value_A])
        if not result_A_values.success:
            results[column_A] = {
                "result": result_A_values.result,
                "exception_info": result_A_values.exception_info,
                "meta": result_A_values.meta,
                "success": result_A_values.success,
            }
        
        # Check if column B exists
        result_B_existence = dataset.expect_column_to_exist(column_B)
        results[column_B] = {
            "result": result_B_existence.result,
            "exception_info": result_B_existence.exception_info,
            "meta": result_B_existence.meta,
            "success": result_B_existence.success,
        }

        return results


    def column_values_to_be_null(self, df: pd.DataFrame, columns_to_validate: list):
        """
        Validate that values in specified columns are null.
        """
        dataset = PandasDataset(df)
        results = {}

        for column in columns_to_validate:
            # Check if the column exists in the DataFrame
            if column not in dataset.columns:
                print(f"Warning: Column '{column}' does not exist in the DataFrame.")
                continue

            # Use the dataset object to define the expectation
            result = dataset.expect_column_values_to_be_null(
                column=column,
                result_format="COMPLETE",
                meta={
                    "notes": {
                        "format": "markdown",
                        "content": [
                            "Completeness",
                            f"{column} must be null.",
                            "",
                            "OK",
                        ],
                    }
                },
            )
            results[column] = {
                "expectation_name": "expect_column_values_to_be_null",
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }
        return results



    def test_column_values_to_be_between(
        self, df, column: str, min_val: float, max_val: float
    ):
        """
        Test if values in the specified column are between the specified min and max values.
        """
        results = {}
        if column not in df.columns:
            print(f"Column '{column} does not exist in the dataframe.")
            return results

        """validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=my_expectation_suite_name,
        )"""
        df[column] = df[column].replace("", pd.NA)

        # Convert valid values in the column to float
        df[column] = pd.to_numeric(df[column], errors="coerce")
        result = PandasDataset(df).expect_column_values_to_be_between(
            column=column,
            allow_cross_type_comparisons=True,
            min_value=min_val,
            max_value=max_val,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must be between {min_val} and {max_val}.",
                        "",
                        "OK",
                    ],
                }
            },
        )
        results[column] = {
            "expectation_name": "expect_column_values_to_be_null",
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def test_column_values_to_be_positive_or_zero(self, df: pd.DataFrame, column: str):
        """
        Test if values in the specified column are positive or zero.
        """
        results = {}

        dataset = PandasDataset(df)
        if column not in df.columns:
            print(f"Column '{column} does not exist in the dataframe.")
            return results

        # Convert column values to numeric type
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

        result = dataset.expect_column_values_to_be_between(
            column=column,
            min_value=0,
            strict_max=False,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must be a positive number or zero.",
                        "",
                        "OK",
                    ],
                }
            },
        )
        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def test_column_to_be_empty(self, df: pd.DataFrame, column: str):
        results = {}

        dataset = PandasDataset(df)

        result = dataset.expect_column_values_to_be_null(
            column=column,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": ["Completeness ", f"{column} must be empty.", "", "OK"],
                }
            },
        )

        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def column_pair_values_to_be_greater_than(
        self, df, column_A, column_B, or_equal=True
    ):
        """
        Test if values in column A are greater than or equal to values in column B.
        """
        dataset = PandasDataset(df)
        results = {}

        # Check if column_A and column_B exist in the DataFrame
        if column_A not in df.columns:
            print(
                f"Column '{column_A}' does not exist in the DataFrame. Skipping quality check."
            )
            return results
        if column_B not in df.columns:
            print(
                f"Column '{column_B}' does not exist in the DataFrame. Skipping quality check."
            )
            return results

        result = dataset.expect_column_pair_values_a_to_be_greater_than_b(
            column_A=column_A,
            column_B=column_B,
            or_equal=or_equal,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Consistency ",
                        f"{column_A} should be greater or equal {or_equal} than {column_B}.",
                        "",
                        "OK",
                    ],
                }
            },
        )
        column = f"{column_A} > {column_B}"
        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def test_column_difference_to_be_between(
        self,
        df: pd.DataFrame,
        column_A: str,
        column_B: str,
        min_diff: float,
        max_diff: float,
    ):
        """
        Test if the difference between values in column A and column B is within a specified range.
        """
        results = {}

        # Check if column_A and column_B exist in the DataFrame
        if column_A not in df.columns:
            print(
                f"Column '{column_A}' does not exist in the DataFrame. Skipping quality check."
            )
            return results
        if column_B not in df.columns:
            print(
                f"Column '{column_B}' does not exist in the DataFrame. Skipping quality check."
            )
            return results

        # Compute the difference between column_A and column_B
        df["difference"] = df[column_A] - df[column_B]

        dataset = ge.from_pandas(df)

        result = dataset.expect_column_values_to_be_between(
            column="difference",
            min_value=min_diff,
            max_value=max_diff,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"Difference between '{column_A}' and '{column_B}' must be between {min_diff} and {max_diff}.",
                        "",
                        "OK",
                    ],
                }
            },
        )

        column = f"{column_A} - {column_B}"
        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }
        return results

    def test_column_values_to_be_of_type_datetime(self, df, column):
        """
        Test if values in the specified column are in the 'YYYY-MM-DD' datetime format.
        """
        results = {}
        validator = ge.from_pandas(df)
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the dataframe.")
            return results
        result = validator.expect_column_values_to_match_regex(
            column=column,
            regex=r"\d{4}-\d{2}-\d{2}",
            mostly=1.0,
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"{column} must be in the format 'YYYY-MM-DD'.",
                        "",
                        "OK",
                    ],
                }
            },
        )

        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }
        return results

    

    def test_check_column_existence_if_a(
        self, df: pd.DataFrame, column_a: str, list_of_values: list, column_b: str
    ):
        """
        Checks if column_a is in the list_of_values and then checks if column_b exists.
        """
        results = {}

        # Check if both column_a and column_b exist in the DataFrame
        if (column_a not in df.columns) or (column_b not in df.columns):
            print(
                f"Column '{column_a}' or '{column_b}' does not exist in the dataframe."
            )
            return results

        # Convert relevant columns and values to uppercase for case-insensitive comparison
        df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        list_of_values_upper = [
            value.upper() if isinstance(value, str) else value
            for value in list_of_values
        ]

        # Filter rows where column_a equals to one of the values in list_of_values
        selected_df = df[df[column_a].isin(list_of_values_upper)]

        # Check if column_b exists in the filtered dataframe
        result = ge.dataset.PandasDataset(selected_df).expect_column_to_exist(column_b)

        # Add additional notes or metadata to the result
        result.meta["notes"] = {
            "format": "markdown",
            "content": [
                "Completeness",
                f"Expected Value: {column_a} is in {list_of_values_upper} then {column_b} should exist",
                "",
                "OK",
            ],
        }

        # Save the expectation suite after adding the expectation

        results[column_a] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    def test_check_value_in_list(
        self, df, column_a, list_of_values, column_b, set_of_values
    ):
        """
        Checks if column_a is in the list_of_values and then checks if column_b values are in set_of_values.
        """
        results = {}

        # Check if both column_a and column_b are present in the DataFrame
        if (column_a not in df.columns) or (column_b not in df.columns):
            print(
                f"Column '{column_a}' or '{column_b}' does not exist in the dataframe."
            )
            return results

        # Convert relevant columns and values to uppercase for case-insensitive comparison
        df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        list_of_values_upper = [
            value.upper() if isinstance(value, str) else value
            for value in list_of_values
        ]
        set_of_values_upper = [
            value.upper() if isinstance(value, str) else value
            for value in set_of_values
        ]

        try:
            # Filter rows where column_a contains values from the specified list
            selected_df = df[df[column_a].isin(list_of_values_upper)]

            # Check if values in column_b are in the specified set of values
            result = ge.dataset.PandasDataset(
                selected_df
            ).expect_column_values_to_be_in_set(column_b, set_of_values_upper)

            # Add additional notes or metadata to the result
            result.meta["notes"] = {
                "format": "markdown",
                "content": [
                    "Value Check",
                    f"If {column_a} is in {list_of_values_upper}",
                    f"then values in {column_b} should be in {set_of_values_upper}",
                    "",
                    "OK",
                ],
            }

            results[column_a] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        except KeyError as e:
            print(f"Warning: An error occurred while performing the validation: {e}")

        return results

    def test_check_value_in_list_2_columns(
        self,
        df,
        column_a,
        list_of_values_a,
        column_b,
        list_of_values_b,
        column_c,
        list_of_values_c,
    ):
        """
        Checks if column_a contains list_of_values_a, column_b contains list_of_values_b, and column_c contains list_of_values_c.
        """
        results = {}

        # Check if both column_a and column_b are present in the DataFrame
        if (column_a not in df.columns) or (column_b not in df.columns):
            print(
                f"Column '{column_a}' or '{column_b}' does not exist in the dataframe."
            )
            return results

        # Convert relevant columns and lists of values to uppercase for case-insensitive comparison
        df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        list_of_values_a_upper = [
            value.upper() if isinstance(value, str) else value
            for value in list_of_values_a
        ]
        list_of_values_b_upper = [
            value.upper() if isinstance(value, str) else value
            for value in list_of_values_b
        ]
        list_of_values_c_upper = [
            value.upper() if isinstance(value, str) else value
            for value in list_of_values_c
        ]

        # Filter rows where column_a contains values from list_of_values_a and column_b contains values from list_of_values_b
        selected_df = df[
                (pd.notna(df[column_a]) & df[column_a].isin(list_of_values_a_upper)) &
                (pd.notna(df[column_b]) & df[column_b].isin(list_of_values_b_upper))
            ]
        # Check if values in column_c are in the specified set of values
        result = ge.dataset.PandasDataset(
            selected_df
        ).expect_column_values_to_be_in_set(
            column_c,
            list_of_values_c_upper,
            result_format="COMPLETE",
            include_config=False,
            catch_exceptions=False,
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Value Check",
                        f"If {column_a} contains one of the values: {', '.join(list_of_values_a_upper)}",
                        f"and {column_b} contains one of the values: {', '.join(list_of_values_b_upper)}",
                        f"then values in {column_c} should be in {', '.join(list_of_values_c_upper)}",
                        "",
                        "OK",
                    ],
                }
            },
        )

        results[column_a] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    # Example: 'If "Product name (SKU)" contains DANONINO, Brand cluster must be equal to DANONINO'
    def test_select_data_and_check_values(
        self,
        df: pd.DataFrame,
        column_a: str,
        column_b: str,
        value_b: str,
        value_a: str,
    ):
        """
        Selects data based on condition and checks for values in another column.
        """
        results = {}

        # Check if both column_a and column_b exist in the DataFrame
        if (column_a not in df.columns) or (column_b not in df.columns):
            print(
                f"Column '{column_a}' or '{column_b}' does not exist in the dataframe."
            )
            return results

        # Select rows where column_a contains any of the specified strings
        selected_df = df[df[column_a].str.contains(value_a, na=False)]

        # Check if column_b is in the set of values {value_b} for the selected rows
        result = ge.dataset.PandasDataset(
            selected_df
        ).expect_column_values_to_be_in_set(column_b, {value_b})

        # Add additional notes or metadata to the result
        result.meta["notes"] = {
            "format": "markdown",
            "content": [
                "Validity",
                f"If {column_a} contains {value_a}, then {column_b} should be equal to one of {value_b}",
                "",
                "OK",
            ],
        }

        results[column_b] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "meta": result.meta,
            "success": result.success,
        }

        return results

    

    def check_value_in_list_than_column_check_is_less_than(self, df, column_condition, list_of_values, column_check, max_value):
        """
        Checks if column_condition is in the list_of_values and then checks if column_check values are <= max_value.
        """
        results = {}

        # Check if both column_condition and column_check are present in the DataFrame
        if (column_condition not in df.columns) or (column_check not in df.columns):
            print(f"Column '{column_condition}' or '{column_check}' does not exist in the dataframe.")
            return results

        try:
            # Convert relevant columns and values to uppercase for case-insensitive comparison
            df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
            list_of_values_upper = [value.upper() if isinstance(value, str) else value for value in list_of_values]

            # Filter rows where column_condition contains values from the specified list
            selected_df = df[df[column_condition].isin(list_of_values_upper)]

            # Define the expectation using Great Expectations
            expectation_suite = ge.dataset.ExpectationSuite()
            expectation_config = {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": column_check,
                    "min_value": None,
                    "max_value": max_value,
                },
            }
            expectation_suite.add_expectation(expectation_config)

            # Get a Batch of data for validation
            batch = ge.dataset.PandasDataset(selected_df)

            # Validate the batch against the expectation
            result = batch.validate(expectation_suite, result_format="COMPLETE")

            # Add additional notes or metadata to the result
            result.meta["notes"] = {
                "format": "markdown",
                "content": [
                    "Value Check",
                    f"If {column_condition} is in {list_of_values_upper}",
                    f"then values in {column_check} should be <= {max_value}",
                    "OK",
                ],
            }

            # Save the expectation suite after adding the expectation
            context.save_expectation_suite(expectation_suite)

            results[column_condition] = {
                "result": result.result,
                "exception_info": result.exception_info,
                "meta": result.meta,
                "success": result.success,
            }

        except KeyError as e:
            print(f"Warning: An error occurred while performing the validation: {e}")

        return results


#%%


class Data_quality_check_Nutripride(Data_quality_check):
    def test_add_toppers_if_flake(self,df: pd.DataFrame, column: str):
        results = {}

        dataset = PandasDataset(df)

        result = dataset.expect_column_values_to_match_regex(
            column=column,
            regex=r"^((?!- TOPPERS$).*FLAKE.*)$",
            result_format="COMPLETE",
            meta={
                "notes": {
                    "format": "markdown",
                    "content": [
                        "Validity",
                        f"If SKU contains 'FLAKE' and doesn't end with '- TOPPERS', it should match the regex pattern '^((?!- TOPPERS$).*FLAKE.*)$'",
                        "",
                        "OK",
                    ]
                }
            }
        )
        

        results[column] = {
            "result": result.result,
            "exception_info": result.exception_info,
            "current_value":df[column].tolist(),
            "expected_value":"",
            "meta": result.meta,
            "success": result.success,
        }    
        expectation_suite.add_expectation(result.expectation_config)
        context.save_expectation_suite(expectation_suite)

        return results
    



