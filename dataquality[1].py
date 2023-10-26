"""
The dataquality module is developed to enforce checks and constraints on data
for ensuring correctness and quality of the data.
"""
# import packages
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import os
import json

# class for defining checks
class DataQuality():
    """
    The DataQuality class (defined in dataquality.dataquality)
    provides methods to enforce checks and constraints on the data.
    """
    def __init__(self,data):
        self.data = data
        self.checks_table_level = {}
        self.checks_column_level = {}
        self.total_records = len(data)
        self.dq_result = {}
        
    # to create table level results suite
    def create_results_table_level(self, jsres):
        
        if len(self.checks_table_level.keys()) != 0:
            self.checks_table_level[jsres["check_type"]] = jsres
        else:
            self.checks_table_level = {jsres["check_type"] : jsres}
            
    # to create column level results suite
    def create_results_column_level(self, jsres):
        
        if jsres["column_name"] in self.checks_column_level.keys():
            self.checks_column_level[jsres["column_name"]][jsres["check_type"]] = jsres
        else:
            self.checks_column_level[jsres["column_name"]] = {jsres["check_type"] : jsres}
            
    # to generate the final results suite
    def generate_dq_result(self, save_json = True, path = ""):
                
        checks_table = self.checks_table_level.copy()
        checks_column = self.checks_column_level.copy()
        checks_table_failed = 0
        checks_table_passed = 0
        checks_column_failed = 0
        checks_column_passed = 0
        
        self.dq_result["checks_table_level"] = checks_table
        self.dq_result["checks_column_level"] = checks_column
        
        for table_key in checks_table.keys():
            if checks_table[table_key]["result"] == False:
                checks_table_failed += 1
            else:
                checks_table_passed += 1
        
        for column_key in checks_column.keys():
            for column_check_key in checks_column[column_key]:
                if checks_column[column_key][column_check_key]["result"] == False:
                    checks_column_failed += 1
                else:
                    checks_column_passed += 1
                    
        self.dq_result["checks_table_level"]["checks_table_total"] = checks_table_passed + checks_table_failed
        self.dq_result["checks_table_level"]["checks_table_passed"] = checks_table_passed
        self.dq_result["checks_table_level"]["checks_table_failed"] = checks_table_failed
        
        if checks_table_failed > 0:
            self.dq_result["checks_table_level"]["result"] = False
        else:
            self.dq_result["checks_table_level"]["result"] = True
        
        self.dq_result["checks_column_level"]["checks_column_total"] = checks_column_passed + checks_column_failed
        self.dq_result["checks_column_level"]["checks_column_passed"] = checks_column_passed
        self.dq_result["checks_column_level"]["checks_column_failed"] = checks_column_failed
                
        if checks_column_failed > 0:
            self.dq_result["checks_column_level"]["result"] = False
        else:
            self.dq_result["checks_column_level"]["result"] = True
            
        self.dq_result["checks_total"] = checks_table_passed + checks_table_failed + checks_column_passed + checks_column_failed
        self.dq_result["checks_passed"] = checks_table_passed + checks_column_passed
        self.dq_result["checks_failed"] = checks_table_failed + checks_column_failed
        self.dq_result["total_records_actual"] = self.total_records
        self.dq_result["total_records_dropped"] = self.total_records - len(self.data)

        if self.dq_result["checks_failed"] > 0:
            self.dq_result["result"] = False
        else:
            self.dq_result["result"] = True
        
        if save_json == True:
        
            if path == "":
                path = os.path.abspath(os.getcwd())

            save_as = os.path.join(path,"dq_result.json")

            with open(save_as, 'w') as fp:
                json.dump(self.dq_result, fp,  indent=4)
        
    
    ####TABLE LEVEL#####
        
    # check sparsity i.e percentage of missingness in the data
    def check_sparsity(self, threshold_percent = 10):
        """
        Check whether the sparsity of the dataframe is below the specific threshold. 

        Parameters 
        ----------
        threshold_percent : {int, float}
            (default : 10)
            Used to specify the percentage of sparsity allowed.

        Returns 
        -------
        jsres : A json format results suite.

        See Also
        --------
        check_column_values_not_null : Check whether the column values are not null.

        Notes
        -----
        Sparsity of a dataframe is defined by the percentage of missingness in it. 
        This is calculated by looking at the number of cells in the dataframe that are
        empty or null over the total number of cells. The total number of cells is given
        by rows*columns of the dataframe.
        
        Examples
        --------
        To apply the function, we need to pass the *threshold_percent* which defines the
        percentage of missingness you expect in the data. 

        >>> dq.check_sparsity(threshold_percent = 30)
        {'check_type': 'Check sparsity to be below specific threshold',
         'threshold': {'threshold_percent': 30, 'threshold_value': 57000},
         'observed_value': {'total_cells': 190000,
          'total_empty_cells': 49495,
          'sparsity_percent': 26.05},
         'result': True}

        If the percentage of missingness is above the specific threshold, the result is false.

        >>> dq.check_sparsity(threshold_percent = 20)
        {'check_type': 'Check sparsity to be below specific threshold',
         'threshold': {'threshold_percent': 20, 'threshold_value': 38000},
         'observed_value': {'total_cells': 190000,
          'total_empty_cells': 49495,
          'sparsity_percent': 26.05},
         'result': False}
        """
        # get total number of values in the dataframe
        total_cells = int(self.data.shape[0] * self.data.shape[1])
        # empty_cells
        total_empty_cells = int((self.data.isnull().to_numpy() == True).sum())
        # calculate threshold values based on number of cells and threshold percentage 
        try:
            threshold_val = int((total_cells) * threshold_percent/100)
        except TypeError:
            raise TypeError("expected an int or float value for threshold_percent, instead received " + str(threshold_percent))
        
        jsres = {}
        temp_threshold = {}
        temp_observed = {}
        jsres["check_type"] = "Check sparsity to be below specific threshold"
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val      
        temp_observed["total_cells"] = total_cells
        temp_observed["total_empty_cells"] = total_empty_cells
        sparsity_percent = np.round((self.data.isnull().to_numpy() == True).mean() * 100,2)
        temp_observed["sparsity_percent"] = sparsity_percent
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp_observed 
        
        if sparsity_percent <= threshold_percent:
            jsres["result"] = True
        else:
            jsres["result"] = False
            
        self.create_results_table_level(jsres)
        
        return json.dumps(jsres)

    
    # to check the type of the column
    def check_column_datatype(self, column, expected_type):
        """
        Check wether the datatype of the column is as expected.

        Parameters 
        ----------
        column : str
            Name of the column on which the check needs to be implemented.
        expected_type : type {int, float, object, bool}
            The expected type of the column. The class such as int, float, str etc.
            
        Returns 
        -------
        jsres : A json format results suite.

        Notes
        -----
        Column values can be of datatypes belonging to a specific class, such as *int32* which is 
        an instance of the class *int*. By using *expected_type*, we specify the class to see if the 
        column values datatype is as expected.
        
        Examples
        --------
        >>> dq.check_column_datatype(column = "rssi_dbm", expected_type = float)
        {'check_type': 'check column datatype',
        'column_name': 'rssi_dbm',
        'expected_value': "<class 'float'>",
        'observed_value': 'float64',
        'result': True}

        In the above example, if we pass *int* as the *expected_type*, the result is going to be False.

        >>> dq.check_column_datatype(column = "rssi_dbm", expected_type = int)
        {'check_type': 'check column datatype',
        'column_name': 'rssi_dbm',
        'expected_value': "<class 'int'>",
        'observed_value': 'float64',
        'result': False}
        """
        jsres = {}
        jsres["check_type"] = "check column datatype"
        jsres["column_name"] = column
        jsres["expected_value"] = str(expected_type)
        jsres["observed_value"] = str(self.data[column].dtype)

        # condition to check if the given type is equal to the actual type of the column
        if expected_type == int:
            if self.data[column].dtype in [int,np.int32,np.int64]:
                # if true return True     
                jsres["result"] = True
            else:
                # if false return False 
                jsres["result"] = False
                
        elif expected_type == float:
            if self.data[column].dtype in [float,np.float32,np.float64]:
                # if true return True     
                jsres["result"] = True
            else:
                # if false return False 
                jsres["result"] = False
        
        elif expected_type == object:
            if self.data[column].dtype in [object,np.object]:
                # if true return True     
                jsres["result"] = True
            else:
                # if false return False 
                jsres["result"] = False
        
        elif expected_type == bool:
            if self.data[column].dtype in [bool,np.bool]:
                # if true return True     
                jsres["result"] = True
            else:
                # if false return False 
                jsres["result"] = False
        else:
            raise ValueError("unknown value for expected_type :" + str(expected_type))
        
        self.create_results_column_level(jsres)
                
        return json.dumps(jsres)
    
    
    # to check if a list of columns match the columns in the dataframe
    def check_columns_to_match_set(self, columns_list, qlevel = "verify"):
        """
        Check whether the list of columns specified match the columns present in the dataframe.

        Parameters 
        ----------
        column_list : List
            Used to input the list of columns which we want to match with the columns in the dataframe.
        qlevel : str {"verify", "assert"}
            (default : "verify")
            qlevel is the Qualification level of the check.
            If "verify", then checks if the list of expected columns are present in the dataframe.
            If "assert", then checks if all the columns in the dataframe are present in the list of expected columns.

        Returns 
        -------
        jsres : A json format results suite.

        Examples
        --------
        To check if a certain set of columns are present in the dataframe, *verify* can be used as the *qlevel*. 
        By using *verify*, the column names in *column_list* are matched with the actual set of columns present in
        the dataframe. If all the columns passed are present, then the result is True.

        >>> dq.check_columns_to_match_set(columns_list = ["rssi_dbm","crssi_dbm","kp_in_track"], qlevel = "verify")
        {'check_type': 'check dataframe columns to match a set of columns',
         'qualification_level': 'verify',
         'expected_value': ['rssi_dbm', 'crssi_dbm', 'kp_in_track'],
         'observed_value': {'columns_in_data': ['time',
           'train_id',
           'train_speed',
           'obm_color',
           'obm_direction',
           'track_id',
           'kp_in_track',
           'rssi_dbm',
           'ho_count',
           'crssi_dbm'],
          'columns_not_in_data': [],
          'columns_unexpected': ['obm_direction',
           'obm_color',
           'train_speed',
           'train_id',
           'time',
           'ho_count',
           'track_id']},
         'result': True}
         
         To get an exact match of all the columns present in the dataframe and *column_list*, *assert* can be used
         as the *qlevel*. In this case, the above check will result as False.
         
         >>> dq.check_columns_to_match_set(columns_list = ["rssi_dbm","crssi_dbm","kp_in_track"], qlevel = "assert")
         {'check_type': 'check dataframe columns to match a set of columns',
         'qualification_level': 'assert',
         'expected_value': ['rssi_dbm', 'crssi_dbm', 'kp_in_track'],
         'observed_value': {'columns_in_data': ['time',
           'train_id',
           'train_speed',
           'obm_color',
           'obm_direction',
           'track_id',
           'kp_in_track',
           'rssi_dbm',
           'ho_count',
           'crssi_dbm'],
          'columns_not_in_data': [],
          'columns_unexpected': ['obm_direction',
           'obm_color',
           'train_speed',
           'train_id',
           'time',
           'ho_count',
           'track_id']},
         'result': False}
        """
        data_columns_list = list(self.data.columns)
        
        jsres = {}
        temp = {}
        jsres["check_type"] = "check dataframe columns to match a set of columns"
        jsres["qualification_level"] = qlevel
        jsres["expected_value"] = columns_list
        temp["columns_in_data"] = data_columns_list
        temp["columns_not_in_data"] = list(set(columns_list) - set(data_columns_list))
        temp["columns_unexpected"] = list(set(data_columns_list) - set(columns_list))
        jsres["observed_value"] = temp
        
        if qlevel == "verify":
            if not bool(set(columns_list) - set(data_columns_list)):
                jsres["result"] = True
            else:
                
                jsres["result"] = False
                
        elif qlevel == "assert":
            if not bool(set(columns_list) - set(data_columns_list)) and not bool(set(data_columns_list) - set(columns_list)):
                jsres["result"] = True
            else:
                jsres["result"] = False
                
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
        
        self.create_results_table_level(jsres)
        
        return json.dumps(jsres)

    
    # to check if the table has rows in the given range
    def check_table_row_count_to_be_between(self, row_count_range):
        """
        Check table row count to be between a specific range.

        Parameters
        ----------
        row_count_range : List
            The argument is used to input the minimum and maximum values for the number of records the table should contain.
            The range min and max are exclusive. For example, if the row count has to be 30. The row_count_range to specify
            the exact number will be [29,31].

        Returns
        -------
        jsres : A json format results suite.
        
        Examples
        --------
        >>> dq.check_table_row_count_to_be_between(row_count_range = [15000,20000])
        {'check_type': 'check the number of records to be in a specific range',
         'expected_value': [15000, 20000],
         'observed_value': 10000,
         'result': False}
        """
        jsres = {}
        jsres["check_type"] = "check the number of records to be in a specific range"
        jsres["expected_value"] = row_count_range
        
        # get the total number rows in the table
        data_row_count = len(self.data)
        jsres["observed_value"] = data_row_count
        # condition to check if the total number of rows in the table is between the given range
        if row_count_range[0] < data_row_count < row_count_range[1]:
            # if true return True and the total number of rows
            jsres["result"] = True
        else:
            # if false return False and the total number of rows
            jsres["result"] = False
        
        self.create_results_table_level(jsres)
        
        return json.dumps(jsres)

    
    # method to check if the column maximum value is between a specific range
    def check_timeseries_to_be_stationary(self, column, statistical_test = "adf", significance_level = 5):
        """
        Check wether a time series is stationary.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        statistical_test : str {"adf"}
            The statistical test that is used to check if a time series is stationary or non-stationary.
            Currently the function only supports augmented dickey fuller test.
        significan_level : int, float
            The significance level at which the null hypothesis is rejected.

        Returns
        -------
        jsres : A json format results suite.
        
        Notes
        -----
        *Time series are stationary if they do not have trend or seasonal effects. When a time series is stationary, 
        it can be easier to model. Statistical modeling methods assume or require the time series to be stationary 
        to be effective.*
        
        The **Augmented Dickey-Fuller test** is a type of statistical test called a unit root test.

        The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

        There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. 
        It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

        The null hypothesis of the test is that the time series can be represented by a unit root, that it is not 
        stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) 
        is that the time series is stationary.

        **Null Hypothesis (H0)**: If failed to be rejected, it suggests the time series has a unit root, meaning it is 
        non-stationary. It has some time dependent structure.
        **Alternate Hypothesis (H1)**: The null hypothesis is rejected; it suggests the time series does not have a unit root, 
        meaning it is stationary. It does not have time-dependent structure.
        
        We interpret this result using the *p-value* from the test. A p-value below a threshold (such as 5% or 1%) suggests 
        we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the 
        null hypothesis (non-stationary).

        - p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
        - p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
        
        Examples
        --------
        >>> dq.check_timeseries_to_be_stationary("crssi_dbm", statistical_test = "adf", significance_level = 5)
        {'check_type': 'check time series to be stationary',
         'column_name': 'crssi_dbm',
         'statistical_test': 'augmented dickey-fuller test',
         'significance_level': 5,
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'test_statistic': -7.75079,
          'p_value': 0.0,
          'critical_values': {'1%': -3.43101, '5%': -2.86183, '10%': -2.56692}},
         'result': True}
         
        we can see that the test statistic value of -7.7. The more negative this statistic, the more likely we are to 
        reject the null hypothesis (we have a stationary dataset).

        As part of the output, we get a look-up table to help determine the ADF statistic. We can see that our statistic 
        value of -7.7 is less than the value of -3.43 at 1%.

        This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability 
        that the result is a statistical fluke).

        Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or 
        does not have time-dependent structure.
        """
        if statistical_test == "adf":
            statistical_test = "augmented dickey-fuller test"
        else:
            raise ValueError("Unknown value for statistical_test : " + statistical_test)
        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
                
        jsres = {}
        temp = {}
        temp_critical_values = {}
        jsres["check_type"] = "check time series to be stationary"
        jsres["column_name"] = column
        
        jsres["statistical_test"] = statistical_test
        jsres["significance_level"] = significance_level
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        
        timeseries = self.data[column].values
        
        test_result = adfuller(timeseries)
        
        temp["test_statistic"] = np.round(test_result[0],5)
        temp["p_value"] = np.round(test_result[1],5)
        
        for key, value in test_result[4].items():
            temp_critical_values[key] = np.round(value,5)
            
        temp["critical_values"] = temp_critical_values
        jsres["observed_value"] = temp
        
        if test_result[4][f"{significance_level}%"] >= test_result[0]:
            jsres["result"] = True
        else:
            jsres["result"] = False
        
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    
    
    def check_column_values_not_null(self, column, qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False):
        """
        Check whether the column values are not null.

        Parameters
        ----------
        column : str
            Name of the column on which the check needs to be implemented.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if the number of null values present, is less than the permissible threshold.
            If "assert", then checks if there are any null values.
        threshold_percent : int 
            (default : 10)
            Used to specify the percentage of null values allowed when the qlevel is "verify".
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of null values present in the column values,
            are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        Examples
        --------
        >>> dq.check_column_values_not_null("obm_direction", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'Check column values to be not null',
         'column_name': 'obm_direction',
         'qualification_level': 'verify',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0},
         'drop_unexpected': False,
         'result': True}
         
         If null values are expected in the column values, we can use *qlevel* as "verify" and *threshold_percent* to specify the 
         percentage of null values to be expected. If the percentage of null values is below this threshold, the result is True.
         
         >>> dq.check_column_values_not_null("rssi_dbm", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False)
         {'check_type': 'Check column values to be not null',
         'column_name': 'rssi_dbm',
         'qualification_level': 'verify',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505},
         'drop_unexpected': False,
         'result': True}
         
         If we want to drop the unexpected values, i.e null values in this scenario, we can use *drop_unexpected_rows* to be True
         which will drop the null values present in the data irrespective of the result, when *qlevel* is "verify".
         
         >>> dq.check_column_values_not_null("rssi_dbm", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = True)
         {'check_type': 'Check column values to be not null',
         'column_name': 'rssi_dbm',
         'qualification_level': 'verify',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505},
         'drop_unexpected': True,
         'result': True}
         
         If no null values are expected in the column values, using *qlevel* = "assert" will check that there is no null value
         present at all. In this scenario, *threshold_percent* and *drop_unexpected_rows* are optional. Both of the *params* are
         only functional when *qlevel* is "verify". Threshold is save with a value 0.
         
         >>> dq.check_column_values_not_null("rssi_dbm", qlevel = "assert", threshold_percent = 10, drop_unexpected_rows = False)
         {'check_type': 'Check column values to be not null',
         'column_name': 'rssi_dbm',
         'qualification_level': 'assert',
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505},
         'drop_unexpected': False,
         'result': False}
        """
        # get total number of not null values in the column
        data_notnull_val = self.data[column].notna().sum()
        # get total number of values in the column
        data_total_val = len(self.data[column])

        threshold_val = int((threshold_percent/100) * data_total_val)

        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "Check column values to be not null"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel

        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val      
        temp["total_values"] = int(data_total_val)
        temp["not_null_values"] = int(data_notnull_val)
        temp["null_values"] = int(data_total_val - data_notnull_val)
        

        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows

        if qlevel == "verify": 
            if (data_notnull_val == data_total_val) or (data_total_val - data_notnull_val <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
                
            if drop_unexpected_rows == True:
                self.data = self.data[self.data[column].notna()].reset_index(drop = True)

        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if data_notnull_val == data_total_val:
                jsres["result"] = True
            else:
                jsres["result"] = False
                
        else:
            raise ValueError("unknown value for qualification level: " + qlevel)
            
        self.create_results_column_level(jsres)

        return json.dumps(jsres)
    
    # to check if the column has distinct values
    def check_column_values_to_be_distinct(self, column, qlevel = "verify", threshold_percent = 10):
        """
        Check whether a column has a distinct values.

        Parameters
        ----------
        column : str
            Name of the column on which the check needs to be implemented.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are distinct or the number of non-distinct values is less than the permissible threshold.
            If "assert", then checks if all values are distinct.
        threshold_percent : int 
            (default : 10)
            Used to specify the percentage of non-distinct values allowed when the qlevel == "verify".

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_values_to_be_unique : Check whether a column has a unique value.
        
        Notes
        -----
        If all the values present in the column are different from each other, then it is having distinct values. For example,
        
        +------------+------------+
        | signals    |access_pts  |
        +============+============+
        | Signal 1   | AP1        |
        +------------+------------+
        | Signal 1   | AP2        |
        +------------+------------+
        | Signal 2   | AP3        |
        +------------+------------+
        | Signal 3   | AP4        | 
        +------------+------------+
        
        In the above table, access_pts column has distinct values, whereas signals column is not considered to be distinctive because
        Signal 1 and Signal 2 are repeating twice.
        
        Examples
        --------
         >>> dq.check_column_values_to_be_distinct("train_id", qlevel = "verify", threshold_percent = 10)
         {'check_type': 'check values to be distinct',
         'column_name': 'train_id',
         'qualification_level': 'verify',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'distinct_values': 1,
          'distinct_percent': 0.01,
          'non_distinct_values': 9999,
          'non_distinct_percent': 99.99},
         'result': False}
         
         If non distinct values are expected in the column values, we can use *qlevel* as "verify" and *threshold_percent* to specify the 
         percentage of non distinct values to be expected. If the percentage of these values is below the threshold, the result is True.
         
         >>> dq.check_column_values_to_be_distinct("time", qlevel = "verify", threshold_percent = 10)
         {'check_type': 'check values to be distinct',
         'column_name': 'time',
         'qualification_level': 'verify',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'distinct_values': 9999,
          'distinct_percent': 99.99,
          'non_distinct_values': 1,
          'non_distinct_percent': 0.01},
         'result': True}
         
         If no non distinct values are expected in the column values, using *qlevel* = "assert" will check that there is no non distinct values
         present at all. In this scenario, *threshold_percent* is optional. It is only functional when *qlevel* is "verify". Threshold is saved
         with a value 0 in the suite.
         
         >>> dq.check_column_values_to_be_distinct("time", qlevel = "assert", threshold_percent = 10)
         {'check_type': 'check values to be distinct',
         'column_name': 'time',
         'qualification_level': 'assert',
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'distinct_values': 9999,
          'distinct_percent': 99.99,
          'non_distinct_values': 1,
          'non_distinct_percent': 0.01},
         'result': False}
         
        """
        # get total number of not null values in the column
        data_unique_val = self.data[column].nunique()
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total not null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()

        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check values to be distinct"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        temp["total_values"] = int(data_total_val)
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["distinct_values"] = int(data_unique_val)
        temp["distinct_percent"] = np.round((data_unique_val/data_total_val) * 100,2)
        temp["non_distinct_values"] = int(data_total_val - data_unique_val)
        temp["non_distinct_percent"] = np.round(((data_total_val - data_unique_val)/data_total_val) * 100, 2)
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        
        if qlevel == "verify":
            if (data_total_val == data_unique_val) or (data_total_val - data_unique_val <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
                
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            if self.data[column].is_unique:
                jsres["result"] = True
            else:
                jsres["result"] = False
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
            
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    

    # to check if the column has distinct values
    def check_column_values_to_be_unique(self, column):
        """
        Check whether a column has a unique value.

        Parameters
        ----------
        column : str
            Name of the column on which the check needs to be implemented.

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_values_to_be_distinct : Check whether a column has a distinct values.
        
        Notes
        -----
        If all the values present in the column are non-distinct, then it is said to have unique values. For example,
        
        +------------+------------+
        | signals    |access_pts  |
        +============+============+
        | Signal 1   | AP1        |
        +------------+------------+
        | Signal 1   | AP2        |
        +------------+------------+
        | Signal 1   | AP3        |
        +------------+------------+
        | Signal 1   | AP4        | 
        +------------+------------+
        
        In the above table, access_pts column has distinct values, whereas signals column is considered to have unique values. There
        is only a single value i.e "Signal 1".
        
        Examples
        --------
         >>> dq.check_column_values_to_be_unique(column = "train_id")
         {'check_type': 'check values to be unique',
         'column_name': 'train_id',
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'unique_values': 1},
         'result': True}
         
         If there are multiple or distinct values in the column, then the result is False.
         
         >>> dq.check_column_values_to_be_unique(column = "rssi_dbm")
         {'check_type': 'check values to be unique',
         'column_name': 'rssi_dbm',
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505,
          'unique_values': 7},
         'result': False}
        """
        # get total number of not null values in the column
        data_unique_val = self.data[column].nunique()
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total not null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        jsres = {}
        temp = {}
        jsres["check_type"] = "check values to be unique"
        jsres["column_name"] = column
        
        temp["total_values"] = int(data_total_val)
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["unique_values"] = int(data_unique_val)
        
        jsres["observed_value"] = temp
        
        if data_unique_val == 1:
            jsres["result"] = True
        else:
            jsres["result"] = False
            
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)

    
     # to check if the column has values in a given range
    def check_column_values_to_be_between(self, column, values_range, qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False):
        """
        Check wether a column has values in a specific range.

        Parameters
        ----------    
        column : str
            Name of the column on which the check needs to be implemented.
        values_range : list
            The minimum and maximum value range in which the column values are expected to be present. e.g. [10,40]
            where the minimum and maximum values are inclusive.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are present in the specified range or the number of values beyond the 
            specified range is less than the permissible threshold.
            If "assert", then checks if all values are present in the specified range.
        threshold_percent : int 
            (default : 10)
            Used to specify the percentage of values allowed that are not in the specified range when the qlevel == "verify"
        allow_null : bool 
            (default : True)
            If True, will include Null values in the evaluation, otherwise will exclude.
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values present outside of the specified range
            are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_value_length_to_be_between : Check whether length of the values of a column is in a specific range.
        check_column_max_to_be_between : Check wether a column's maximum value is in a specific range.
        check_column_min_to_be_between : Check wether a column's minimum value is in a specific range.
        
        Examples
        --------
        If all the values are inside the expected range, the result is True.
        
        >>> dq.check_column_values_to_be_between(column = "crssi_dbm", values_range = [-110,-40], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check values to be in a specific range',
         'column_name': 'crssi_dbm',
         'qualification_level': 'verify',
         'expected_values_range': [-110, -40],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'values_in_range': 10000,
          'values_out_range': 0},
         'drop_unexpected': False,
         'result': True}
         
        If the column has Null values, these can be excluded by specifying *include_null* as False. Since Null values will be considered to be 
        out of range, this arg will allow us to proceed with our analysis by not considering them while checking if the values are in the specified 
        range. 
         
        For example, below the column has Null values. When we apply the function with *include_null* to be True, it will include the null values which
        will be considered to be outside the specified range. In the results suite, we observe that the values_out_range is equal to the null_values. The 
        check has a result True, because *qlevel* is "verify" and the number of values outside the specified range is within the threshold. 
         
        >>> dq.check_column_values_to_be_between("rssi_dbm", [-110,-40], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
         {'check_type': 'check values to be in a specific range',
         'column_name': 'rssi_dbm',
         'qualification_level': 'verify',
         'expected_values_range': [-110, -40],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505,
          'values_in_range': 9495,
          'values_out_range': 505},
         'drop_unexpected': False,
         'result': True}
         
        If we pass *include_null* as False, it will not include Null values during the evaluation, although no rows will be dropped from the actual data.
        Here, we observe that the not_null_values is equal to the values_in_range. It is also observed that the null_values i.e 505 in total, is not included during the 
        analysis. This is why we have values_out_range equal to 0. This means all the non-null values are within the range.
        
        >>> dq.check_column_values_to_be_between("rssi_dbm", [-110,-40], qlevel = "verify", threshold_percent = 10, include_null = False, drop_unexpected_rows = False)
        {'check_type': 'check values to be in a specific range',
         'column_name': 'rssi_dbm',
         'qualification_level': 'verify',
         'expected_values_range': [-110, -40],
         'include_null': False,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505,
          'values_in_range': 9495,
          'values_out_range': 0},
         'drop_unexpected': False,
         'result': True}
         
        To drop the unexpected values i.e values out of range in this case, we can pass *drop_unexpected_rows* as True. 
         
        **Please note** that when *include_null* is
        True and *drop_unexpected_rows* is True, it will not only drop the values out of range, but also drop Null values present in the column.
        If *include_null* is False and *drop_unexpected_rows* is True, then it will drop only the values out of the specified range. In both the cases,
        since we drop the unexpected values, the result is changed to True.

        In the below scenario, it drops all the Null values present in the column along with values outside the range.
         
        >>> dq.check_column_values_to_be_between("rssi_dbm", [-110,-40], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = True)
        {'check_type': 'check values to be in a specific range',
         'column_name': 'rssi_dbm',
         'qualification_level': 'verify',
         'expected_values_range': [-110, -40],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505,
          'values_in_range': 9495,
          'values_out_range': 505},
         'drop_unexpected': True,
         'result': True}
         
        If *qlevel* is "assert", then the params threshold_percent, include_null, drop_unexpected_rows are not functional. If atleast one of the 
        column values is outside the specified range, the result is False, irrespective of any condition. *include_null* and *drop_unexpected_rows*
        are saved with there default values, whereas threshold becomes 0.
        
        >>> dq.check_column_values_to_be_between("rssi_dbm", [-110,-40], qlevel = "assert", threshold_percent = 10, include_null = False, drop_unexpected_rows = True)
        {'check_type': 'check values to be in a specific range',
         'column_name': 'rssi_dbm',
         'qualification_level': 'assert',
         'expected_values_range': [-110, -40],
         'include_null': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 9495,
          'null_values': 505,
          'values_in_range': 9495,
          'values_out_range': 505},
         'drop_unexpected': False,
         'result': False}
        """
        if qlevel == "verify":
            if include_null == False:
                # get total number of values in the column present in the specific range when incude null is false
                data_range_len = self.data[self.data[column].notna()][column].between(values_range[0],values_range[1]).sum()
            else:
                # get total number of values in the column present in the specific range when incude null is true
                data_range_len = self.data[column].between(values_range[0],values_range[1]).sum()
                
        elif qlevel == "assert":
            include_null = True
            # get total number of values in the column present in the specific range when incude null is true
            data_range_len = self.data[column].between(values_range[0],values_range[1]).sum()
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
            
        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total not null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)

        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check values to be in a specific range"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        jsres["expected_values_range"] = values_range
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["include_null"] = include_null
        temp["total_values"] = int(data_total_val)
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["values_in_range"] = int(data_range_len)
        
        if include_null == True:
            temp["values_out_range"] = int(data_total_val - data_range_len)
        else:
            temp["values_out_range"] = int(data_not_null_val - data_range_len)
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            # condition to check if the total number of values in the given range is equal to the total number of values present in the column
            if (temp["values_out_range"] == 0) or (temp["values_out_range"] <= threshold_val):
                # if true return True and a string mentioning the values are in a specified range
                jsres["result"] = True
            else:
                # if false return False and a string mentioning the number of values which are not in the specified range
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                if include_null == True:
                    self.data = self.data[self.data[column].between(values_range[0],values_range[1])].reset_index(drop = True)
                    jsres["result"] = True
                elif include_null == False:
                    self.data = self.data[(self.data[column].isna()) | (self.data[column].between(values_range[0],values_range[1]))].reset_index(drop = True)
                    jsres["result"] = True
                else:
                    raise ValueError("Please enter a bool value for include_null :" + include_null)

        else:
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if temp["values_out_range"] == 0:
                jsres["result"] = True
            else:
                jsres["result"] = False
                
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    
    
    # method to check if the column has only the specified values
    def check_column_values_to_be_in_set(self, column, list_values, qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False):
        """
        Check wether a column has only a specific set of values.

        Parameters
        ----------    
        column : str
            Name of the column on which the check needs to be implemented.
        list_values : list
            The set of values which are expected in the column values.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values of the column belong to the specific set or the number of values out of 
            the specific set is less than the permissible threshold.
            If "assert", then checks if all values of the column are present in the specific set.
        threshold_percent : int 
            (default : 10)
            Used to specify the percentage of values allowed that are not in the specific set of values when the qlevel == "verify"
        include_null : bool 
            (default : True)
            If True, will include Null values in the evaluation, otherwise will exclude.
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values not present in the specific set of values
            are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_columns_to_match_set : Check whether the list of columns specified match the columns present in the dataframe.
        
        Examples
        --------
        To check if a column has values belonging to a set of values, we can use *qlevel* as "verify". Below we use a column, which has 
        a value i.e. "Tail" or "Head". We therefore mention the *list_values* which is the expected set of values to be ["Head","Tail"].
        
        >>> dq.check_column_values_to_be_in_set(column = "obm_direction", list_values = ["Head","Tail"], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column to have values from a specific set',
         'column_name': 'obm_direction',
         'qualification_level': 'verify',
         'expected_values_list': ['Head', 'Tail'],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'column_values_list': ['Tail'],
          'values_not_in_column_list': ['Head'],
          'unexpected_values_list': [],
          'unexpected_values_count': 0,
          'unexpected_values_percent': 0},
         'drop_unexpected': False,
         'result': True}
         
        'column_values_list' is the unique list of values present in the column.
        'values_not_in_column_list' is the list of values which are present in the *list_values* but not in the 'column_values_list'.
        'unexpected_values_list' is the list of values which are present in 'column_values_list' and not in *list_values*.
        
        If the column has Null values, When we apply the function with *include_null* to be True, it will include the null values which 
        will be considered to be a value in the 'column_values_list'. 
        
        >>> dq.check_column_values_to_be_in_set(column = "ws_source_ip", list_values = ['10.69.147.24 '], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column to have values from a specific set',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_list': ['10.69.147.24 '],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'column_values_list': ['10.69.147.24 ', nan],
          'values_not_in_column_list': [],
          'unexpected_values_list': [nan],
          'unexpected_values_count': 9495,
          'unexpected_values_percent': 94},
         'drop_unexpected': False,
         'result': False}
         
        Since *qlevel* is "verify" and the number of null values are above the threshold, the result is False. One approach would be
        to specify nan in the *list_values* by using numpy.nan.
        
        >>> dq.check_column_values_to_be_in_set(column = "ws_source_ip", list_values = ['10.69.147.24 ',np.nan], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column to have values from a specific set',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_list': ['10.69.147.24 ', nan],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'column_values_list': ['10.69.147.24 ', nan],
          'values_not_in_column_list': [],
          'unexpected_values_list': [],
          'unexpected_values_count': 0,
          'unexpected_values_percent': 0},
         'drop_unexpected': False,
         'result': True}
        
        To handle null values efficiently, we can make use of the *include_null* param. If False, then the Null values present in the 
        column are excluded from the evaluation.
        
        >>> dq.check_column_values_to_be_in_set(column = "ws_source_ip", list_values = ['10.69.147.24 '], qlevel = "verify", threshold_percent = 10, include_null = False, drop_unexpected_rows = False)
        {'check_type': 'check column to have values from a specific set',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_list': ['10.69.147.24 '],
         'include_null': False,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'column_values_list': ['10.69.147.24 '],
          'values_not_in_column_list': [],
          'unexpected_values_list': [],
          'unexpected_values_count': 0,
          'unexpected_values_percent': 0},
         'drop_unexpected': False,
         'result': True}
        
        To drop the unexpected values i.e values not present in the specific list of values, we can pass *drop_unexpected_rows* as True. 
         
        **Please note** that when *include_null* is True and *drop_unexpected_rows* is True, it will not only drop the values not present in 
        the specific list of values, but also drop Null values present in the column.
        If *include_null* is False and *drop_unexpected_rows* is True, then it will drop only the values that are not present in the specific 
        list of values. In both the cases, since we drop the unexpected values, the result is changed to True.
         
        To check an exact match of all the list of values present in the column with the expected list of values, pass "assert" as *qlevel*.
        In this case threshold_percent, include_null, drop_unexpected_rows are not functional. All the values in the expected list
        should to be present in the list of unique values in the column and vice versa. *include_null* and *drop_unexpected_rows*
        are saved with their default values, whereas threshold becomes 0.
         
        >>> dq.check_column_values_to_be_in_set(column = "ws_source_ip", list_values = ['10.69.147.24 '], qlevel = "assert", threshold_percent = 10, include_null = False, drop_unexpected_rows = False)
        {'check_type': 'check column to have values from a specific set',
         'column_name': 'ws_source_ip',
         'qualification_level': 'assert',
         'expected_values_list': ['10.69.147.24 '],
         'include_null': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'column_values_list': ['10.69.147.24 ', nan],
          'values_not_in_column_list': [],
          'unexpected_values_list': [nan],
          'unexpected_values_count': 9495,
          'unexpected_values_percent': 94},
         'drop_unexpected': False,
         'result': False}
        """
        if qlevel == "verify":
            if include_null == False:
                # get the list of unique values present in the column
                data_unique_list = list(self.data[self.data[column].notna()][column].unique())
            else:
                # get the list of unique values present in the column
                data_unique_list = list(self.data[column].unique())
                
        elif qlevel == "assert":
            include_null = True
            # get the list of unique values present in the column
            data_unique_list = list(self.data[column].unique())
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
            
        
        # get the list of values which are present in the column other than the expected values
        data_other_vals = list(set(data_unique_list) - set(list_values))
        # get the list of values which are present in the expected list of values and not in the column values
        data_not_in_col_vals = list(set(list_values) - set(data_unique_list))
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total not null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check column to have values from a specific set"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        jsres["expected_values_list"] = list_values
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["include_null"] = include_null
        temp["total_values"] = int(data_total_val)
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["column_values_list"] = data_unique_list
        temp["values_not_in_column_list"] = data_not_in_col_vals
        temp["unexpected_values_list"] = data_other_vals
        data_other_vals_count = self.data[column].isin(data_other_vals).sum()
        temp["unexpected_values_count"] = int(data_other_vals_count)
        temp["unexpected_values_percent"] = int((data_other_vals_count * 100)/data_total_val)
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            # condition to check if the values present in the column are having only the specified set of values
            if len(data_other_vals) == 0 or (data_other_vals_count <= threshold_val):
                # if true return True and the list of values present in the column
                jsres["result"] = True
            else:
                # if false return False and list of values present in the column other than the specified list of values
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                    
                if include_null == True:
                    self.data = self.data[self.data[column].isin(list_values)].reset_index(drop = True)
                    jsres["result"] = True
                elif include_null == False:
                    self.data = self.data[(self.data[column].isna()) | (self.data[column].isin(list_values))].reset_index(drop = True)
                    jsres["result"] = True
                else:
                    raise ValueError("Please enter a bool value for allow_null :" + include_null)
                
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if len(data_other_vals) == 0 and len(data_not_in_col_vals) == 0:
                jsres["result"] = True
            else:
                jsres["result"] = False
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)

        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    
    
    # to check if the values are monotonically increasing or decreasing
    def check_column_values_to_be_monotonic(self, column, increasing = True, qlevel = "verify", strictly_monotonic = False, threshold_percent = 10, drop_unexpected_rows = False):
        """
        Check whether a column has monotonic values (increasing or decreasing).

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented.
        increasing : bool, 
            (default : True)
            If True, checks for monotonically increasing values, otherwise decreasing.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are monotonic or the number of non-monotonic values are below the permissible threshold.
            If "assert", then checks if all values are monotonic.
        strictly_monotonic : bool
            (default : False)
            If True, checks if the values are strictly monotonic.
            If False, checks if the values are monotonic by allowing repetition of the values.
        threshold_percent : int, float 
            (default : 10)
            Used to specify the percentage of values allowed that are not monotonic when the qlevel == "verify".
        drop_unexpected_rows : bool
            (default : False)
            If True, drops the rows where the values are not monotonic when qlevel = "verify".

        Returns
        -------
        jsres : A json format results suite.
        
        Examples
        --------
        To check if column values are monotonically increasing, pass *increasing* as True otherwise False.
        
        >>> dq.check_column_values_to_be_monotonic(column = "time", increasing = True, qlevel = "verify", strictly_monotonic = False, threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'check column values to be monotonic',
         'column_name': 'time',
         'qualification_level': 'verify',
         'monotonic_increasing': True,
         'strictly_monotonic': False,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'monotonic_values': 10000,
          'non_monotonic_values': 0,
          'non_monotonic_percent': 0.0},
         'drop_unexpected': False,
         'result': True}
        
        To check if the values are strictly monotonic, pass *strictly_monotonic* as True. In this case, the values which repeat or duplicated 
        are considered to be non-monotonic.
        
        >>> dq.check_column_values_to_be_monotonic(column = "time", increasing = True, qlevel = "verify", strictly_monotonic = True, threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'check column values to be monotonic',
         'column_name': 'time',
         'qualification_level': 'verify',
         'monotonic_increasing': True,
         'strictly_monotonic': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'monotonic_values': 9999,
          'non_monotonic_values': 1,
          'non_monotonic_percent': 0.0},
         'drop_unexpected': False,
         'result': True}
         
        Based on the result, we observe that 'non_monotonic_values' is 1, which denotes that there is one value which is duplicated.
        
        To drop the unexpected rows, pass *drop_unexpected_rows* as False, which will drop all values where values are non-monotonic.
        
        If *qlevel* is "assert" then params *threshold_percent* and *drop_unexpected_rows* are not functional. *drop_unexpected_rows*
        is saved with its default value, whereas threshold becomes 0.
        
        >>> dq.check_column_values_to_be_monotonic(column = "time", increasing = True, qlevel = "assert", strictly_monotonic = True, threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'check column values to be monotonic',
         'column_name': 'time',
         'qualification_level': 'assert',
         'monotonic_increasing': True,
         'strictly_monotonic': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'monotonic_values': 9999,
          'non_monotonic_values': 1,
          'non_monotonic_percent': 0.0},
         'drop_unexpected': False,
         'result': False}
        """        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check column values to be monotonic"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["monotonic_increasing"] = increasing
        jsres["strictly_monotonic"] = strictly_monotonic
        
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        
        jsres["threshold"] = temp_threshold

        temp_data = self.data[column]
        
        if increasing == True:
            if strictly_monotonic == True:
                # check if the column values are monotonically increasing
                monotonic = [x<y for x, y in zip(temp_data, temp_data[1:])]
            elif strictly_monotonic == False:
                monotonic = [x<=y for x, y in zip(temp_data, temp_data[1:])]
            else:
                raise ValueError("unknown value for strictly_monotonic :" + increasing)
                

        elif increasing == False:
            if strictly_monotonic == True:
                # check if the column values are monotonically decreasing
                monotonic = [x>y for x, y in zip(temp_data, temp_data[1:])]
            elif strictly_monotonic == False:
                monotonic = [x>=y for x, y in zip(temp_data, temp_data[1:])]
            else:
                raise ValueError("unknown value for strictly_monotonic :" + increasing)

        else:
            raise ValueError("unknown value for increasing :" + increasing)
        
        monotonic.append(monotonic[-1])
    
        total_monotonic = sum(monotonic)
        temp["monotonic_values"] = total_monotonic
        non_monotonic_val = data_total_val - total_monotonic
        temp["non_monotonic_values"] = non_monotonic_val
        temp["non_monotonic_percent"] = np.round((non_monotonic_val * 100)/data_total_val,1)
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            
            if (all(monotonic) or (non_monotonic_val <= threshold_val)):
                jsres["result"] = True
            else:
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                self.data = self.data[monotonic].reset_index(drop = True)
                jsres["result"] = True
        
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if all(monotonic):
                jsres["result"] = True
            else:
                jsres["result"] = False
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)  
        
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    
    # to check if the column has only the specified values
    def check_column_value_length_to_be_between(self, column, length_range, qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False):
        """
        Check whether length of the values of a column is in a specific range of length.

        Parameters
        ----------
        column : str
            Name of the column on which the check is implemented.
        length_range : list
            The expected range of the length of the column values.(starting and ending number inclusive)
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are having lengths in the specific range of length or 
            the number of values with lengths beyond the range are below the permissible threshold.
            If "assert", then checks if all values are having lengths in the specific range of length.
        threshold_percent : int, float 
            (default : 10)
            Used to specify the percentage of values allowed with lengths beyond the specific threshold when the qlevel == "verify"
        include_null : bool 
            (default : True)
            If True, will include Null values in the evaluation, otherwise will exclude.
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values with length beyond the specific threshold
            are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        Examples
        --------
        If all the values are inside the expected range of length, the result is True.
        
        >>> dq.check_column_value_length_to_be_between(column = "obm_direction", length_range = [3,5], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check length of column values to be in specific range of length',
         'column_name': 'obm_direction',
         'qualification_level': 'verify',
         'expected_values_len_range': [3,5],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'values_in_range': 10000,
          'values_not_in_range': 0,
          'percent_not_in_range': 0.0},
         'drop_unexpected': False,
         'result': True}
         
        If the column has Null values, these can be excluded by specifying *include_null* as False. Since Null values will be considered to be 
        as NaN which is 3 character length, this param will allow us to proceed with our analysis by not considering them while checking if the 
        values are in the specified range of length. 
         
        For example, below the column has Null values. When we apply the function with *include_null* to be True, it will include null values which
        will be considered to be having values beyond the specified range of length. In the results suite, we observe that the values_not_in_range is equal 
        to null_values. The check has a result False, because *qlevel* is "verify" and the number of values outside the specified range of length
        exceeds the threshold. 
         
        >>> dq.check_column_value_length_to_be_between(column = "obm_direction", length_range = [13,14], qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
         {'check_type': 'check length of column values to be in specific range of length',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_len_range': [13, 14],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_in_range': 505,
          'values_not_in_range': 9495,
          'percent_not_in_range': 95.0},
         'drop_unexpected': False,
         'result': False}
         
        If we pass *include_null* as False, it will not include Null values during the evaluation, although no rows will be dropped from the actual data.
        Here, we observe that the not_null_values is equal to the null_values. It is also observed that the null_values i.e 505 in total, is not included during the 
        evaluation. This is why we have values_not_in_range equal to 0. This means all the non-null values are within the range.
        
        >>> dq.check_column_value_length_to_be_between(column = "obm_direction", length_range = [13,14], qlevel = "verify", threshold_percent = 10, include_null = False, drop_unexpected_rows = False)
        {'check_type': 'check length of column values to be in specific range of length',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_len_range': [13, 14],
         'include_null': False,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_in_range': 505,
          'values_not_in_range': 0,
          'percent_not_in_range': 0.0},
         'drop_unexpected': False,
         'result': True}
         
        To drop the unexpected values i.e values having lengths beyond the expected range of length in this case, we can pass *drop_unexpected_rows* as True. 
         
        **Please note** that when *include_null* is True and *drop_unexpected_rows* is True, it will not only drop the values having lengths out of the specified 
        range of length, but also drop Null values present in the column.
        If *include_null* is False and *drop_unexpected_rows* is True, then it will drop only the values out of the specified range of length. In both the cases,
        since we drop the unexpected values, the result is changed to True.

        In the below scenario, it drops all the Null values present in the column along with values outside the range.
         
        >>> dq.check_column_value_length_to_be_between(column = "obm_direction", length_range = [13,14], qlevel = "verify", threshold_percent = 10, include_null = False, drop_unexpected_rows = True)
        {'check_type': 'check length of column values to be in specific range of length',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_values_len_range': [13, 14],
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_in_range': 505,
          'values_not_in_range': 9495,
          'percent_not_in_range': 95.0},
         'drop_unexpected': True,
         'result': True}
         
        If *qlevel* is "assert", then the params threshold_percent, include_null, drop_unexpected_rows are not functional. If atleast one of the 
        column values is having length beyond the specified range of length, the result is False, irrespective of any condition. *include_null* and *drop_unexpected_rows*
        are saved with there default values, whereas threshold becomes 0.
        
        >>> dq.check_column_value_length_to_be_between(column = "obm_direction", length_range = [13,14], qlevel = "assert", threshold_percent = 10, include_null = False, drop_unexpected_rows = True)
        {'check_type': 'check length of column values to be in specific range of length',
         'column_name': 'obm_direction',
         'qualification_level': 'assert',
         'expected_values_len_range': [13, 14],
         'include_null': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 505,
          'not_null_values': 505,
          'null_values': 0,
          'values_in_range': 0,
          'values_not_in_range': 505,
          'percent_not_in_range': 100.0},
         'drop_unexpected': False,
         'result': False}
        """
        if qlevel == "verify":
            if include_null == False:
                # get the number of values which are in the specific range
                data_in_range_val = len(self.data[(self.data[column].notna()) | (self.data[column].str.len().between(length_range[0],length_range[1]))][column])
                temp_total_val = len(self.data[self.data[column].notna()])
            else:
                # get the number of values which are in the specific range
                data_in_range_val = len(self.data[self.data[column].str.len().between(length_range[0],length_range[1])][column])
                temp_total_val = len(self.data[column])
                
        elif qlevel == "assert":
            include_null = True
            # get the number of values which are in the specific range
            data_in_range_val = len(self.data[self.data[column].str.len().between(length_range[0],length_range[1])][column])
            temp_total_val = len(self.data[column])
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check length of column values to be in specific range of length"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        jsres['expected_values_len_range'] = length_range
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["include_null"] = include_null
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["values_in_range"] = data_in_range_val
        data_no_in_range_vals = temp_total_val - data_in_range_val
        temp["values_not_in_range"] = data_no_in_range_vals
        try :
            temp["percent_not_in_range"] = np.round((data_no_in_range_vals * 100)/data_total_val,1)
        except ZeroDivisionError:
            temp["percent_not_in_range"] = 0
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            if (temp_total_val == data_in_range_val) or (data_no_in_range_vals <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                if include_null == True:
                    self.data = self.data[self.data[column].str.len().between(length_range[0],length_range[1])].reset_index(drop = True)
                    jsres["result"] = True
                elif include_null == False:
                    self.data = self.data[(self.data[column].isna()) | (self.data[column].str.len().between(length_range[0],length_range[1]))].reset_index(drop = True)
                    jsres["result"] = True
                else:
                    raise ValueError("Please enter a bool value for allow_null :" + include_null)
                
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if temp_total_val == data_in_range_val:
                jsres["result"] = True
            else:
                jsres["result"] = False
                
        else:
            raise ValueError("unknown value for qualification level :" + qlevel) 
        
        self.create_results_column_level(jsres)

        return json.dumps(jsres)
    
    # to check if the column values have the specififed sub string
    def check_column_value_to_have_string(self, column, sub_string, qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False):
        """
        Check wether column values have a specific sub string.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        sub_string : str
            The sub-string which has to be in the values of the column.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are having the specified sub-string or 
            the number of values without the sub-string are below the permissible threshold.
            If "assert", then checks if all values are having the specific sub-string.
        threshold_percent : int, float 
            (default : 10)
            Used to specify the percentage of values allowed that do not have the specified 
            sub string when the qlevel == "verify"
        include_null : bool 
            (default : True)
            If True, will include Null values in the evaluation, otherwise will exclude.
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values that do not have the
            sub string are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_value_to_not_have_string : Check whether column values do not have a specific sub string.
        
        Examples
        --------
        If all the values are having the specific sub string, the result is True.
        
        >>> dq.check_column_value_to_have_string(column = "obm_direction", sub_string = "Ta", qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column values to have a specific string',
         'column_name': 'obm_direction',
         'qualification_level': 'verify',
         'expected_value': 'Ta',
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'values_without_sub_string': 0,
          'percent_without_sub_string': 0.0},
         'drop_unexpected': False,
         'result': True}
         
        If the column has Null values, these can be excluded by specifying *include_null* as False. Since Null values will be considered to be 
        as "NaN" when converted string, this param will allow us to proceed with our analysis by not considering them while checking if the 
        values have the specific sub string. 
         
        If we pass *include_null* as False, it will not include Null values during the evaluation, although no rows will be dropped from the actual data.
        
        >>> dq.check_column_value_to_have_string(column = "obm_direction", sub_string = "Ta", qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column values to have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_value': '13',
         'include_null': False,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_without_sub_string': 505,
          'percent_without_sub_string': 5.0},
         'drop_unexpected': False,
         'result': True}
         
        To drop the unexpected values i.e values having lengths beyond the expected range of length in this case, we can pass *drop_unexpected_rows* as True. 
         
        **Please note** that when *include_null* is True and *drop_unexpected_rows* is True, it will not only drop the values not having sub strings 
        but also drop Null values present in the column.
        If *include_null* is False and *drop_unexpected_rows* is True, then it will drop only the values not having sub string. In both the cases,
        since we drop the unexpected values, the result is changed to True.

        In the below scenario, it drops all the Null values present in the column along with values not having sub string.
         
        >>> dq.check_column_value_to_have_string(column = "ws_source_ip", sub_string = "13", qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = True)
        {'check_type': 'check column values to have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_value': '13',
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_without_sub_string': 505,
          'percent_without_sub_string': 5.0},
         'drop_unexpected': True,
         'result': True}
         
        If *qlevel* is "assert", then the params threshold_percent, include_null, drop_unexpected_rows are not functional. If atleast one of the 
        column values is not having the specific sub string, the result is False, irrespective of any condition. *include_null* and *drop_unexpected_rows*
        are saved with there default values, whereas threshold becomes 0.
        
        >>> dq.check_column_value_to_have_string(column = "ws_source_ip", sub_string = "13", qlevel = "assert", threshold_percent = 10, include_null = True, drop_unexpected_rows = True)
        {'check_type': 'check column values to have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'assert',
         'expected_value': '13',
         'include_null': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_without_sub_string': 505,
          'percent_without_sub_string': 5.0},
         'drop_unexpected': False,
         'result': False}
        """
        if qlevel == "verify":
            if include_null == False:
                # get the number of values which do not have the specific sub string
                vals_no_sub_string = len(self.data[(self.data[column].notna()) & (self.data[column].str.find(sub_string) == -1)][column]) 
                data_total_val_in = len(self.data[self.data[column].notna()])
            else:
                # get the number of values which do not have the specific sub string including null
                vals_no_sub_string = len(self.data[(self.data[column].isna()) | (self.data[column].str.find(sub_string) == -1)][column]) 
                data_total_val_in = len(self.data[column])
                
        elif qlevel == "assert":
            include_null = True
            # get the number of values which do not have the specific sub string including null
            vals_no_sub_string = len(self.data[(self.data[column].isna()) | (self.data[column].str.find(sub_string) == -1)][column]) 
            data_total_val_in = len(self.data[column])
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check column values to have a specific string"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        jsres["expected_value"] = sub_string
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["include_null"] = include_null
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["values_without_sub_string"] = vals_no_sub_string
        try:
            temp["percent_without_sub_string"] = np.round((vals_no_sub_string * 100)/data_total_val_in,1)
        except ZeroDivisionError:
            temp["percent_without_sub_string"] = 0
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            if (vals_no_sub_string == 0) or (vals_no_sub_string <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                if include_null == True:
                    self.data = self.data[self.data[column].str.find(sub_string) == 0].reset_index(drop = True)
                    jsres["result"] = True
                elif include_null == False:
                    self.data = self.data[(self.data[column].isna()) | (self.data[column].str.find(sub_string) == 0)].reset_index(drop = True)
                    jsres["result"] = True
                else:
                    raise ValueError("Please enter a bool value for include_null :" + include_null)
        
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if vals_no_sub_string == 0:
                jsres["result"] = True
            else:
                jsres["result"] = False

        else:
            raise ValueError("unknown value for qualification level :" + qlevel) 
        
        self.create_results_column_level(jsres)
            
        return json.dumps(jsres)
    
    
    # to check if the column values do not have the specififed sub string
    def check_column_value_to_not_have_string(self, column, sub_string, qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False):
        """
        Check wether column values do not have a specific sub string.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        sub_string : str
            The sub-string which should not to be in the values of the column.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are not having the specified sub-string or 
            the number of values with the sub-string are below the permissible threshold.
            If "assert", then checks if all values are not having the specific sub-string.
        threshold_percent : int, float 
            (default : 10)
            Used to specify the percentage of values allowed that have the specified 
            sub string when the qlevel == "verify"
        include_null : bool 
            (default : True)
            If True, will include Null values in the evaluation, otherwise will exclude.
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values that have the
            sub string are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_value_to_have_string : Check whether column values have a specific sub string.
        
        Examples
        --------
        If all the values are not having the specific sub string, the result is True.
        
        >>> dq.check_column_value_to_not_have_string(column = "obm_direction", sub_string = "He", qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = False)
        {'check_type': 'check column values to not have a specific string',
         'column_name': 'obm_direction',
         'qualification_level': 'verify',
         'expected_value': 'He',
         'include_null': True,
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'values_with_sub_string': 0,
          'percent_with_sub_string': 0.0},
         'drop_unexpected': False,
         'result': True}
         
        If the column has Null values, these can be excluded by specifying *include_null* as False. Since Null values will be considered to be 
        as "NaN" when converted to string, this param will allow us to proceed with our analysis by not considering them while checking if the 
        values have the specific sub string. 
         
        If we pass *include_null* as False, it will not include Null values during the evaluation, although no rows will be dropped from the actual data.
        Below the check fails because there are only 505 not_null_values, and all them have the specific sub string. And the percent of values
        is above the specific threshold.
        
        >>> dq.check_column_value_to_not_have_string(column = "ws_source_ip", sub_string = "10", qlevel = "verify", threshold_percent = 10, include_null = False, drop_unexpected_rows = False)
        {'check_type': 'check column values to not have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_value': '10',
         'include_null': False,
         'threshold': {'threshold_percent': 2, 'threshold_value': 200},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_with_sub_string': 505,
          'percent_with_sub_string': 5.0},
         'drop_unexpected': False,
         'result': False}

        To drop the unexpected values i.e values having specific sub string in this case, we can pass *drop_unexpected_rows* as True. 
         
        **Please note** that when *include_null* is True and *drop_unexpected_rows* is True, it will not only drop the values having sub strings 
        but also drop Null values present in the column.
        If *include_null* is False and *drop_unexpected_rows* is True, then it will drop only the values having sub string. In both the cases,
        since we drop the unexpected values, the result is changed to True.

        In the below scenario, it drops all the Null values present in the column along with values having the sub string.
         
        >>> dq.check_column_value_to_not_have_string(column = "ws_source_ip", sub_string = "10", qlevel = "verify", threshold_percent = 10, include_null = True, drop_unexpected_rows = True)
        {'check_type': 'check column values to not have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'verify',
         'expected_value': '10',
         'include_null': True,
         'threshold': {'threshold_percent': 2, 'threshold_value': 200},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_with_sub_string': 505,
          'percent_with_sub_string': 5.0},
         'drop_unexpected': True,
         'result': True}
         
        If *qlevel* is "assert", then the params threshold_percent, include_null, drop_unexpected_rows are not functional. If atleast one of the 
        column values is having the specific sub string, the result is False, irrespective of any condition. *include_null* and *drop_unexpected_rows*
        are saved with there default values, whereas threshold becomes 0.
        
        >>> dq.check_column_value_to_not_have_string(column = "ws_source_ip", sub_string = "10", qlevel = "assert", threshold_percent = 10, include_null = True, drop_unexpected_rows = True)
        {'check_type': 'check column values to not have a specific string',
         'column_name': 'ws_source_ip',
         'qualification_level': 'assert',
         'expected_value': '10',
         'include_null': True,
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 505,
          'null_values': 9495,
          'values_with_sub_string': 505,
          'percent_with_sub_string': 5.0},
         'drop_unexpected': False,
         'result': False}
        """
        if qlevel == "verify":
            if include_null == False:
                # get the number of values which do not have the specific sub string without including null values
                vals_sub_string = len(self.data[(self.data[column].notna()) & (self.data[column].str.find(sub_string) != -1)][column]) 
                data_total_val_in = len(self.data[column].notna())
            else:
                # get the number of values which do not have the specific sub string including null values
                vals_sub_string = len(self.data[self.data[column].str.find(sub_string) != -1][column]) 
                data_total_val_in = len(self.data[column])
                
        elif qlevel == "assert":
            include_null = True
            # get the number of values which do not have the specific sub string including null values
            vals_sub_string = len(self.data[self.data[column].str.find(sub_string) != -1][column]) 
            data_total_val_in = len(self.data[column])
        
        else:
            raise ValueError("unknown value for qualification level :" + qlevel)
        
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check column values to not have a specific string"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        jsres["expected_value"] = sub_string
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["include_null"] = include_null
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val) 
        if include_null == True:
            vals_sub_string = int(vals_sub_string - data_null_val)
        temp["values_with_sub_string"] = vals_sub_string
        temp["percent_with_sub_string"] = np.round((vals_sub_string * 100)/data_total_val_in,1)
        
        jsres["threshold"] = temp_threshold
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
        
        if qlevel == "verify":
            if (vals_sub_string == 0) or (vals_sub_string <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                if include_null == True:
                    self.data = self.data[self.data[column].str.find(sub_string) == -1].reset_index(drop = True)
                    jsres["result"] = True
                elif include_null == False:
                    self.data = self.data[(self.data[column].isna()) | (self.data[column].str.find(sub_string) == -1)].reset_index(drop = True)
                    jsres["result"] = True
                else:
                    raise ValueError("Please enter a bool value for include_null :" + include_null)
        
        elif qlevel == "assert":
            jsres["threshold"]['threshold_percent'] = 0
            jsres["threshold"]['threshold_value'] = 0
            jsres["drop_unexpected"] = False
            if vals_sub_string == 0:
                jsres["result"] = True
            else:
                jsres["result"] = False

        else:
            raise ValueError("unknown value for qualification level :" + qlevel) 
        
        self.create_results_column_level(jsres)
            
        return json.dumps(jsres)

    
    # to check if the column maximum value is between a specific range
    def check_column_max_to_be_between(self, column, max_range):
        """
        Check wether a column maximum value is in a specific range.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        max_range : list
            The maximum value is expected to be in this range(minimum and maximum is inclusive). e.g. [10,40]

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_min_to_be_between : Check wether a column minimum value is in a specific range.
        
        Examples
        --------
        
        >>> dq.check_column_max_to_be_between(column = "crssi_dbm", max_range = [-70,-60])
        {'check_type': 'check maximum value of the column to be in specific range',
         'column_name': 'crssi_dbm',
         'expected_max_range': [-70, -60],
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'count_values_max': 7,
          'max_value': -63.0},
         'result': True}
        """
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        jsres = {}
        temp = {}
        jsres["check_type"] = "check maximum value of the column to be in specific range"
        jsres["column_name"] = column
        jsres["expected_max_range"] = max_range
        max_val = self.data[column].max()
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["count_values_max"] = len(self.data[self.data[column] == max_val][column])
        temp["max_value"] = max_val

        jsres["observed_value"] = temp
        
        # get the boolean result by checking if the max value of the column is between the given range
        max_value_bool = self.data[self.data[column] == max_val][column].between(max_range[0],max_range[1]).values[0] 
    
        # condition to check if the column value has max value in the given range
        if max_value_bool:
            jsres["result"] = True
        else:
            jsres["result"] = False
        
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
    
    # method to check if the column maximum value is between a specific range
    def check_column_min_to_be_between(self,column, min_range):
        """
        Check wether a column minimum value is in a specific range.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        min_range : list
            The minimum value is expected to be in this range(minimum and maximum is inclusive). e.g. [10,40]

        Returns
        -------
        jsres : A json format results suite.
        
        See Also
        --------
        check_column_max_to_be_between : Check wether a column maximum value is in a specific range.
        
        Examples
        --------
        
        >>> dq.check_column_min_to_be_between(column = "crssi_dbm", min_range = [-30,-20])
        {'check_type': 'check maximum value of the column to be in specific range',
         'column_name': 'crssi_dbm',
         'expected_min_range': [-30, -20],
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'count_values_min': 1,
          'min_value': -69.0},
         'result': False}
        """
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        jsres = {}
        temp = {}
        jsres["check_type"] = "check maximum value of the column to be in specific range"
        jsres["column_name"] = column
        jsres["expected_min_range"] = min_range
        min_val = self.data[column].min()
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)
        temp["count_values_min"] = len(self.data[self.data[column] == min_val][column])
        temp["min_value"] = min_val

        jsres["observed_value"] = temp
        
        # get the boolean result by checking if the max value of the column is between the given range
        min_value_bool = self.data[self.data[column] == min_val][column].between(min_range[0],min_range[1]).values[0] 
    
        # condition to check if the column value has max value in the given range
        if min_value_bool:
            jsres["result"] = True
        else:
            jsres["result"] = False
        
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)

    
    # method to check if the column maximum value is between a specific range
    def check_column_values_to_match_strftime(self, column, strftime_format, qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False):
        """
        Check column values match strftime format.

        Parameters
        ----------
        column : str
            Name of the column on which the check has to be implemented. 
        strftime_format : strftime format
            The expected strftime format of the column values.
        qlevel : str {"verify","assert"}
            (default : "verify")
            If "verify", then checks if all values are in the specific strftime format or 
            the number of values not in the specific strftime format are below the permissible threshold.
            If "assert", then checks if all values are in the specific strftime format.
        threshold_percent : int, float 
            (default : 10)
            Used to specify the percentage of values allowed that are not in the specific strftime format
            beyond the specific threshold when the qlevel == "verify".
        drop_unexpected_rows : bool
            (default : False)
            When qlevel = "verify" and if *drop_unexpected_rows* set to True: 
            The number of unexpected values, in this case is the number of values not in the specific strftime format
            beyond the specific threshold are dropped. If set to False, will not perform a drop operation.

        Returns
        -------
        jsres : A json format results suite.
        
        Examples
        --------
        To check if the column values are in specific strftime format.
        
        >>> dq.check_column_values_to_match_strftime(column = "time", strftime_format = "%Y-%m-%d %H:%M:%S.%f", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'check column values to match strftime',
         'column_name': 'time',
         'qualification_level': 'verify',
         'expected_strftime_format': '%Y-%m-%d %H:%M:%S.%f',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'expected_format_values': 10000,
          'unexpected_format_values': 0},
         'drop_unexpected': False,
         'result': True}
         
        If some of the values are not in the specific strftime format, when *qlevel* is "verify", the *threshold_percent* 
        is used to check the number of values that are allowed, which are not in the specific strftime format.
        
        Below, since the number of values not in the specific format are below the threshold, the result is True.
        
        >>> dq.check_column_values_to_match_strftime(column = "time", strftime_format = "%Y-%m-%d %H:%M:%S.%f", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = False)
        {'check_type': 'check column values to match strftime',
         'column_name': 'time',
         'qualification_level': 'verify',
         'expected_strftime_format': '%Y-%m-%d %H:%M:%S.%f',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'expected_format_values': 9999,
          'unexpected_format_values': 1},
         'drop_unexpected': False,
         'result': True}
         
        To drop this unexpected value, in this case 1 value, we can use *drop_unexpected_rows* as True.

        >>> dq.check_column_values_to_match_strftime(column = "time", strftime_format = "%Y-%m-%d %H:%M:%S.%f", qlevel = "verify", threshold_percent = 10, drop_unexpected_rows = True)
        {'check_type': 'check column values to match strftime',
         'column_name': 'time',
         'qualification_level': 'verify',
         'expected_strftime_format': '%Y-%m-%d %H:%M:%S.%f',
         'threshold': {'threshold_percent': 10, 'threshold_value': 1000},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'expected_format_values': 9999,
          'unexpected_format_values': 1},
         'drop_unexpected': True,
         'result': True}
         
        If *qlevel* is "assert", then the params threshold_percent and drop_unexpected_rows are not functional. If atleast one of the 
        column values is not in the specific strftime format, the result is False, irrespective of any condition. *drop_unexpected_rows*
        is saved with its default value, whereas threshold becomes 0.
        
        >>> dq.check_column_values_to_match_strftime(column = "time", strftime_format = "%Y-%m-%d %H:%M:%S.%f", qlevel = "assert", threshold_percent = 10, drop_unexpected_rows = True)
        {'check_type': 'check column values to match strftime',
         'column_name': 'time',
         'qualification_level': 'assert',
         'expected_strftime_format': '%Y-%m-%d %H:%M:%S.%f',
         'threshold': {'threshold_percent': 0, 'threshold_value': 0},
         'observed_value': {'total_values': 10000,
          'not_null_values': 10000,
          'null_values': 0,
          'expected_format_values': 9999,
          'unexpected_format_values': 1},
         'drop_unexpected': False,
         'result': False}
        """
        # get total number of values in the column
        data_total_val = len(self.data[column])
        # get total null values in the column
        data_not_null_val = self.data[column].notna().sum()
        # get total null values in the column
        data_null_val = self.data[column].isna().sum()
        
        threshold_val = int((threshold_percent/100) * data_total_val)
        
        jsres = {}
        temp = {}
        temp_threshold = {}
        jsres["check_type"] = "check column values to match strftime"
        jsres["column_name"] = column
        jsres["qualification_level"] = qlevel
        
        temp_threshold['threshold_percent'] = threshold_percent
        temp_threshold['threshold_value'] = threshold_val
        jsres["expected_strftime_format"] = strftime_format
        temp["total_values"] = data_total_val
        temp["not_null_values"] = int(data_not_null_val)
        temp["null_values"] = int(data_null_val)

        jsres["threshold"] = temp_threshold
        
        
        faulty_ind_arr = np.array([])
        temp_time = self.data[column]

        for t in temp_time:
            try:
                datetime.strptime(t, strftime_format).strftime(strftime_format)
            except ValueError:
                faulty_ind_arr = np.append(faulty_ind_arr,temp_time[temp_time == t].index.values)
            except TypeError:
                raise TypeError("Please input only string values")

        faulty_ind_arr = np.unique(faulty_ind_arr.astype(int))
        faulty_vals = len(faulty_ind_arr)
        
        temp["expected_format_values"] = data_total_val - faulty_vals
        temp["unexpected_format_values"] = faulty_vals
        jsres["observed_value"] = temp
        jsres["drop_unexpected"] = drop_unexpected_rows
    
        if qlevel == "verify":
            if (faulty_vals == 0) or (faulty_vals <= threshold_val):
                jsres["result"] = True
            else:
                jsres["result"] = False
            
            if drop_unexpected_rows == True:
                faulty_df = self.data.index.isin(faulty_ind_arr)
                self.data = self.data[~faulty_df]
        
        elif qlevel == "assert":
            temp_threshold['threshold_percent'] = 0
            temp_threshold['threshold_value'] = 0
            jsres['drop_unexpected'] = False
            if faulty_vals == 0:
                jsres["result"] = True
            else:
                jsres["result"] = False

        else:
            raise ValueError("unknown value for qualification level :" + qlevel) 
        
        self.create_results_column_level(jsres)
        
        return json.dumps(jsres)
