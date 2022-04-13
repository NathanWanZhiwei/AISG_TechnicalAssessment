# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:29:57 2022

@author: Nathan
"""

import pandas as pd
import sqlite3 as sql

def ExtractDataset(datasetURL):
    """Takes in a URL for the dataset and loads it into a panda dataframe and returns it
    
    :param datasetURL : Dataset URL
            
    """
    try:
        connection = sql.connect( datasetURL )
        query = "SELECT * FROM survive"
        #Extract the dataset
        extractedDataset = pd.read_sql_query( query, connection )
        print("Dataset Extracted Successfully\n")
        return extractedDataset
    except:
        print('File Doesnt Exists')
        
def DropDataColumn(df,col_name):
    """Takes in a pandas dataframe and drops the column name given
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
            
    """
    df=df.drop([col_name], axis=1)
    return df;

def DropNullValues(df):
    """Takes in a pandas dataframe and drops all null values
    
    :param df : Dataframe to work on
            
    """
    df=df.dropna()
    return df;

def StandardizeYesNoColumn(df,col_name):
    """Takes in a pandas dataframe and standardizes the given column to 1 or 0
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
    
    Uses map and lambda function to perform standardization
        
    """
    
    def ChangeYesNoValues(val):
        """Inner Function that takes in a string and returns a 1 if "Yes" and 0 if "No"
        
        :param val : Value "Yes" or "No". Ignores letter case
        
        Expected Values: '0' '1' 'No' 'Yes'
        
        """
        
        if val=="1":
            return 1
        elif val=="0":
            return 0
        elif val.upper()=="YES":
            return 1
        else:
            return 0
    
    try:
        df[col_name]=df[col_name].apply( lambda x: ChangeYesNoValues(x) )
        return df;
    except:
        print('Dataset Doesnt Have' + col_name + 'Column')
        
def StandardizeEjectionFractionColumn(df,col_name):
    """Takes in a pandas dataframe and standardizes the given column to 2, 1 or 0
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
    
    Uses map and lambda function to perform standardization
        
    """
    
    def ChangeEjectionFractionValues(val):
        """Inner Function that takes in a string and returns a 2, 1 or 0
        
        :param val : Value to change. Ignores letter case
        
        Expected Values: 'Low' 'Normal' 'High' 'L' 'N'
        
        """
        
        if val.upper( ).startswith("L"):
            return 0
        elif val.upper( ).startswith("N"):
            return 1
        elif val.upper( ).startswith("H"):
            return 2
        else:
            return 0
    
    try:
        df[col_name]=df[col_name].apply( lambda x: ChangeEjectionFractionValues(x) )
        return df;
    except:
        print('Dataset Doesnt Have' + col_name + 'Column')   
        
def RemoveNegativeValuesFromColumn(df,col_name):
    """Takes in a pandas dataframe and removes any values from the given column
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
    
    Uses map and lambda function to perform standardization
        
    """
    df = df[(df[col_name] > 0)]
    return df

def StandardizingDiabetesColumn(df,col_name):
    """Takes in a pandas dataframe and standardizes the given column
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
    
    Uses map and lambda function to perform standardization
        
    """
    
    def ChangeDiabetesValues(val):
        """Inner Function that takes in a string and returns the standardized value
        
        :param val : Value to change. Ignores letter case
        
        Expected Values: ['Normal' 'Pre-diabetes' 'Diabetes']
        
        """
        
        if val.upper( ).startswith("N"):
            return 0
        elif val.upper( ).startswith("P"):
            return 1
        elif val.upper( ).startswith("D"):
            return 2
        else:
            return 0
    
    try:
        df[col_name]=df[col_name].apply( lambda x: ChangeDiabetesValues(x) )
        return df;
    except:
        print('Dataset Doesnt Have' + col_name + 'Column')   
            
def StandardizingGenderColumn(df,col_name):
    """Takes in a pandas dataframe and standardizes the given column
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
    
    Uses map and lambda function to perform standardization
        
    """
    
    def ChangeGenderValues(val):
        """Inner Function that takes in a string and returns the standardized value
        
        :param val : Value to change. Ignores letter case
        
        Expected Values: ['Male' 'Female']
        
        """
        
        if val.upper( ).startswith("M"):
            return 1
        else:
            return 0
    
    try:
        df[col_name]=df[col_name].apply( lambda x: ChangeGenderValues(x) )
        return df;
    except:
        print('Dataset Doesnt Have' + col_name + 'Column')   