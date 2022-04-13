# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:29:57 2022

@author: Nathan
"""

# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


import numpy as np

def PerformLogNormalization(df,col_name):
    """Takes in a dataframe and performs log normalization
    
    :param df : Dataframe to work on
    :param col_name : Column name to work on
            
    """
    df[col_name] = np.log(df[col_name])
    return df

