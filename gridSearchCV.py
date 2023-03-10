import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

loan = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
loan = loan.drop(['Loan_ID'],axis=1)
loan.head().style.set_properties(**{'background-color': 'Yellow',
                            'color': 'Blue',
                            'border-color': 'Blue',
                            'font-size' : '15px','font-family': 'Lucida Calligraphy'})
display(loan.head())
        
