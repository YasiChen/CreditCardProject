# this is a kaggle project for predicting the credit card approval/credit card default
# in this project, i will first understand the data, and then do data preparation
# modeling will utilize the machine learning methods
# finally i will talk about the evaluation of the models


import os
import pandas as pd
import numpy as np

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)



# intsall data in from my local env
path=os.listdir('/Users/chenyasi/Documents/PythonPractice/426827_1031720_bundle_archive')

application=pd.read_csv('/Users/chenyasi/Documents/PythonPractice/426827_1031720_bundle_archive/application_record.csv')
credit=pd.read_csv('/Users/chenyasi/Documents/PythonPractice/426827_1031720_bundle_archive/credit_record.csv')


# check the duplicate first
application[application.duplicated()]
credit[credit.duplicated()]
# no duplicate for the application/credit data

# check the primary keys as well
application[application['ID'].duplicated()]
# there's duplication of the customer ID in the application record
duplist=application[application['ID'].duplicated()==True].ID
list(duplist)
len(duplist)
# 47 ID in total, apply for the credit card more than once or there's data entry mistake
# however, since they take up about 0.01%(47/438557) of the sample, i decide to delete the data include in the list
# duplist.dtype
# application['ID'].dtype
application[~application['ID'].isin(duplist)]
# the application data is clear

# join the dataset by ID
# want to preserve the orginal index for the application dataset
df_join=pd.merge(left=application, right=credit, left_on='ID', right_on='ID')





