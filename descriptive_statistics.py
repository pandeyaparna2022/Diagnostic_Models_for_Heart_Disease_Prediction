# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:47:07 2023

@author: Aparna Pandey, Spyridoula Georgiou
"""
# Load required modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data as dataframe using pandas
df = pd.read_csv("ProcessedClevelandData.csv")

# Print a concise summary of a DataFrame.
# This method prints information about a DataFrame including the index dtype
# and columns, non-null values and memory usage.

df.info()

# Check for missing data in the dataframe for quantitative
Missing_data_count = df.isna().sum()
print("No. of missing data points in different columns:\n",Missing_data_count)

# In case of missing data impute the data based on the most common value in the
# column
for i in df.columns:
    column_index = df.columns.get_loc(i)
    mode = df.mode().iloc[:,column_index]
    print(mode[0])
    df[i] = df[i].fillna(mode[0])

# Assess the summary of aggregate measures for the data
df.describe()

# We need to see if there any outliers in the dataset for continuous variables
# we check that based on the quartiles.

continous_features = ['age','trestbps','chol','thalach','oldpeak']

#Calculate and identify outliers
def detect_outliers(df_out):
    """Returned the number of outliers for continuous variable based on 
    interquartile range.
    """
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        q_1 = np.percentile(feature_data, 25.)
        q_3 = np.percentile(feature_data, 75.)
        iqr = q_3 - q_1
        outlier_step = iqr
        outliers = feature_data[~((feature_data >= q_1 - outlier_step) \
                & (feature_data <= q_3 + outlier_step))].index.tolist()
        print(f"For the feature {each_feature}, the number of outliers is {len(outliers)}.")
detect_outliers(df[continous_features])
###-------------------------------------------------------------------------###
###                     Descriptive Data Analyis                            ###
###-------------------------------------------------------------------------###
#Set the parameters that control the general style of the plots.
sns.set_style('white')
sns.set_palette('twilight')
p1 = sns.countplot(data= df, x='sex',hue='num')
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease', 'Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Gender',fontsize=13)
p1.set_xticklabels(('Female', 'Male'))
sns.despine()
###-------------------------------------------------------------------------###

p2 = sns.countplot(data= df, x='cp',hue='num')
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Chest Pain',fontsize=13)
p2.set_xticklabels(('Typical Angina', 'Atypical Angina','Non - Angina Pain','Asymptomatic'))
sns.despine()
###-------------------------------------------------------------------------###

p3 = sns.countplot(data= df, x='fbs',hue='num',order=[1,0])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Fasting Blood Sugar (>120 mg/dl)',fontsize=13)
p3.set_xticklabels(('True', 'False'))
sns.despine()
###-------------------------------------------------------------------------###

p4 = sns.countplot(data= df, x='restecg',hue='num',order=[1,0,2])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Resting Electrocardiographic Results',fontsize=13)
p4.set_xticklabels(('Normal', 'Abnormal','Ventricular Hypertrophy'))
sns.despine()
###-------------------------------------------------------------------------###

p5 = sns.countplot(data= df, x='exang',hue='num',order=[1,0])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Exercise Induced Angina',fontsize=13)
p5.set_xticklabels(('Yes', 'No'))
sns.despine()
###-------------------------------------------------------------------------###

p6 = sns.countplot(data= df, x='slope',hue='num',order=[3,2,1])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('The Slope Of The Peak Exercise',fontsize=13)
p6.set_xticklabels(('Downsloping', 'Flat','Upsloaping'))
sns.despine()
###-------------------------------------------------------------------------###

p7 = sns.countplot(data= df, x='thal',hue='num',order=[6,3,7])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('Type of Thalassemia',fontsize=13)
p7.set_xticklabels(('Downsloping', 'Flat','Upsloaping'))
sns.despine()
###-------------------------------------------------------------------------###

p8 = sns.countplot(data= df, x='ca',hue='num',order=[3,2,1,0])
plt.legend(title='"Diagnosis', loc='upper left', labels=['No Heart Disease','Heart Disease'])
plt.ylabel('Number of Cases',fontsize=13)
plt.xlabel('The Number Of Major Blood Vessels',fontsize=13)
p8.set_xticklabels(('3', '2','1','0'))
sns.despine()
###-------------------------------------------------------------------------###

sns.histplot(data=df, x="age", kde=True, bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Number of Cases')
sns.despine()
###-------------------------------------------------------------------------###

sns.histplot(data=df, x="chol", kde=True, bins=20)
plt.title('Distribution of Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Number of Cases')
sns.despine()
###-------------------------------------------------------------------------###

sns.histplot(data=df, x="trestbps", kde=True, bins=20)
plt.title('Distribution of Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Number of Cases')
sns.despine()
###-------------------------------------------------------------------------###

sns.histplot(data=df, x="thalach", kde=True, bins=20)
plt.title('Distribution of the Maximum Heart Rate')
plt.xlabel('Maximum Heart Rate')
plt.ylabel('Number of Cases')
sns.despine()
###-------------------------------------------------------------------------###

sns.histplot(data=df, x="oldpeak", kde=True, bins=20)
plt.title('Distribution of the ST Depression Induced by Exercise')
plt.xlabel('ST Depression Induced by Exercise')
plt.ylabel('Number of Cases')
sns.despine()
###-------------------------------------------------------------------------###

p=sns.boxplot(data=df,x='num',y='age')
plt.xlabel('Diagnosis')
plt.ylabel('Age')
plt.xlabel('Diag')
p.set_xticklabels(['No Heart Disease', 'Heart Disease'])
###-------------------------------------------------------------------------###

p=sns.boxplot(data=df,x='num',y='trestbps')
plt.xlabel('Diagnosis')
plt.ylabel('Resting Blood Pressure')
p.set_xticklabels(['No Heart Disease', 'Heart Disease'])
###-------------------------------------------------------------------------###

p=sns.boxplot(data=df,x='num',y='chol')
plt.xlabel('Diagnosis')
plt.ylabel('Cholesterol')
p.set_xticklabels(['No Heart Disease', 'Heart Disease'])
###-------------------------------------------------------------------------###

p=sns.boxplot(data=df,x='num',y='thalach')
plt.xlabel('Diagnosis')
plt.ylabel('Maximum Heart Rate Achieved')
p.set_xticklabels(['No Heart Disease', 'Heart Disease'])
###-------------------------------------------------------------------------###

p=sns.boxplot(data=df,x='num',y='oldpeak')
plt.xlabel('Diagnosis')
plt.ylabel('ST Depression Induced By\n Exercise Relative To Rest')
p.set_xticklabels(['No Heart Disease', 'Heart Disease'])
###-------------------------------------------------------------------------###
# Assess the correlation between variables in the dataset. Correlation between
# variables might affect the regression analysis. So, this information will be
# helpful to explain any unexpected results or for model selection
corrmat = df.corr()
matrix = np.triu(corrmat)
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, square=True, annot=True, cmap="Purples", mask = matrix)
plt.show()
#save the plot
fig.savefig('./Correlation Heatmap', bbox_inches='tight')
###-------------------------------------------------------------------------###



#Pylint
# (base) C:\Users\pandapar\Desktop\Uni_courses_2022\2023\Python\project>pylint python_project_ap_sg_draft_1.py
#
#-------------------------------------------------------------------
#Your code has been rated at 10.00/10 (previous run: 9.92/10, +0.08)
