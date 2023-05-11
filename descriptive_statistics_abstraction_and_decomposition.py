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
    df[i] = df[i].fillna(mode[0])

# Assess the summary of aggregate measures for the data
df.describe()

# We need to see if there any outliers in the dataset for continuous variables
# we check that based on the quartiles.

continous_features = ['age','trestbps','chol','thalach','oldpeak']
categorical_features = ['sex','cp','fbs','restecg','exang', 'slope', 'ca', 'thal']

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
###          Descriptive Data Analyis (abstraction and decomposition)      ###
###-------------------------------------------------------------------------###
#Set the parameters that control the general style of the plots.
sns.set_style('white')
sns.set_palette('twilight')

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
# Classify the variables based on their characteristic such as categorical, continuous
# numerical, etc. as different plots are used to visualize different kinds of data
# Also indicate which is the traget/ variable of interest or response variable

continous_features = ['age','trestbps','chol','thalach','oldpeak']
categorical_features = ['sex','cp','fbs','restecg','exang', 'slope', 'ca', 'thal']
dependent_variable = df.iloc[:,-1] #target column

# Make a  copy of the dataframe
df2 = df.copy()

# Change the labels in the copy based on conditions and column descriptions(This
# is optional, just to make the plot labels more intutive). We have done this
# in multiple ways just to show that there are different ways to acheive the same
# result.
# Note: We can use the following code to infer the number and names of the categories
# of a categorical variable if it is not known
# List_Of_Categories_In_Column=list(df['name_of_column'].value_counts().index)
# print(List_Of_Categories_In_Column)
df2['sex'] = np.where(df['sex'] == 0, "Female", "Male")

df2['cp'] = df2['cp'].map({1:'ypical Angina',2:'Atypical Angina',\
                           3:'Non - Angina Pain',4:'Asymptomatic'})

def change_fbs(fasting_blood_sugar):
    """Change the fbs categories to descriptive string literals for more 
    intutive understanding of the plots
    """
    if fasting_blood_sugar == 0:
        return 'False'
    return 'True'
df2['fbs'] = df2['fbs'].apply(change_fbs)

df2['restecg'] = df2['restecg'].map({0:'Normal',1:'Abnormal',2:'Ventricular Hypertrophy'})

df2['exang'] = np.where(df['exang'] == 0, "No", "Yes")

df2['slope'] = df2['slope'].map({1:'Upsloaping',2:'Flat',3:'Downsloping'})

df2['thal'] = df2['thal'].map({3:'Normal',6:'Fixed',7:'Reversed'})

# Write a function to plot different kinds of plots for different knids of
# variables

def plot_descriptive_statistics(df_input,continuous_variables,\
                                categorical_variables,response_variable):
    """A general function for plotting descriptive statistics
    """
    #Set the parameters that control the general style of the plots.
    sns.set_style('white')
    sns.set_palette('twilight')
    for each_feature in df_input.columns:
        index_column = df_input.columns.get_loc(each_feature)
        if each_feature in continuous_variables:
            print(each_feature, "is a continuous variable.")
            # code for plotting histograms
            plt.figure(index_column)
            plot_1=sns.histplot(data=df_input, x=each_feature, kde=True, bins=20)
            plt.title(f"Distribution of {each_feature}")
            plt.xlabel(each_feature)
            plt.ylabel('Number of Cases')
            sns.despine()
            figure = plot_1.get_figure()
            figure.savefig(f"Histogram_{each_feature}.png")
            # code for plotting boxplots
            plt.figure(index_column+len(df_input.columns))
            plot_2=sns.boxplot(data=df_input,x=response_variable,y=each_feature)
            plt.xlabel('Diagnosis')
            plt.ylabel(f"{each_feature}")
            plot_2.set_xticklabels(['No Heart Disease', 'Heart Disease'])
            figure = plot_2.get_figure()
            figure.savefig(f"Box_plot_{each_feature}.png")
        elif each_feature in categorical_variables:
            print(each_feature,"is a continuous variable.")
            # Code for plotting count plots
            plt.figure(index_column)
            plot_1= sns.countplot(data= df_input, x=each_feature,hue=response_variable)
            plt.legend(title='"Diagnosis', labels=['No Heart Disease','Heart Disease'],\
                       bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.ylabel('Number of Cases',fontsize=13)
            plt.xlabel(f"{each_feature}",fontsize=13)
            sns.despine()
            figure = plot_1.get_figure()
            figure.savefig(f"Count_plot_{each_feature}.png")
            #plot_1.set_xticklabels(('Downsloping', 'Flat','Upsloaping'))
# Employ the function to generate and save multiple plots at the same time.
plot_descriptive_statistics(df2,continous_features,categorical_features,dependent_variable)
