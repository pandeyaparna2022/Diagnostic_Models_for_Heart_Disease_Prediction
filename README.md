# Python code

Thank you for your interest in this project repository that we hope will be useful. This README document is a summary of how we (the coders) are going to analyse the Heart Disease Dataset, which was produced as a result of a study conducted in the late 1980s by the Hungarian Institute of Cardiology (Budapest), the Zurich and Basel University Hospitals (Switzerland), and the Long Beach and Cleveland Clinic Foundation (Long Beach, United States of America). The following points describe the main characteristics of the analysis plan:

# Diagnostic Models for Heart Disease Prediction

The aim of project is to predict whether a person with some particular characteristics, such as their cholesterol and the type of angina they experience, will develop a cardiovascular disease which could possibly be used to design interventions to prevent/lower the risk of cardiovascular disease.

# Installation

Install the Python interpreter correctly on your system.

# Data and file overview
The data is obtained from four databases from Cleveland, Hungary, Switzerland, and the VA Long Beach.

This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them which will be used for our analysis as well.

## Variables Characteristics
1. Age: The person’s age in years
2. Sex: The person’s gender (0 = female, 1 = male)
3. Cp: Chest pain type: 
   Value 1: Typical angina;
   Value 2: Atypical angina;
   Value 3: Non-angina pain;
   Value 4: Asymptomatic;
4. Trestbps: The person’s resting blood pressure (in mm Hg on admission to the hospital)
5. Chol: The person’s cholesterol measured in mg/dl
6. Fbs: The person’s fasting blood sugar (fasting blood sugar > 120 mg/dl; 0 = False, 1 = True)
7. Restecg: The person’s resting electrocardiographic results:
   Value 0: Normal;
   Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV);
   Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
8. Thalach: The maximum heartrate the person can achieve
9. Exang: Exercise induced angina (0 = No, 1 = Yes)
10. Oldpeak: ST depression induced by exercise relative to rest
11. Slope: The slope of the peak exercise ST segment:
 Value 1: Upsloping;
 Value 2: Flat;
 Value 3: Downsloping
12. Ca: The number of major blood vessels (0-3)
13. Thal: Sortcut for a blood disorder called Thalassemia:
 Value 3: Normal;
 Value 6: Fixed defect;
 Value 7: Reversable defect
14. Num: The diagnosis of heart disease:
 Value 0: < 50% diameter narrowing (No heart disease);
 Value 1: > 50% diameter narrowing (Heart disease)

# Analysis and utput
1. Descriptive Statistics of variables considered important for the study

   - Split  between quantitative and qualitative, bar charts, correlation heatmap describing and summarizing various data points and their relation to other data points so that any patterns that exit may become clear.

2. Feature Selection

   - This address the relationship between different factors and heart disease

3. Machine Learning

   - Logistic Regression (to access the probability of the given variable, which in this case was whether the person suffers from some type of cardiovascular disease based on the 13 factors that were studied) 

   - Naïve Bayes (to make quick predictions regarding possibility of suffering cardiovascular disease)



# License

[MIT](https://choosealicense.com/licenses/mit/)
