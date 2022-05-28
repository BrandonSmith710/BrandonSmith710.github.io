---
layout: post
title: Customer Churn Analysis and Prediction
subtitle: Why do customers leave?
cover-img: ![image](https://user-images.githubusercontent.com/75755695/170802669-05f8b628-0223-4149-8147-b5d18f115262.png)
<!-- thumbnail-img: /assets/img/Airline_confusion_matrix.png -->
<!-- share-img: /assets/img/path.jpg -->
tags: [data analysis, machine learning, python]
---
<!-- ![image](https://user-images.githubusercontent.com/75755695/170802669-05f8b628-0223-4149-8147-b5d18f115262.png) -->

Hello, welcome to Customer Churn Analysis. If you'll be following allowing with the code, the csv file needed can be found at https://github.com/BrandonSmith710/customerChurnAnalysis. Click on the text reading, "churn.csv" and then click download. Also,
you'll need to make sure the following requirements are installed in your environment: category_encoders==2.*, pdpbox==0.2.1, imgaug==0.2.5

~~~
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
import io
from google.colab import files
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, f1_score
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('churn.csv', parse_dates = ['joining_date'])
df.drop('Unnamed: 0', axis = 1, inplace = True)
~~~
Explore the data: Visualize the feature distributions
~~~
# first group the numeric features
df_nums = df.drop('churn_risk_score', axis = 1).select_dtypes(include = [int,
                                                                         float]
                  )
plt.figure(figsize = (18, 13))
for i, col in enumerate(df_nums):
    plt.subplot(2, 3, i + 1)
    plt.title(col)
    df_nums[col].plot(kind = 'kde')
~~~

![image](https://user-images.githubusercontent.com/75755695/170802370-5d654c94-a5a2-4912-a288-811d40ec3ca1.png)

~~~
# next isolate the categorical variables
df_cats = df.drop(['referral_id','last_visit_time','avg_frequency_login_days',
                      'security_no','joining_date'], axis = 1
                     ).select_dtypes(include = 'object')
plt.figure(figsize = (20, 24))
for i, col in enumerate(df_cats):
    plt.subplot(4, 3, i + 1)
    ax = sns.countplot(x = df_cats[col])
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 35)
    plt.tight_layout()
plt.show()                    
~~~
![image](https://user-images.githubusercontent.com/75755695/170802427-b0bb7239-200e-4592-aa57-1bad6a183da6.png)

Plot Target Distribution
~~~
pd.Series([16980, 20012]).plot(kind = 'bar')
plt.show()
~~~
![image](https://user-images.githubusercontent.com/75755695/170803014-b56e6cc5-4e2a-4100-a221-6ce072e56dd2.png)

Data Cleaning and Imputation
~~~
# several columns contain invalid values which need to be replaced or removed
df['joined_through_referral'] = pd.Series(
    np.nan if i == '?' else i for i in df['joined_through_referral'])
df['gender'] = df['gender'].replace('Unknown', np.nan)
df['medium_of_operation'] = df['medium_of_operation'].replace('?', np.nan)
df['avg_time_spent'] = df['avg_time_spent'].apply(lambda x: x if x >= 0 else
                                                  np.nan)
df['points_in_wallet'] = df['points_in_wallet'].apply(lambda x: x if x >= 0 else
                                                      np.nan)
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(
    lambda x: x if type(x) == float and x >= 0 else np.nan)
df['days_since_last_login'] = df['days_since_last_login'].apply(
    lambda x: x if x >= 0 else np.nan)

# impute null categorical values with mode of the column
for col in '''joined_through_referral gender medium_of_operation
preferred_offer_types region_category'''.split():
    df[col].fillna(df[col].mode()[0], inplace = True)
m_n = df[['points_in_wallet', 'avg_time_spent', 'days_since_last_login']
                 ]
imputer = KNNImputer(n_neighbors = 3)
imputed_values = imputer.fit_transform(m_n)

dft = pd.DataFrame({
    x: imputed_values.T[i] for i, x in enumerate(missing_num.columns)
})
df['year'] = df['joining_date'].apply(lambda x: x.year)
cols2drop = '''internet_option joining_date complaint_status region_category
               preferred_offer_types medium_of_operation gender
               joined_through_referral offer_application_preference
               used_special_discount past_complaint'''.split()

df.drop(['avg_frequency_login_days','points_in_wallet','days_since_last_login',
         'avg_time_spent'] + cols2drop, axis = 1, inplace = True)

df = pd.concat([df, dft], axis = 1)
df = df.drop('security_no referral_id last_visit_time'.split(), axis = 1)
~~~
Split data and establish baseline accuracy
~~~
target = 'churn_risk_score'
X = df.drop(columns = [target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                    test_size = .19)
baseline = [1] * len(y_test)
print('Baseline accuracy', accuracy_score(y_test, baseline))
~~~
Baseline accuracy 0.5436050647318253
Encode categorical features
~~~
cols2encode = []
for x in df:
    if df[x].dtype == 'object':
        if df[x].nunique() <= 10:
            cols2encode += [x]

ord_enc = OrdinalEncoder(cols = cols2encode)

pipe_rf = make_pipeline(ord_enc,
                        StandardScaler(),
                        RandomForestClassifier(random_state = 42, n_jobs = -1))
~~~
