---
layout: post
title: Online Order Exploratory Data Analysis
subtitle: 
cover-img: 
thumbnail-img: assets/img/Screenshot (6).png
#share-img:
tags: [regional, shopping, sales, python, other]
---


This post takes a look at the online sales of myriad products products.


This post takes a closer look at the satisfaction of passengers of a certain airline. Passengers in the age range of 7-85 answered questions pertaining to their comfort during the flight and overall satisfaction. For the purposes of this project, I limited the number of samples to 12,200. The airline data examined was made available on Kaggle by teejmahal20. Also, you can view all of the code through a link at the bottom of this post, as well as the Github Repository.

Starting this project required a small bit of data cleaning and wrangling. The only NaN values in the data were in the 'arrival delay in minutes' feature, which could be easily identified to mean that the flight had arrived on time. I simply replaced these values with zeros. Next, I dropped any high cardinality categorical features, and any irrelevant features(ex. Gate Location). The last step in wrangling the data was to encode the target feature in binary.

Luckily, this dataset was relatively free of features that might cause data leakage, and so calculating the permutation importances of the model was all I needed to sift out the features which were negatively impacting the model.
~~~
def wrangle(path,limit=None):
  df = pd.read_csv(path)
  df.fillna(0,inplace=True)
  df.drop(columns=['Unnamed: 0','id','Gate location'],inplace=True)  
  df.columns = ['_'.join(' '.join(x.split('-')).split()) for x in df.columns]
  if limit: 
    return df.head(limit)                                    
  return df
~~~

The feature targeted was satisfaction, with a total of two subcategories(neutral/dissatisfied or satisfied), and the baseline accuracy for the selected data was 57% neutral/dissatisfied(the majorative subcategory). The target was chosen because it was the feature that held information relating to all other features. 

~~~
df['satisfaction'] = df['satisfaction'].apply(lambda b: 1 if b=='satisfied' else 0)
~~~

While exploring this dataset, I fit two predictive models, Logistic Regression and XGBClassifier. For the regression model, my evaluation metric was accuracy score. After executing a couple for loops, I found that tuning the C hyperparameter led to an improved model.

~~~
# pipeline logistic regression model with a tuned C hyperparameter
model_lr = make_pipeline(OrdinalEncoder(),
                         LogisticRegression(C=.5))
~~~
The validation accuracy for the Logistic Regression model was 0.8345286885245902

For the XGBClassifier I used cross-validation score, ROC-AUC score, precision, recall and accuracy score as evaluation metrics; I was able to utilize GridSearchCV to find the optimal combination for n_estimators, learning_rate, min_child_weight, and booster.

~~~
# XGBClassifier pipeline with GridSearchCV-tuned hyperparameters
model_xgb2 = make_pipeline(OrdinalEncoder(),                          
                          XGBClassifier(random_state=42,n_jobs=-1,
                                        learning_rate=0.30000000000000004,
                                        min_child_weight=2))
~~~
The validation accuracy for the XGBClassifier model was 0.9523565573770492

The testing accuracy for the XGBClassifier model was 0.9475409836065574


Here is the ROC-AUC curve for the XGBClassifier:
