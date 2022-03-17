---
layout: post
title: Online Order Exploratory Data Analysis
subtitle: 
cover-img: 
thumbnail-img: assets/img/Screenshot (6).png
#share-img:
tags: [regional, shopping, sales, python]
---


The dataset was made available on Kaggle.com by user vivek468, and contains 10000 rows of information for orders made to a superstore.


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
