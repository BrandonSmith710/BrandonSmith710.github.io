---
layout: post
title: Superstore Sales Exploratory Data Analysis
subtitle: 
cover-img: 
thumbnail-img: assets/img/Screenshot (6).png
#share-img:
tags: [regional, shopping, sales, python]
---


The dataset examined in this post was made available on Kaggle.com by user vivek468, and contains 10000 rows of information about orders made to a superstore.

My initial inspection of the sample data found that overall the three most profitable states, in descending order of profitability, are California($76k), New York($74k), and Washington($33k). In considering each year from which the data was provided, these were the most profitable states per year:
  - 2014 New York($13k), California($12k), Washington($6k)
  - 2015 New York($19k), California($14k), Washington($5k)
  - 2016 California($20k), New York($16k), Michigan($9k)
  - 2017 California($29k), New York($24k), Washington($17k)

![Screenshot (6)](https://user-images.githubusercontent.com/75755695/158914522-aefbb521-fd9d-4020-a0be-3afc38503208.png)
![image](https://user-images.githubusercontent.com/75755695/158914344-6dce84b3-75e3-4425-9ad1-629ee6ceacb1.png)
![Screenshot (10)](https://user-images.githubusercontent.com/75755695/158914569-8acd32ed-6c7c-4a89-b211-1e1b2f5ca139.png)

Here are the profits for each cateogory of product from 2014-2017
![Screenshot (16)](https://user-images.githubusercontent.com/75755695/158914792-4f435d36-1d5b-4b7d-ac9c-3c316e5db5a5.png)
![Screenshot (13)](https://user-images.githubusercontent.com/75755695/158914889-34d16aa6-9a8b-4e6d-bda2-176b47041d18.png)


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
