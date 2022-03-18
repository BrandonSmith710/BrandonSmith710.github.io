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

My initial inspection of the sample data found that overall the three most profitable states, in descending order of profitability, are California, New York, and Washington. In considering each year from which the data was provided, these were the most profitable states per year:
  - 2014 New York($13k), California($12k), Washington($6k)
  - 2015 New York($19k), California($14k), Washington($5k)
  - 2016 California($20k), New York($16k), Michigan($9k)
  - 2017 California($29k), New York($24k), Washington($17k)

![Screenshot (6)](https://user-images.githubusercontent.com/75755695/158914522-aefbb521-fd9d-4020-a0be-3afc38503208.png)
![image](https://user-images.githubusercontent.com/75755695/158914344-6dce84b3-75e3-4425-9ad1-629ee6ceacb1.png)
![Screenshot (10)](https://user-images.githubusercontent.com/75755695/158914569-8acd32ed-6c7c-4a89-b211-1e1b2f5ca139.png)


Below are the respective profits of each product category from 2014-2017, note similar patterns in zones central and south technology profits from 2015-2017, and also in zones east and west technology profits from 2016-2017


![Screenshot (16)](https://user-images.githubusercontent.com/75755695/158914792-4f435d36-1d5b-4b7d-ac9c-3c316e5db5a5.png)
![Screenshot (27)](https://user-images.githubusercontent.com/75755695/158942285-60b84b78-ddea-4dca-b698-74bcb658368a.png)


Here is a snapshot of profits by product category per postal code:
![Screenshot (26)](https://user-images.githubusercontent.com/75755695/158942196-9cee2e60-ec74-4f39-80ca-c7de9f8051e0.png)

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
