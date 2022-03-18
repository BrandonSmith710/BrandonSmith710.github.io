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

My initial inspection of the sample data found that overall the three most profitable states, in descending order of profitability, are California, New York, and Washington.

![Screenshot (5)](https://user-images.githubusercontent.com/75755695/158944199-f1f03264-6896-4593-a880-881233ea8f06.png)
In considering each year from which the data was provided, these were the most profitable states per year:
  - 2014 New York($13k), California($12k), Washington($6k)
  - 2015 New York($19k), California($14k), Washington($5k)
  - 2016 California($20k), New York($16k), Michigan($9k)
  - 2017 California($29k), New York($24k), Washington($17k)

![image](https://user-images.githubusercontent.com/75755695/158914344-6dce84b3-75e3-4425-9ad1-629ee6ceacb1.png)
![Screenshot (10)](https://user-images.githubusercontent.com/75755695/158914569-8acd32ed-6c7c-4a89-b211-1e1b2f5ca139.png)


The orange bar charts show profits by region


![image](https://user-images.githubusercontent.com/75755695/158943517-dea3f270-05d4-457a-87c9-a89224a2fd6e.png)


Below are the respective profits of each product category from 2014-2017. Among the apparent relations in regional profits, most notable are within central and south technology from 2015-2017, and in east and west technology profits from 2016-2017.


![Screenshot (16)](https://user-images.githubusercontent.com/75755695/158914792-4f435d36-1d5b-4b7d-ac9c-3c316e5db5a5.png)

![Screenshot (27)](https://user-images.githubusercontent.com/75755695/158942285-60b84b78-ddea-4dca-b698-74bcb658368a.png)


Here are the profits by postal code per product category:

![Screenshot (29)](https://user-images.githubusercontent.com/75755695/158942873-c9bffe70-8803-43a4-a5eb-9eb9d291861e.png)


The following code was implemented to pull the 50 best customers
~~~
# find ID numbers of 50 best customers

a = df['Customer_ID'].unique()

s = sorted(a, key= lambda x: df[df['Customer_ID'] == x]['Sales'].sum(),
           reverse=True)

best_50 = s[:50]

print((2*'\n').join(' | '.join(best_50[i*10:i*10+10]) for i in range(5)))
~~~

~~~
SM-20320 | TC-20980 | RB-19360 | TA-21385 | AB-10105 | KL-16645 | SC-20095 | HL-15040 | SE-20110 | CC-12370

TS-21370 | GT-14710 | BM-11140 | SV-20365 | CJ-12010 | CL-12565 | ME-17320 | KF-16285 | BS-11365 | EH-13765

JL-15835 | GT-14635 | HW-14935 | TB-21400 | PF-19120 | CM-12385 | JD-16150 | JE-15715 | LA-16780 | PK-19075

DR-12940 | NF-18385 | KD-16270 | NC-18535 | HM-14860 | KD-16495 | SB-20290 | ZC-21910 | JH-15985 | NP-18700

AH-10690 | AB-10060 | JE-15610 | JW-15220 | LC-16885 | JM-15865 | JD-15895 | PO-18850 | MS-17365 | RW-19540
~~~

Next, I took a look at the products purchased most frequently by our most frequent customers

~~~
# find the sub-categories & products most frequently purchased by best_50

# s will hold the sub-categories, and s2 products
s, s2 = set(), set()

for id in best_50:

  tmp = df[df['Customer_ID'] == id]

  u = list(tmp['Sub-Category'].unique())
  u2 = list(tmp['Product_Name'].unique())
  m = max(u, key=lambda x: u.count(x))
  m2 = max(u2, key=lambda x: u2.count(x))
  print(f'{id} Main purchase type: {m} Favorite Item: {m2}')

  s.add(m)
  s2.add(m2)
~~~

When considering the list comprised of, the single most purchased item for each of the 50 best customers, there were a total of 49 unique items, and 14 unique sub-categories(with 17 possible sub-categories).

~~~
plt.figure(figsize=(20, 15))

# plot the profits by year for favorite products of top 9 customers

for i in range(1, 10):
  product = s2[i-1]
  dfx = df[df['Product_Name'] == product]
  dfx_dict = {year: 0 for year in years}
  plt.subplot(3, 3, i)

  for year in years:
    dfx_e = dfx[dfx['Order_Year'] == year]
    sales = dfx_e['Profit'].sum()
    dfx_dict[year] = sales
  dfx = pd.Series(dfx_dict.values(), dfx_dict.keys()).plot(kind='bar', color='red')
  plt.title(product)

~~~

The code above creates nine red barplots, each displays profits for the favorite products of the top nine customers.


![image](https://user-images.githubusercontent.com/75755695/158950869-f9036c6b-b135-46f1-b183-2b204fbea228.png)
