---
layout: post
title: Superstore Data Visualization and Hypothesis Testing
subtitle: 
cover-img: 
thumbnail-img: assets/img/Screenshot (6).png
#share-img:
tags: [regional, shopping, sales, python]
---


The dataset examined in this post was made available on Kaggle.com by user vivek468, and contains 10000 rows of information for orders made to a superstore.

~~~
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
~~~

~~~
df = pd.read_csv('Sample - Superstore.csv',
     encoding = 'latin1', parse_dates = ['Order Date', 'Ship Date'])
df.columns = ['_'.join(x.split()) for x in df.columns]
df['Order_Year'] = df['Order_Date'].apply(lambda x: x.year)
~~~
Considering the entire timeline of data, the three most profitable states, in descending order of profitability, were California, New York, and Washington.
The color-coded images below shaded the least profitable states with blue, the neutral with grey, and the most profitable with yellow.

![Screenshot (5)](https://user-images.githubusercontent.com/75755695/158944199-f1f03264-6896-4593-a880-881233ea8f06.png)
In considering each year from which the data was provided, these were the most profitable states per year:
  - 2014 New York($13k), California($12k), Washington($6k)
  - 2015 New York($19k), California($14k), Washington($5k)
  - 2016 California($20k), New York($16k), Michigan($9k)
  - 2017 California($29k), New York($24k), Washington($17k)

![image](https://user-images.githubusercontent.com/75755695/158914344-6dce84b3-75e3-4425-9ad1-629ee6ceacb1.png)
![Screenshot (10)](https://user-images.githubusercontent.com/75755695/158914569-8acd32ed-6c7c-4a89-b211-1e1b2f5ca139.png)

Before exploring anymore visualizations, I conducted two product-specific hypothesis tests.

Null Hypothesis 1: There is no association between the city of Fort Lauderdale and the Bretford CR4500 Series Slim Rectangular Table.

Null Hypothesis 2: There is no association between the state of New York and the Bretford CR4500 Series Slim Rectangular Table.
~~~
# engineer features which indicate possession of target characteristics

target_name = 'Bretford CR4500 Series Slim Rectangular Table'
mask = df['Product_Name'] == target_name
mask2 = df['City'] == 'Fort Lauderdale'
mask3 = df['State'] == 'New York'
df.loc[mask, 'Bretford'] = 1
df.loc[~mask, 'Bretford'] = 0
df.loc[mask2, 'Fort_Laud'] = 1
df.loc[~mask2, 'Fort_Laud'] = 0
df.loc[mask3, 'New_York'] = 1
df.loc[~mask3, 'New_York'] = 0

# 12.5% of Bretford Table purchases were in Fort Lauderdale
fl_probabilities = pd.crosstab(df['Bretford'], df['Fort_Laud'], normalize = 'index') * 100

# Also 12.5% of Bretford Table purchases in New York
ny_probabilities = pd.crosstab(df['Bretford'], df['New_York'], normalize = 'index') * 100

# implement the chi-square test
chi, fl_p_val, dof, expected = chi2_contingency(pd.crosstab(df['Bretford'], df['Fort_Laud']))
chi, ny_p_val, dof, expected = chi2_contingency(pd.crosstab(df['Bretford'], df['New_York']))
~~~

The resulting p-value for Fort Lauderdale was 8.252040908989258e-06; at the .05 significance level we can reject Null Hypothesis 1 and conclude that there is an association between Fort Lauderdale and the Bretford Table.

The resulting p-value for New York was 0.6524267307791584; at the .05 significance level we fail to reject Null Hypothesis 2 and conclude that there is no association between New York and the Bretford Table.


This line of code reports the distribution of states amongst purchases of the Bretford Table; the output is in the following cell.
~~~
df[df['Product_Name'] == target_name]['State'].value_counts()
~~~
This output tells us that Florida and Texas were the most popular states with the Bretford Table,
and therefore a statistically significant association between Fort Lauderdale and the Bretford Table is conceivable.
~~~
Florida       2
Texas         2
Utah          1
California    1
New York      1
Idaho         1
Name: State, dtype: int64
~~~

The orange bar charts show company-wide profits by region.


![image](https://user-images.githubusercontent.com/75755695/158943517-dea3f270-05d4-457a-87c9-a89224a2fd6e.png)

In 2016, the best year for central, there were 505 unique products, out of 2359 total products sold.
~~~
df_16 = df[df['Order_Year'] == 2016]
df_c16 = df_16[df_16['Region'] == 'Central']
unique_products = df_c16['Product_Name'].nunique()
total_products = df_c16['Quantity'].sum()
~~~
Let's have a look at the 10 most popular purchases in central during 2016.
~~~
s = list(df_c16['Product_Name'])
s1 = set(s)
si = [str(s.count(x)) for x in s1]
s1 = sorted(zip(s1, si), key= lambda x: x[1], reverse = True)

# print the name and quantity purchased of the most popular items
print('\n'.join(': '.join(x) for x in s1[:10]))
~~~
Output
~~~
    Flat Face Poster Frame: 4
    Staples: 4
    Xerox 212: 3
    Pressboard Covers with Storage Hooks, 9 1/2" x 11", Light Blue: 3
    Staple-based wall hangings: 3
    Westinghouse Clip-On Gooseneck Lamps: 3
    Staple envelope: 3
    Avery Binding System Hidden Tab Executive Style Index Sets: 3
    Storex Dura Pro Binders: 3
    Panasonic KP-380BK Classic Electric Pencil Sharpener: 2
~~~
Below are the respective profits of each product category from 2014-2017. Among the apparent relations in regional profits, the most notable are within central and south technology from 2015-2017, and in east and west technology from 2016-2017. Another observation is that in the west and central regions, customers purchased tech and office supplies at a similar rate, whereas in the east and south regions, tech and office supplies were purchased more inversely.


![Screenshot (16)](https://user-images.githubusercontent.com/75755695/158914792-4f435d36-1d5b-4b7d-ac9c-3c316e5db5a5.png)

![Screenshot (27)](https://user-images.githubusercontent.com/75755695/158942285-60b84b78-ddea-4dca-b698-74bcb658368a.png)


Here is a section of the profits by postal code per product category:

![Screenshot (29)](https://user-images.githubusercontent.com/75755695/158942873-c9bffe70-8803-43a4-a5eb-9eb9d291861e.png)

This boxplot suggests that throughout the four year timeframe of available data, there were three extraordinarily outlying purchases, one 2017 office supplies purchase from the west with a profit of $22,171, one 2017 tech purchase from the west with a profit of $18,984 and one 2017 tech purchase from the east with a profit of $19,301.

![Screenshot (33)](https://user-images.githubusercontent.com/75755695/159067842-9d980a34-48ff-4b52-b613-328766925827.png)


The following code was implemented to pull the 50 best customers.
~~~
# find ID numbers of 50 best customers

a = df['Customer_ID'].unique()

s = sorted(a, key= lambda x: df[df['Customer_ID'] == x]['Sales'].sum(),
           reverse = True)

best_50 = s[:50]

print((2 * '\n').join(' | '.join(best_50[i * 10: i * 10 + 10]) for i in range(5)))
~~~
top 50 customers(first at top left and fiftieth at bottom right)
~~~
SM-20320 | TC-20980 | RB-19360 | TA-21385 | AB-10105 | KL-16645 | SC-20095 | HL-15040 | SE-20110 | CC-12370

TS-21370 | GT-14710 | BM-11140 | SV-20365 | CJ-12010 | CL-12565 | ME-17320 | KF-16285 | BS-11365 | EH-13765

JL-15835 | GT-14635 | HW-14935 | TB-21400 | PF-19120 | CM-12385 | JD-16150 | JE-15715 | LA-16780 | PK-19075

DR-12940 | NF-18385 | KD-16270 | NC-18535 | HM-14860 | KD-16495 | SB-20290 | ZC-21910 | JH-15985 | NP-18700

AH-10690 | AB-10060 | JE-15610 | JW-15220 | LC-16885 | JM-15865 | JD-15895 | PO-18850 | MS-17365 | RW-19540
~~~

Now lets look at the products purchased most frequently by our most frequent customers.

~~~
# find the sub-categories & products most frequently purchased by best_50

# s will hold the sub-categories, and s2 products
s, s2 = set(), set()

for id in best_50:

    tmp = df[df['Customer_ID'] == id]

    u = list(tmp['Sub-Category'].unique())
    u2 = list(tmp['Product_Name'].unique())
    m = max(u, key = lambda x: u.count(x))
    m2 = max(u2, key = lambda x: u2.count(x))
    print(f'{id} Main purchase type: {m} Favorite Item: {m2}')

    s.add(m)
    s2.add(m2)
~~~

When considering the list comprised of, the single most purchased item for each of the 50 best customers, there were a total of 49 unique items, and 14 unique sub-categories(with 17 possible sub-categories).

~~~
plt.figure(figsize = (20, 15))

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
    dfx = pd.Series(dfx_dict.values(), dfx_dict.keys()).plot(kind = 'bar', color = 'red')
    plt.title(product)

~~~

The code above creates nine red barplots, each displays profits for the favorite products of the top nine customers. Only the Hon Olsen Stacker Stools and the Rogers Handheld Barrel Pencil Sharpener returned profits each year. Next, I will take a look at the regional and city-wide profits of the Hon Olsen Stacker Stools to find where the majority and minority of sales come from.

![image](https://user-images.githubusercontent.com/75755695/158950869-f9036c6b-b135-46f1-b183-2b204fbea228.png)


Regional profits for the stacker stools(key is shown right)
![Screenshot (42)](https://user-images.githubusercontent.com/75755695/159199049-d715b810-186d-4174-b360-9abdae5dcdf6.png)

The key and plot below tell us that during 2015 and 2017, the Hon Olsen Stacker Stools product only sold in one city. We can also safely assume that Philadelphia is not a signifcant source of sales for this product.

![Screenshot (38)](https://user-images.githubusercontent.com/75755695/159184268-b150919d-2ebd-4cc6-9def-5a4144cfb6a3.png)
![Screenshot (41)](https://user-images.githubusercontent.com/75755695/159184407-29fc5f76-8195-4c93-a2a4-407ba698eeb9.png)



Throughout this article we took a look at the most popular areas of sale, some of the most frequent customers, and the items that the most frequent customers were buying. I also conducted a couple of statistical tests to ascertain the correlation of certain products with certain locations. With the data brought to light in this presentation, the process of focusing energy and resources into a nationwide business is hopefully quicker and more precise.
