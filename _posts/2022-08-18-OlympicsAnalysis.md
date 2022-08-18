---
layout: post
title: Olympic Games Analysis with Pyspark
title-color: orange
subtitle: Years 1896-2014
tags: [data analysis, big data, pyspark, python]
---

This article will focus on the methods used to answer certain questions about one winter olympic games dataset.
Each question will be answered using the Python and Apache Spark library, Pyspark, which offers its own syntax for querying,
as well as the ability to query using SQL syntax. The data has been loaded into a Pyspark dataframe, and also into a Pyspark view,
allowing me to write each example using both Pyspark and SQL syntax, with the goal of showing the variance between the two.

The data consists of columns, which after transformation, have the following data types:\
 |-- Year: long (integer)\
 |-- City: string\
 |-- Sport: string\
 |-- Discipline: string\
 |-- Athlete: string\
 |-- Country: string\
 |-- Gender: string\
 |-- Event: string\
 |-- Medal: string\ 
 
 1) Which countries won the most gold medals each year?
 ~~~
df_winter.filter(col('Medal') == 'Gold') \
	.groupBy('Year', 'Country') \
    .agg(count('*').alias('Gold_Count')) \
    .select('*', rank().over(Window.partitionBy('Year') \
        .orderBy(desc('Gold_Count'))).alias('rank')) \
    .filter(col('rank') == 1) \
    .orderBy(desc('Year')) \
    .select('Year', 'Country', 'Gold_Count') \
    .show()

spark.sql('''with one as
		 (SELECT Year
	          , Country
	          , COUNT(*) as Gold_Count
	          , rank() OVER( PARTITION BY Year
	   	  		 ORDER BY COUNT(*) DESC ) as rank
	          FROM winter
	          WHERE Medal = "Gold"
	          GROUP BY Year, Country)
	     SELECT Year
	     , Country
	     , Gold_Count
	     FROM one
	     WHERE rank = 1
	     ORDER BY Year DESC;''').show()
 ~~~
