---
layout: post
title: Neflix Television & Movies
title-color: orange
subtitle: Investigating the bingeworthy
tags: [data analysis, natural language understanding, python]
---


In this report I will characterize the most popular shows and movies on Netflix.
Feel free to make your way over to https://github.com/BrandonSmith710/netflix_nlp_analysis if you would like the full dataset and notebook.

I will begin with some introductory analysis and visualize the ratio of shows to movies.

![image](https://user-images.githubusercontent.com/75755695/176339508-c5438251-698e-4c33-add0-1ce5486223f0.png)

Next, I'll check the distributions of age certifications on Netflix.

![image](https://user-images.githubusercontent.com/75755695/176339260-86cb7e08-7f68-4c3d-97ba-c827da1c3c6d.png)

![image](https://user-images.githubusercontent.com/75755695/176339443-4e0d078d-0911-486d-8168-25019e040824.png)

Take a look at the average IMDB scores for television, and then movies.

![image](https://user-images.githubusercontent.com/75755695/176340219-22639de1-a835-441b-8543-afb0a9e10b00.png)

![image](https://user-images.githubusercontent.com/75755695/176340345-c6c5571f-dc71-4254-83b0-238cef16ed97.png)

An important next step will be to locate the extreme outliers for IMDB score, as this score will be a primary metric in gauging overall success for a movie or show.
In this case, rows with an IMDB score further than three standard deviations from the mean will be considered outlying.
Notably the shows depicted by their respective box plot have a higher average IMDB score than movies.

TV Show Outliers:

![image](https://user-images.githubusercontent.com/75755695/176341099-ea9fc209-8690-4872-bcfb-c4013a622e71.png)

Movie Outliers:

![image](https://user-images.githubusercontent.com/75755695/176341151-a4c4a782-a9e7-4654-81d6-fcc28d0f594b.png)

Now I can remove the noisy titles, as the descriptions of these plotlines will not help with natural language analysis.

~~~
tv_std, tv_mean = df_tv.imdb_score.std(), df_tv.imdb_score.mean()
mv_std, mv_mean = df_mv.imdb_score.std(), df_mv.imdb_score.mean()
df_tv = df_tv.iloc[[x for x in range(len(df_tv)) if df_tv.iloc[x]['imdb_score']
                     >= (tv_mean - (3 * tv_std)) and df_tv.iloc[x]['imdb_score']
                     <= (tv_mean + (3 * tv_std))]]
df_mv = df_mv.iloc[[x for x in range(len(df_mv)) if df_mv.iloc[x]['imdb_score']
                     >= (mv_mean - (3 * mv_std)) and df_mv.iloc[x]['imdb_score']
                     <= (mv_mean + (3 * mv_std))]]
~~~

Popular TV: Drama is the most popular television genre according to our dataset.

![image](https://user-images.githubusercontent.com/75755695/176343327-a2f31c64-a6d9-449a-b2c7-5485c617e09e.png)

Popular Movies: The people love drama, and comedy is not far behind.

![image](https://user-images.githubusercontent.com/75755695/176343368-be795e30-9b5d-4d18-8d67-cf33a4b01fda.png)

In order to visualize the impact of genre, a categorical variable, I've encoded it two ways. The first will support inspection of the genre attribute as a whole, and the second will scrutinize each of the 19 genres.

TV presents a minimal relationship between genre and IMDB score.

![image](https://user-images.githubusercontent.com/75755695/176342414-ad061681-4466-4f76-a800-890fdff9c05b.png)

Movies show virtually no relationship between genre and IMDB score.

![image](https://user-images.githubusercontent.com/75755695/176345486-67c1f00d-a058-44ba-8ed8-5917e012d5dd.png)

Drama has a higher correlation with IMDB score than any other TV genre. Also visible is the relation between age certification and number of seasons, this is another indicator of popularity(Or at least financial success).

![image](https://user-images.githubusercontent.com/75755695/176342770-543cbaa1-3864-497c-84e7-b290b8a643be.png)

Family oriented shows average the most seasons.

![image](https://user-images.githubusercontent.com/75755695/176345627-48e93c87-ea89-4315-80e4-8c13df9cce5c.png)


Documentary movies have the highest correlation with IMDB score, with a coefficient of roughly .3.

![image](https://user-images.githubusercontent.com/75755695/176345854-0aed310c-d6d5-452a-9ae3-23607334d764.png)

What can be learned from the language used to describe each movie or show? Next I'll dissect the descriptions and find out what words and topics are associated with top scores. After fitting a gensim LDA model to the descriptions, both movies and shows can be represented as 15 topics, respectively.


Shows concerning topic 8 have the highest correlation with number of seasons, and several other topics bear a similar relationship.

![image](https://user-images.githubusercontent.com/75755695/176353449-79cf3739-da38-4c0c-828c-56f38cae09a2.png)

Movies focused on topics 8 and 10 have the highest correlations to IMDB score.

![image](https://user-images.githubusercontent.com/75755695/176353499-4959ddd3-205a-4c71-aa29-febea7e1ff7a.png)

Now that topic and genre data is available in numeric a format, it will be easy to comb through other features, like production countries, and see which topics and genres are most popular in each country.

~~~
lst = list()
for countries in df_titles['production_countries']:
    lamb_alf = lambda x: x.isalpha()
    re = [''.join(filter(lamb_alf, c)) for c in countries.split(',')]
    for country in re:
        lst.append(country)
countries = [c for c in list(set(lst)) if c]
countries_dict = {}
for country in countries:
    tv_max_genre, tv_min_genre = '', ''
    mv_max_genre, mv_min_genre = '', ''
    tv_max, tv_min = 0, 0
    mv_max, mv_min = 0, 0
    df_xtv = df_tv[df_tv['production_countries'].apply(
        lambda x: country in x)]
    df_xtv = df_xtv[df_xtv['production_countries'] != '[]']
    df_xmv = df_mv[df_mv['production_countries'].apply(
        lambda x: country in x)]
    df_xmv = df_xmv[df_xmv['production_countries'] != '[]']
    max_top_tv, min_top_tv = df_xtv['topic'].max(), df_xtv['topic'].min()
    max_top_mv, min_top_mv = df_xmv['topic'].max(), df_xtv['topic'].min()
    for genre in df_xtv.columns[14: -3]:
        stv = df_xtv[genre].sum()
        smv = df_xmv[genre].sum()
        if stv > tv_max:
            tv_max, tv_max_genre = stv, genre
        elif stv < tv_min:
            tv_min, tv_min_genre = stv, genre
        if smv > mv_max:
            mv_max, mv_max_genre = smv, genre
        elif smv < mv_min:
            mv_min, mv_min_genre = smv, genre
    countries_dict[country] = [max_top_tv, min_top_tv, tv_max_genre,
                               tv_min_genre, max_top_mv, min_top_mv,
                               mv_max_genre, mv_min_genre]
country_df = pd.DataFrame(countries_dict)
country_df.index = ['max_topic_tv', 'min_topic_tv', 'max_genre_tv',
                    'min_genre_tv', 'max_topic_mv', 'min_topic_mv',
                    'max_genre_mv', 'min_genre_mv']
~~~

The country dataframe holds null values due to the number of topics and genres for a given title occasionally being one or zero. Also, there are numerous countries which only have one title recorded. For these reasons, and due to the defaults of Python, the minimum genres for TV and movies can be dropped, and the remaining null values can be safely ignored.

~~~
country_df.drop('min_genre_tv min_genre_mv'.split(), axis = 0, inplace = True)
~~~

In working with the new data we've acquired, 107 countries can be seen popularizing certain topics and genres in the country dataframe.

![Screenshot (52)](https://user-images.githubusercontent.com/75755695/176356800-a61d84a0-38fa-40d2-80e5-cd9ba8a8e81e.png)
