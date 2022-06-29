---
layout: post
title: Neflix Television & Movies
title-color: orange
subtitle: Why do customers leave?
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

In order to visualize the impact of genre, a categorical variable, I've encoded the feature two ways. The first will support inspection of the genre attribute as a whole, and the second will scrutinize each of the 19 genres.

TV

![image](https://user-images.githubusercontent.com/75755695/176342414-ad061681-4466-4f76-a800-890fdff9c05b.png)

Movies

![image](https://user-images.githubusercontent.com/75755695/176344724-e2a25c49-e65e-4f6e-ad8c-77fd986f7cbe.png)

Both television and movies are showing little to not relationship between genre as a whole and IMDB score.


Age certification appears to have a correlation with number of seasons

![image](https://user-images.githubusercontent.com/75755695/176342770-543cbaa1-3864-497c-84e7-b290b8a643be.png)


