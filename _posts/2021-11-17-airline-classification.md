---
layout: post
title: Airline Passenger Classification
subtitle: What makes a flight experience preferable?
cover-img: /assets/img/clouds.jpg
thumbnail-img: /assets/img/Airline_confusion_matrix.png
<!-- share-img: /assets/img/path.jpg -->
tags: [airline, customer classification, predictive modeling, python]
---

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

![ROC-AUC Curve](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de5xVdb3/8ddbLoIX9Ch4CpBAQRMBUUcFTZQ0JDXRn4iK2vFSdEzydLyUZCFRp5toZVmGxkETAS+Z491TiZYKAoqIeAEVZUATsExCBPTz+2OtGTdz22sue48z+/18POYx674+a++Z/dnf73et71cRgZmZla5tWjoAMzNrWU4EZmYlzonAzKzEORGYmZU4JwIzsxLXvqUDaKiuXbtG7969WzoMM7NWZeHChWsjoltt61pdIujduzcLFixo6TDMzFoVSa/Vtc5VQ2ZmJc6JwMysxDkRmJmVOCcCM7MS50RgZlbiCpYIJE2T9JakJXWsl6RrJC2XtFjSAYWKxczM6lbIEsF0YGQ96z8P9Et/xgG/LmAsZmZWh4I9RxARj0rqXc8mo4CbIukHe66knSV9MiLeKFRMVly3zHuduxataukwzNqM/t27cMUX9m3247bkA2U9gJU58xXpshqJQNI4klIDvXr1Kkpw9pHGfqDPe/VtAA7ps0tzh2RmzahVPFkcEVOBqQBlZWUeSSdVrG/cjf1AP6TPLowa3IOxhzh5m32ctWQiWAXsnjPfM11mqXwf9MX6xu0PdLO2rSUTQTkwXtIs4BDgnVJuH6jtQz/fB70/oM2sORQsEUiaCRwJdJVUAVwBdACIiOuA+4BjgeXABuCcQsXycVHfN/zaPvT9QW9mxVDIu4ZOz7M+gAsKdf6W1tBv+P7QN7OW0ioaiz/O6vqW72/4ZtZaOBE0wS3zXudbdz4L1PyW7w99M2stnAgaKLcEUPmt/wcnDfQHvpm1Wk4EGVUmgNwqH3/rN7O2wImgHrV9+/eHv5m1NU4Edahe/+8EYGZtVaZEIGkbYD+gO/AesCQi3ipkYC0pNwm4/t/M2rp6E4GkPYFvAkcDy4A1QCdgL0kbgN8AN0bEh4UOtJgqq4OcBMysFOQrEXyfZJyAr6QPgFWRtBswFjgLuLEw4bWcQ/rs4iRgZiWh3kRQ39PBadXQz5o9ohZ2y7zXmffq2+462cxKRqNHKJP0ueYM5OMgt21g1OAeLRyNmVlxNGWoyt82WxQfE24bMLNSlK+xuLyuVcCuzR9Oy8mtEnISMLNSkq+x+HDgTGB9teUCDi5IRC2ksjTgKiEzKzX5EsFcYENEPFJ9haQXCxNS8bk0YGalLN9dQ5+vZ92w5g+n+NxAbGalrimNxa2enyA2MyvxROC7hMzMSjwRgJ8gNjMr2URQ2UBsZlbqMicCSZPqm29tfLuomVmiISWChXnmWx1XC5mZNSARRMTd9c2bmVnrlK+LiV8AUdf6iLiw2SMqAvcwamb2kXxPFi8oShRF5vYBM7OP5HuyeKsBZyRtFxEbChtScbh9wMwskamNQNJQSUuBF9L5/ST9qqCRmZlZUWRtLP4ZcAywDiAingHaRF9DZmalriF3Da2stuiDZo6lKPwgmZnZ1vI1FldaKelQICR1AP4LeL5wYRWOG4rNzLaWtUTwn8AFQA9gNTA4nW9VPO6AmVlNmRJBRKyNiDMi4t8joltEnBkR6/LtJ2mkpBclLZd0WS3re0l6WNLTkhZLOrYxF5GVSwNmZjVlvWtoD0l3S1oj6S1Jd0naI88+7YBrgc8D/YHTJfWvttm3gVsjYn/gNKDgdyK5NGBmtrWsVUO3ALcCnwS6A7cBM/PsczCwPCJeiYhNwCxgVLVtAuiSTu9EUu1kZmZFlDURbBcRv4uILenPzUCnPPv0AHLvNKpIl+WaBJwpqQK4D/habQeSNE7SAkkL1qxZkzFkMzPLot5EIGkXSbsA90u6TFJvSZ+S9A2SD+6mOh2YHhE9gWOB30mqEVNETI2Isogo69atWzOc1szMKuW7fXQhSfWN0vmv5KwLYEI9+64Cds+Z75kuy3UeMBIgIp6Q1AnoCryVJy4zM2sm+foa6tOEY88H+knqQ5IATgPGVtvmdeAoYLqkfUiqm1z3Y2ZWRFkfKEPSAJK7f6raBiLiprq2j4gtksYDDwLtgGkR8ZykycCCiCgHLgaul/TfJCWMsyOizm6vzcys+WVKBJKuAI4kSQT3kdwS+legzkQAEBH3Ua0tISIm5kwvBQ5rUMRmZtasst41NJqkCufNiDgH2I/kdk8zM2vlsiaC9yLiQ2CLpC4kjbm759nHzMxagaxtBAsk7QxcT3In0XrgiYJFZWZmRZMpEUTEV9PJ6yQ9AHSJiMWFC8vMzIol3+D1B9S3LiKeav6QzMysmPKVCK6qZ10An23GWMzMrAXke6BseLECMTOzlpF5qEozM2ubnAjMzEqcE4GZWYnLOkKZJJ0paWI630vSwYUNzczMiiFrieBXwFCS8QMA3iUZhtLMzFq5rE8WHxIRB0h6GiAi/i6pYwHjMjOzIslaIticDkYfAJK6AR8WLCozMyuarIngGuBOYDdJ/0PSBfUPChaVmZkVTda+hmZIWkjSFbWAEyPi+YJGZmZmRZF1YJprgFkR4QZiM7M2JmvV0ELg25JeljRFUlkhgzIzs+LJlAgi4saIOBY4CHgR+LGkZQWNzMzMiqKhTxb3BT4NfAp4ofnDMTOzYsv6ZPFP0hLAZGAJUBYRXyhoZGZmVhRZHyh7GRgaEWsLGYyZmRVfvhHKPh0RLwDzgV6SeuWu9whlZmatX74SwUXAOGofqcwjlJmZtQH5Rigbl05+PiI25q6T1KlgUZmZWdFkvWvo8YzLzMyslcnXRvAJoAfQWdL+JN1LAHQBtitwbGZmVgT52giOAc4GegJX5yx/F/hWgWIyM7MiytdGcCNwo6STI+KOIsVkZmZFlK9q6MyIuBnoLemi6usj4upadjMzs1YkX2Px9unvHYAda/mpl6SRkl6UtFzSZXVsM0bSUknPSbqlAbGbmVkzyFc19Jv093cbeuB0RLNrgc8BFcB8SeURsTRnm37ABOCwdPjL3Rp6HjMza5qG9DXURVIHSX+StEbSmXl2OxhYHhGvRMQmYBYwqto2XwaujYi/A0TEWw29ADMza5qszxGMiIh/AscDK0h6Ib00zz49gJU58xXpslx7AXtJekzSXEkjazuQpHGSFkhasGbNmowhm5lZFlkTQWUV0nHAbRHxTjOdvz3QDzgSOB24XtLO1TeKiKkRURYRZd26dWumU5uZGWRPBPdIegE4EPiTpG7Axjz7rAJ2z5nvmS7LVQGUR8TmiHgVeIkkMZiZWZFkHaHsMuBQknEINgP/omZ9f3XzgX6S+kjqCJwGlFfb5g8kpQEkdSWpKnolc/RmZtZkWQev7wCcCQyTBPAIcF19+0TEFknjgQeBdsC0iHhO0mRgQUSUp+tGSFoKfABcGhHrGn01ZmbWYFkHpvk10AH4VTp/VrrsS/XtFBH3AfdVWzYxZzpIurqu8bCamZkVR9ZEcFBE7Jcz/2dJzxQiIDMzK66sjcUfSNqzckbSHiRVOWZm1splLRFcCjws6RWSrqg/BZxTsKjMzKxo8iaC9FbRd0ieFK7sAuLFiHi/kIGZmVlx1Fs1JOlLwHPAL4BFQO+IWOwkYGbWduQrEXwd2Dci1qTtAjOo+SyAmZm1YvkaizdFxBqAiHgF2LbwIRXGLfNeZ96rb7d0GGZmHzv5SgQ9JV1T13xEXFiYsJrfXYuS3i1GDa7e752ZWWnLlwiq9zC6sFCBFMMhfXZh7CG9WjoMM7OPlSxjFpuZWRuW766h6yUNqGPd9pLOlXRGYUIzM7NiyFc1dC0wUdJAYAmwBuhE0lV0F2AayZ1EZmbWSuWrGloEjJG0A1AGfBJ4D3g+Il4sQnxmZlZgmbqYiIj1wJzChmJmZi0ha6dzZmbWRjkRmJmVuAYlAknbFSoQMzNrGZkSgaRD0+EkX0jn95P0qzy7mZlZK5C1RPBT4BhgHUBEPAMMK1RQZmZWPJmrhiJiZbVFHqHMzKwNyDpC2UpJhwIhqQPwX8DzhQvLzMyKJWuJ4D+BC4AewCpgMPDVQgVlZmbFk7VEsHdEbNWnkKTDgMeaPyQzMyumrCWCX2RcZmZmrUy9JQJJQ4FDgW6SLspZ1QVoV8jAzMysOPJVDXUEdki32zFn+T+B0YUKyszMiidf76OPAI9Imh4RrxUpJjMzK6KsjcUbJF0J7EsyHgEAEfHZgkRlZmZFk7WxeAZJ9xJ9gO8CK4D5BYrJzMyKKGsi2DUifgtsjohHIuJcwKUBM7M2IGvV0Ob09xuSjgNWA7sUJiQzMyumrCWC70vaCbgYuAS4Afh6vp0kjZT0oqTlki6rZ7uTJYWksozxmJlZM8k6VOU96eQ7wHCoerK4TpLaAdcCnwMqgPmSyiNiabXtdiTpu2hew0I3M7PmUG+JQFI7SadLukTSgHTZ8ZIeB36Z59gHA8sj4pWI2ATMAkbVst33gB8DGxsevpmZNVW+qqHfAl8CdgWukXQzMAX4SUTsn2ffHkBu19UV6bIqkg4Ado+Ie+s7kKRxkhZIWrBmzZo8pzUzs4bIVzVUBgyKiA8ldQLeBPaMiHVNPbGkbYCrgbPzbRsRU4GpAGVlZdHUc5uZ2UfylQg2RcSHABGxEXilAUlgFbB7znzPdFmlHYEBwBxJK4AhQLkbjM3MiitfieDTkhan0wL2TOcFREQMqmff+UA/SX1IEsBpwNjKlRHxDtC1cl7SHOCSiFjQ4KswM7NGy5cI9mnsgSNii6TxwIMkPZVOi4jnJE0GFkREeWOPbWZmzSdfp3NN6mguIu4D7qu2bGId2x7ZlHOZmVnjZB683szM2iYnAjOzEpc5EUjqLGnvQgZjZmbFlykRSPoCsAh4IJ0fLMmNvWZmbUDWEsEkki4j/gEQEYtIxiYwM7NWLmsi2Jze95/LT/iambUBWccjeE7SWKCdpH7AhcDjhQvLzMyKJWuJ4Gsk4xW/D9xC0h113vEIzMzs4y9rieDTEXE5cHkhgzEzs+LLWiK4StLzkr5XOS6BmZm1DZkSQUQMJxmZbA3wG0nPSvp2QSMzM7OiyPxAWUS8GRHXAP9J8kxBrX0GmZlZ65L1gbJ9JE2S9CzwC5I7hnoWNDIzMyuKrI3F04DZwDERsbqA8ZiZWZFlSgQRMbTQgZiZWcuoNxFIujUixqRVQrlPEmcZoczMzFqBfCWC/0p/H1/oQMzMrGXU21gcEW+kk1+NiNdyf4CvFj48MzMrtKy3j36ulmWfb85AzMysZeRrIzif5Jv/HpIW56zaEXiskIGZmVlx5GsjuAW4H/ghcFnO8ncj4u2CRWVmZkWTLxFERKyQdEH1FZJ2cTIwM2v9spQIjgcWktw+qpx1AexRoLjMzKxI6k0EEXF8+tvDUpqZtVFZ+xo6TNL26fSZkq6W1KuwoZmZWTFkvX3018AGSfsBFwMvA78rWFRmZlY0WRPBlogIYBTwy4i4luQWUjMza+Wy9j76rqQJwFnA4ZK2AToULiwzMyuWrCWCU0kGrj83It4kGYvgyoJFZWZmRZN1qMo3gRnATpKOBzZGxE0FjczMzIoi611DY4AngVOAMcA8SaMz7DdS0ouSlku6rJb1F0laKmmxpD9J+lRDL8DMzJomaxvB5cBBEfEWgKRuwB+B2+vaQVI74FqSDusqgPmSyiNiac5mTwNlEbEh7dfoJyTVUGZmViRZ2wi2qUwCqXUZ9j0YWB4Rr0TEJmAWyV1HVSLi4YjYkM7OxeMgm5kVXdYSwQOSHgRmpvOnAvfl2acHsDJnvgI4pJ7tzyPp4K4GSeOAcQC9evk5NjOz5pR1zOJLJf0/4DPpoqkRcWdzBSHpTKAMOKKO808FpgKUlZVFbduYmVnj5BuPoB8wBdgTeBa4JCJWZTz2KmD3nPme6bLq5ziapA3iiIh4P+OxzcysmeSr558G3AOcTNID6S8acOz5QD9JfSR1BE4DynM3kLQ/8BvghGptEGZmViT5qoZ2jIjr0+kXJT2V9cARsUXSeOBBoB0wLSKekzQZWBAR5SQPpe0A3CYJ4PWIOKHBV2FmZo2WLxF0Sr+1V45D0Dl3PiLqTQwRcR/VGpUjYmLO9NENjtjMzJpVvkTwBnB1zvybOfMBfLYQQZmZWfHkG5hmeLECMTOzlpH1gTIzM2ujnAjMzEqcE4GZWYnL2vuo0rGKJ6bzvSQdXNjQzMysGLKWCH4FDAVOT+ffJelZ1MzMWrmsnc4dEhEHSHoaICL+nj4tbGZmrVzWEsHmdHyBgKrxCD4sWFRmZlY0WRPBNcCdwG6S/gf4K/CDgkVlZmZFk7Ub6hmSFgJHkXQvcWJEPF/QyMzMrCgyJQJJvYANwN25yyLi9UIFZmZmxZG1sfhekvYBAZ2APsCLwL4FisvMzIoka9XQwNx5SQcAXy1IRGZmVlSNerI47X66vvGHzcyslcjaRnBRzuw2wAHA6oJEZGZmRZW1jWDHnOktJG0GdzR/OGZmVmx5E0H6INmOEXFJEeIxM7Miq7eNQFL7iPgAOKxI8ZiZWZHlKxE8SdIesEhSOXAb8K/KlRHx+wLGZmZmRZC1jaATsI5kjOLK5wkCcCIwM2vl8iWC3dI7hpbwUQKoFAWLyqzEbd68mYqKCjZu3NjSoVgr06lTJ3r27EmHDh0y75MvEbQDdmDrBFDJicCsQCoqKthxxx3p3bs3Um3/fmY1RQTr1q2joqKCPn36ZN4vXyJ4IyImNy00M2uojRs3OglYg0li1113Zc2aNQ3aL9+Txf4rNGshTgLWGI35u8mXCI5qXChmZtZa1JsIIuLtYgViZh8v7dq1Y/DgwQwYMIBTTjmFDRs2sGDBAi688MJGH3OHHXYAYPXq1YwePbq5QuXrX/86jz76aNX82rVr6dChA9ddd12t5680ffp0xo8fXzV/0003MWDAAAYOHMj+++/PlClTmhzbAw88wN57703fvn350Y9+VOs2r732GkcddRSDBg3iyCOPpKKiomrdN7/5TQYMGMCAAQOYPXt21fLTTjuNZcuWNTk+aGSnc2bW9nXu3JlFixaxZMkSOnbsyHXXXUdZWRnXXHNNk4/dvXt3br/99maIEtatW8fcuXMZNmxY1bLbbruNIUOGMHPmzMzHuf/++/nZz37GQw89xLPPPsvcuXPZaaedmhTbBx98wAUXXMD999/P0qVLmTlzJkuXLq2x3SWXXMIXv/hFFi9ezMSJE5kwYQIA9957L0899RSLFi1i3rx5TJkyhX/+858AnH/++fzkJz9pUnyVsj5HYGYt5Lt3P8fS1f9s1mP2796FK76QfTiRww8/nMWLFzNnzhymTJnCPffcw6RJk3j55ZdZvnw5a9eu5Rvf+AZf/vKXAbjyyiu59dZbef/99znppJP47ne/u9XxVqxYwfHHH8+SJUuYPn065eXlbNiwgZdffpmTTjqp6gPuoYce4oorruD9999nzz335H//939rfKu/4447GDly5FbLZs6cyVVXXcXYsWOpqKigZ8+eea/xhz/8IVOmTKF79+4AbLvttlXX01hPPvkkffv2ZY899gCSb/F33XUX/fv332q7pUuXcvXVVwMwfPhwTjzxxKrlw4YNo3379rRv355BgwbxwAMPMGbMGA4//HDOPvtstmzZQvv2Tfsod4nAzOq1ZcsW7r//fgYOHFhj3eLFi/nzn//ME088weTJk1m9ejUPPfQQy5Yt48knn2TRokUsXLhwq2qb2ixatIjZs2fz7LPPMnv2bFauXMnatWv5/ve/zx//+EeeeuopysrKqj4scz322GMceOCBVfMrV67kjTfe4OCDD2bMmDFbVafUZ8mSJVsdpy4zZsxg8ODBNX5qq+patWoVu+++e9V8z549WbVqVY3t9ttvP37/++T53DvvvJN3332XdevWsd9++/HAAw+wYcMG1q5dy8MPP8zKlSsB2Gabbejbty/PPPNMpuurj0sEZh9zDfnm3pzee+89Bg8eDCQlgvPOO4/HH398q21GjRpF586d6dy5M8OHD+fJJ5/kr3/9Kw899BD7778/AOvXr2fZsmVbVd1Ud9RRR1VVw/Tv35/XXnuNf/zjHyxdupTDDku6Otu0aRNDhw6tse8bb7xBt27dquZnz57NmDFjgOQb+LnnnsvFF19c57kbepfNGWecwRlnnNGgffKZMmUK48ePZ/r06QwbNowePXrQrl07RowYwfz58zn00EPp1q0bQ4cOpV27dlX77bbbbqxevTpTAqtPQROBpJHAz0keTLshIn5Ubf22wE3AgSRdWJwaESsKGZOZZVPZRlCf6h+ikogIJkyYwFe+8pXM59p2222rptu1a8eWLVuICD73uc/lrefv3LnzVk9gz5w5kzfffJMZM2YAScP0smXL6NevH507d2bTpk107NgRgLfffpuuXbsCsO+++7Jw4UI++9nP1nu+GTNmcOWVV9ZY3rdv3xrtHj169Kj6Bg/Jg4I9evSosW/37t2rSgTr16/njjvuYOeddwbg8ssv5/LLLwdg7Nix7LXXXlX7bdy4kc6dO9cbbxYFqxpKu6++Fvg80B84XVL/apudB/w9IvoCPwV+XKh4zKz53XXXXWzcuJF169YxZ84cDjroII455himTZvG+vXrgaR65K233mrwsYcMGcJjjz3G8uXLAfjXv/7FSy+9VGO7ffbZp2qbl156ifXr17Nq1SpWrFjBihUrmDBhQlUyOeKII7j55puBpMRz6623Mnz4cAAmTJjApZdeyptvvgkkJZAbbrihxvnOOOMMFi1aVOOntsbvgw46iGXLlvHqq6+yadMmZs2axQknnFBju7Vr1/Lhhx8CSVvFueeeCySNzevWrQOSarjFixczYsSIqv1eeuklBgwYkOXlrFch2wgOBpZHxCsRsQmYBYyqts0o4MZ0+nbgKPkpGrNWY9CgQQwfPpwhQ4bwne98h+7duzNixAjGjh3L0KFDGThwIKNHj+bdd99t8LG7devG9OnTOf300xk0aBBDhw7lhRdeqLHdcccdx5w5c4CkNHDSSSdttf7kk0+uSgQ///nP+f3vf8/gwYMZMmQIp5xySlWV1bHHHsv48eM5+uij2XfffTnggAOq7tBprPbt2/PLX/6SY445hn322YcxY8aw775JVd/EiRMpLy8HYM6cOey9997stdde/O1vf6sqAWzevJnDDz+c/v37M27cOG6++eaqhuG//e1vdO7cmU984hNNihFAEYXpMkjSaGBkRHwpnT8LOCQixudssyTdpiKdfzndZm21Y40DxgH06tXrwNdee63B8Xz37ueAlqtvNWuI559/nn322aelw6jXpEmT2GGHHbjkkpYfs+ozn/kM99xzT1V1Sin46U9/SpcuXTjvvPNqrKvt70fSwogoq+1YraKxOCKmAlMBysrKGpW5nADM2q6rrrqK119/vaQSwc4778xZZ53VLMcqZCJYBeyeM98zXVbbNhWS2gM7kTQam9nH3KRJk1o6hCqHHHJIS4dQdOecc06zHauQbQTzgX6S+kjqCJwGlFfbphz4j3R6NPDnKFRdlVkr438Fa4zG/N0ULBFExBZgPPAg8Dxwa0Q8J2mypMpm898Cu0paDlwEXFaoeMxak06dOrFu3TonA2uQyvEIOnXq1KD9CtZYXChlZWWxYMGClg7DrKA8Qpk1Vl0jlLX6xmKzUtOhQ4cGjTBl1hTua8jMrMQ5EZiZlTgnAjOzEtfqGoslrQEa/mhxoiuwNu9WbYuvuTT4mktDU675UxHRrbYVrS4RNIWkBXW1mrdVvubS4GsuDYW6ZlcNmZmVOCcCM7MSV2qJYGpLB9ACfM2lwddcGgpyzSXVRmBmZjWVWonAzMyqcSIwMytxbTIRSBop6UVJyyXV6NFU0raSZqfr50nqXfwom1eGa75I0lJJiyX9SdKnWiLO5pTvmnO2O1lSSGr1txpmuWZJY9L3+jlJtxQ7xuaW4W+7l6SHJT2d/n0f2xJxNhdJ0yS9lY7gWNt6SbomfT0WSzqgySeNiDb1A7QDXgb2ADoCzwD9q23zVeC6dPo0YHZLx12Eax4ObJdOn18K15xutyPwKDAXKGvpuIvwPvcDngb+LZ3fraXjLsI1TwXOT6f7AytaOu4mXvMw4ABgSR3rjwXuBwQMAeY19ZxtsURwMLA8Il6JiE3ALGBUtW1GATem07cDR0lSEWNsbnmvOSIejogN6exckhHjWrMs7zPA94AfA22hP+cs1/xl4NqI+DtARLxV5BibW5ZrDqBLOr0TsLqI8TW7iHgUeLueTUYBN0ViLrCzpE825ZxtMRH0AFbmzFeky2rdJpIBdN4Bdi1KdIWR5ZpznUfyjaI1y3vNaZF594i4t5iBFVCW93kvYC9Jj0maK2lk0aIrjCzXPAk4U1IFcB/wteKE1mIa+v+el8cjKDGSzgTKgCNaOpZCkrQNcDVwdguHUmztSaqHjiQp9T0qaWBE/KNFoyqs04HpEXGVpKHA7yQNiIgPWzqw1qItlghWAbvnzPdMl9W6jaT2JMXJdUWJrjCyXDOSjgYuB06IiPeLFFuh5LvmHYEBwBxJK0jqUstbeYNxlve5AiiPiM0R8SrwEkliaK2yXPN5wK0AEfEE0Imkc7a2KtP/e0O0xUQwH+gnqY+kjiSNweXVtikH/iOdHg38OdJWmFYq7zVL2h/4DUkSaO31xpDnmiPinYjoGhG9I6I3SbvICRHRmsc5zfK3/QeS0gCSupJUFb1SzCCbWZZrfh04CkDSPiSJYE1RoyyucuCL6d1DQ4B3IuKNphywzVUNRcQWSeOBB0nuOJgWEc9JmgwsiIhy4LckxcflJI0yp7VcxE2X8ZqvBHYAbkvbxV+PiBNaLOgmynjNbUrGa34QGCFpKfABcGlEtNrSbsZrvhi4XtJ/kzQcn92av9hJmkmSzLum7R5XAB0AIuI6knaQY4HlwAbgnCafsxW/XmZm1gzaYtWQmZk1gBOBmVmJcyIwMytxTgRmZiXOicDMrMQ5EZQASR9IWpTz07uebdc3w/mmS3o1PddT6dOeDT3GDZL6p9Pfqrbu8abGmHVY9BwAAAaASURBVB6n8nVZIuluSTvn2X5wY3q2lPRJSfek00dKeic97/OSrmjE8U6o7IVT0omVr1M6Pzl9cLBJ0vdwdJ5t5jTkAb302u/JsF2tvW9KmiLps1nPZ9k5EZSG9yJicM7PiiKc89KIGAxcRvIgW4NExJciYmk6+61q6w5thvjgo9dlAMnzJBfk2X4wyf3bDXURcH3O/F/S16aMpI+cBnUjHBHlEfGjdPZEkh43K9dNjIg/NiLGj5PpQG19JP2C5O/JmpkTQQmStIOSMQmekvSspBq9dqbfYh/N+cZ8eLp8hKQn0n1vk7RDntM9CvRN970oPdYSSV9Pl20v6V5Jz6TLT02Xz5FUJulHQOc0jhnpuvXp71mSjsuJebqk0ZLaSbpS0nwl/bV/JcPL8gRpx12SDk6v8WlJj0vaO32qdTJwahrLqWns0yQ9mW5bW++nACcDD1RfGBH/AhYCfdPSxtw03jsl/Vsay4X6aByJWemysyX9UtKhwAnAlWlMe+a8BiMl3Zbz2lR9G2/oeyhpYvpaLpE0Vdqqp96zcv5GDk63z/q61Kqu3jcj4jVgV0mfaMjxLIOW6G/bP8X9IXnCdFH6cyfJE+Vd0nVdSZ5QrHy4cH36+2Lg8nS6HUnfPV1JPti3T5d/E5hYy/mmA6PT6VOAecCBwLPA9iRPOD8H7E/yIXl9zr47pb/nkI4fUBlTzjaVMZ4E3JhOdyTpkbEzMA74drp8W2AB0KeWONfnXN9twMh0vgvQPp0+GrgjnT4b+GXO/j8Azkyndybp12f7aufoAyzMmT8SuCed3hVYAewLLAaOSJdPBn6WTq8Gtq08R/U4cl/r3Pn0PX495736NXBmI9/DXXKW/w74Qs57dH06PYy0//y6Xpdq114G3FDP32xvaumPn6RkdXJL/0+1tZ8218WE1eq9SKoiAJDUAfiBpGHAhyTfhP8deDNnn/nAtHTbP0TEIklHkFRDPJZ+KexI8k26NldK+jZJny/nkfQFc2ck34KR9HvgcJJvyldJ+jHJh8RfGnBd9wM/l7QtSVXCoxHxnqQRwKCcOu6dSDpee7Xa/p0lLUqv/3ng/3K2v1FSP5IuCzrUcf4RwAmSLknnOwG90mNV+iQ1+705XNLTJK/9j0g6its5Ih5J199IkpggSRAzJP2BpB+hTCLpmuEB4AuSbgeOA75B0uts1vew0nBJ3wC2A3YhSeJ3p+tmpud7VFIXJe0sdb0uufEtAL6U9XpyvAV0b8R+Vg8ngtJ0BtANODAiNivpnbNT7gbpP/Ywkg+Q6ZKuBv4O/F9EnJ7hHJdGxO2VM5KOqm2jiHgprSM/Fvi+pD9FxOQsFxERGyXNAY4BTiUZtASSkZu+FhEP5jnEexExWNJ2JH3ZXABcQzKYzcMRcZKShvU5dewvkm+nL9Z3Dqq9tiRtBMdXHUTaqZ79jyP5tv0F4HJJA+vZtrpZwHiSapYFEfFuWq2T9T1EUifgVySls5WSJrH19VTvoyao43WR9O8NiL0unUheU2tGbiMoTTsBb6VJYDhQY/xiJWMa/y0irgduIBk6by5wmKTKOv/tJe2V8Zx/AU6UtJ2k7Umqdf4iqTuwISJuJukYr7aG081pyaQ2s0k63aosXUDyoX5+5T6S9krPWatIRm67ELhYH3VLXtmt79k5m75LUkVW6UHga5V15kp6eK3uJZJqjjpFxDvA35W2wwBnAY8oGVNh94h4mKQKZyeSarVc1WPK9QjJ6/llPkqSDX0PKz/016ZtCdXvJKps0/kMSS+Y75DtdWmsvYBax/K1xnMiKE0zgDJJzwJfBF6oZZsjgWfSKoxTgZ9HxBqSD8aZkhaTVCl8OssJI+IpknrnJ0naDG6IiKeBgcCTaRXNFcD3a9l9KrBYaWNxNQ+RVHf8MZKhDCFJXEuBp5Tcgvgb8pR+01gWkwxy8hPgh+m15+73MNC/srGYpOTQIY3tuXS++nH/Bbxc+cFbj/8gqU5bTHJ30mSStoub0/fpaeCaqDnAzCzg0rRRds9q5/4AuAf4fPqbhr6H6fmuJ/nwfZCkyjDXxvR1uo6kChAyvC5KbgS4obZzKul98wlgb0kVks5Ll3cgufGgNXcl/rHk3kfNCkzSSSTVcN9u6Vhas/R1PCAivtPSsbQ1biMwK7CIuFNSax4T++OiPXBVSwfRFrlEYGZW4txGYGZW4pwIzMxKnBOBmVmJcyIwMytxTgRmZiXu/wOXzrEOGOJH3QAAAABJRU5ErkJggg==)

The ROC-AUC score was 0.9509269778971288

I believe that the XGBClassifier model would hold useful mostly to the specific airline that generated the data(because of a possible change in passenger demographics amongst airlines), or to airlines that offer flights at around the same price. Other pricier airlines might benefit from knowing what factors had key influence in predicting satisfaction in this airline, rather than trying to apply predictions directly to their own passengers.

Some tangible results of these models, would be the highest contributing factors to a satisfied passenger. For this selection of data, and of the factors that the airline actually has control over, the most important in satisfying the passenger were in-flight wifi, customer type, online boarding, and checkin service. 

Here is a link to the code:
[My Airline Classification Notebook](https://colab.research.google.com/drive/15OvHHNBbFoGCVkTMWNWve3vW4U5QR35X?usp=sharing)

GitHub Repository:
[GitHub Repository](https://github.com/BrandonSmith710/BrandonSmith710.github.io)