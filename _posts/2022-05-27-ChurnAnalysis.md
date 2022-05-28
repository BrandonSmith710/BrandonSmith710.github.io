Hello, welcome to Customer Churn Analysis. If you'll be following allowing with the code, you'll need to make sure the following requirements are installed in your environment: category_encoders==2.*, pdpbox==0.2.1, imgaug==0.2.5
~~~
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
import io
from google.colab import files
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, f1_score
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('churn.csv', parse_dates = ['joining_date'])
df.drop('Unnamed: 0', axis = 1, inplace = True)

# group the numeric features
df_nums = df.drop('churn_risk_score', axis = 1).select_dtypes(include = [int,
                                                                         float]
                  )
plt.figure(figsize = (18, 13))
for i, col in enumerate(df_nums):
    plt.subplot(2, 3, i + 1)
    plt.title(col)
    df_nums[col].plot(kind = 'kde')
~~~

![image](https://user-images.githubusercontent.com/75755695/170802370-5d654c94-a5a2-4912-a288-811d40ec3ca1.png)
