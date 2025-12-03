# Final Project
Yuxi Wang

### Introduction

Mobile gaming is a fast-growing and highly profitable industry, and
understanding why players spend money can meaningfully shape game design
and revenue strategies. If developers can better anticipate spending
behavior, they can design more effective pricing models, personalized
offers, and targeted features.

This motivates two guiding questions for my analysis:

Q1: Which behavioral features are most strongly associated with players’
in-app spending? Q2: Do spending levels differ across game genres?

By examining these questions, we may find some behavioral and
genre-based patterns that help explain how and why players choose to
spend.

### The Data

The data – `mobile_game_inapp_purchases.csv` – comes from a Kaggle
dataset on mobile game monetization behavior. It contains player-level
information on demographics, gameplay activity, device type, game genre,
and in-app purchase history. The dataset includes variables such as age,
gender, country, session counts, average session length, engagement
metrics, spending amounts, and timestamps of the last purchase. These
fields allow us to analyze both behavioral patterns and spending
behavior. Several variables, such as SessionCount, AverageSessionLength,
and the engineered EngagementScore, reflect how actively players engage
with the game. The InAppPurchaseAmount variable captures the total
amount each player has spent on in-app purchases.

Together, these variables provide a detailed view of player behavior and
spending patterns, making it possible to explore what drives
monetization in mobile games.

### Read

``` python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

purchases = pd.read_csv("/Users/xixi/Desktop/wrangling_final/mobile_game_inapp_purchases.csv", encoding='latin-1')

purchases.head()
purchases.info()
purchases.describe(include='all')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3024 entries, 0 to 3023
    Data columns (total 13 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   UserID                         3024 non-null   object 
     1   Age                            2964 non-null   float64
     2   Gender                         2964 non-null   object 
     3   Country                        2964 non-null   object 
     4   Device                         2964 non-null   object 
     5   GameGenre                      2964 non-null   object 
     6   SessionCount                   3024 non-null   int64  
     7   AverageSessionLength           3024 non-null   float64
     8   SpendingSegment                3024 non-null   object 
     9   InAppPurchaseAmount            2888 non-null   float64
     10  FirstPurchaseDaysAfterInstall  2888 non-null   float64
     11  PaymentMethod                  2888 non-null   object 
     12  LastPurchaseDate               2888 non-null   object 
    dtypes: float64(4), int64(1), object(8)
    memory usage: 307.3+ KB

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | UserID | Age | Gender | Country | Device | GameGenre | SessionCount | AverageSessionLength | SpendingSegment | InAppPurchaseAmount | FirstPurchaseDaysAfterInstall | PaymentMethod | LastPurchaseDate |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| count | 3024 | 2964.000000 | 2964 | 2964 | 2964 | 2964 | 3024.000000 | 3024.000000 | 3024 | 2888.000000 | 2888.000000 | 2888 | 2888 |
| unique | 3024 | NaN | 3 | 27 | 2 | 15 | NaN | NaN | 3 | NaN | NaN | 7 | 225 |
| top | c9889ab0-9cfc-4a75-acd9-5eab1df0015c | NaN | Male | India | Android | Simulation | NaN | NaN | Minnow | NaN | NaN | Debit Card | 2025-05-31 |
| freq | 1 | NaN | 1810 | 242 | 1738 | 219 | NaN | NaN | 2544 | NaN | NaN | 433 | 25 |
| mean | NaN | 33.533738 | NaN | NaN | NaN | NaN | 10.074735 | 20.073978 | NaN | 102.582864 | 15.384003 | NaN | NaN |
| std | NaN | 11.992258 | NaN | NaN | NaN | NaN | 3.115863 | 8.585208 | NaN | 454.339708 | 8.946191 | NaN | NaN |
| min | NaN | 13.000000 | NaN | NaN | NaN | NaN | 1.000000 | 5.010000 | NaN | 0.000000 | 0.000000 | NaN | NaN |
| 25% | NaN | 23.000000 | NaN | NaN | NaN | NaN | 8.000000 | 12.680000 | NaN | 5.987500 | 8.000000 | NaN | NaN |
| 50% | NaN | 33.000000 | NaN | NaN | NaN | NaN | 10.000000 | 20.315000 | NaN | 11.975000 | 16.000000 | NaN | NaN |
| 75% | NaN | 44.000000 | NaN | NaN | NaN | NaN | 12.000000 | 27.420000 | NaN | 17.762500 | 23.000000 | NaN | NaN |
| max | NaN | 54.000000 | NaN | NaN | NaN | NaN | 22.000000 | 34.990000 | NaN | 4964.450000 | 30.000000 | NaN | NaN |

</div>

So we see we have 3024 samples in the mobile_game_inapp_purchases
dataset.

### Feature Engineering

``` python
purchases['EngagementScore'] = purchases['SessionCount'] * purchases['AverageSessionLength']
purchases.describe(include='all')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | UserID | Age | Gender | Country | Device | GameGenre | SessionCount | AverageSessionLength | SpendingSegment | InAppPurchaseAmount | FirstPurchaseDaysAfterInstall | PaymentMethod | LastPurchaseDate | EngagementScore |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| count | 3024 | 2964.000000 | 2964 | 2964 | 2964 | 2964 | 3024.000000 | 3024.000000 | 3024 | 2888.000000 | 2888.000000 | 2888 | 2888 | 3024.000000 |
| unique | 3024 | NaN | 3 | 27 | 2 | 15 | NaN | NaN | 3 | NaN | NaN | 7 | 225 | NaN |
| top | c9889ab0-9cfc-4a75-acd9-5eab1df0015c | NaN | Male | India | Android | Simulation | NaN | NaN | Minnow | NaN | NaN | Debit Card | 2025-05-31 | NaN |
| freq | 1 | NaN | 1810 | 242 | 1738 | 219 | NaN | NaN | 2544 | NaN | NaN | 433 | 25 | NaN |
| mean | NaN | 33.533738 | NaN | NaN | NaN | NaN | 10.074735 | 20.073978 | NaN | 102.582864 | 15.384003 | NaN | NaN | 201.957553 |
| std | NaN | 11.992258 | NaN | NaN | NaN | NaN | 3.115863 | 8.585208 | NaN | 454.339708 | 8.946191 | NaN | NaN | 109.139207 |
| min | NaN | 13.000000 | NaN | NaN | NaN | NaN | 1.000000 | 5.010000 | NaN | 0.000000 | 0.000000 | NaN | NaN | 11.010000 |
| 25% | NaN | 23.000000 | NaN | NaN | NaN | NaN | 8.000000 | 12.680000 | NaN | 5.987500 | 8.000000 | NaN | NaN | 115.920000 |
| 50% | NaN | 33.000000 | NaN | NaN | NaN | NaN | 10.000000 | 20.315000 | NaN | 11.975000 | 16.000000 | NaN | NaN | 186.375000 |
| 75% | NaN | 44.000000 | NaN | NaN | NaN | NaN | 12.000000 | 27.420000 | NaN | 17.762500 | 23.000000 | NaN | NaN | 273.210000 |
| max | NaN | 54.000000 | NaN | NaN | NaN | NaN | 22.000000 | 34.990000 | NaN | 4964.450000 | 30.000000 | NaN | NaN | 732.160000 |

</div>

To better capture how actively players interact with the game, I created
an EngagementScore that combines two key behaviors: how often players
log in and how long they tend to stay. SessionCount and
AverageSessionLength each describe different dimensions of activity, but
on their own they offer only a partial view. By multiplying them, the
engagement score reflects the overall intensity of gameplay, making it a
more comprehensive measure when evaluating whether higher engagement
translates into higher spending.

### Feature Cleaning

``` python
purchases = purchases.dropna(subset = ['Age', 'Gender', 'Country', 'Device', 'GameGenre'])

purchases['InAppPurchaseAmount'] = pd.to_numeric(purchases['InAppPurchaseAmount'], errors='coerce').fillna(0)

purchases['LastPurchaseDate'] = pd.to_datetime(purchases['LastPurchaseDate'], errors='coerce')

purchases = purchases.drop(columns='UserID')

purchases.describe(include='all')
purchases.shape
```

    (2733, 13)

Before analyzing spending behavior, I cleaned and standardized the
dataset to ensure that all key variables were available for numerical
and time-based calculations. Now the final modeling dataset contains
2734 valid samples after preprocessing.

### Question 1: Which behavioral features are most strongly associated with players’ in-app spending?

### Reshaping

To make behavioral metrics easier to analyze and visualize, I reshaped
the dataset into a long format.

``` python
behavior_cols = ['EngagementScore','AverageSessionLength','SessionCount']
purchases_long = purchases.melt(
    id_vars='GameGenre',
    value_vars=behavior_cols,
    var_name='behavior_metric',
    value_name='value'
)
purchases_long.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | GameGenre     | behavior_metric | value  |
|-----|---------------|-----------------|--------|
| 0   | Battle Royale | EngagementScore | 115.47 |
| 1   | Action RPG    | EngagementScore | 213.29 |
| 2   | Fighting      | EngagementScore | 79.83  |
| 3   | Racing        | EngagementScore | 234.72 |
| 4   | Battle Royale | EngagementScore | 152.30 |

</div>

#### Iteration

To understand which engagement behaviors might influence spending, I
iterated through each behavioral variable and computed its correlation
with in-app purchase amount.

``` python
corr_results = []

for col in behavior_cols:
    corr = purchases['InAppPurchaseAmount'].corr(purchases[col])
    corr_results.append((col, corr))

results_spend = pd.DataFrame(corr_results, columns=['Variable','Correlation'])

results_spend
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | Variable             | Correlation |
|-----|----------------------|-------------|
| 0   | EngagementScore      | 0.005084    |
| 1   | AverageSessionLength | -0.023128   |
| 2   | SessionCount         | 0.033376    |

</div>

Based on the correlation analysis across three engagement variables,
none of the behavioral metrics show a strong linear relationship with
spending. SessionCount has the highest correlation value of 0.03 with
spending, although still weak, while EngagementScore and
AverageSessionLength are close to zero.

This suggests that spending behavior is not well explained by basic
activity levels.

#### Regression Model

``` python
X = purchases[behavior_cols]
X = sm.add_constant(X)
y = purchases['InAppPurchaseAmount']

model = sm.OLS(y, X).fit()
model.summary()
```

|                   |                     |                     |           |
|-------------------|---------------------|---------------------|-----------|
| Dep. Variable:    | InAppPurchaseAmount | R-squared:          | 0.002     |
| Model:            | OLS                 | Adj. R-squared:     | 0.001     |
| Method:           | Least Squares       | F-statistic:        | 1.739     |
| Date:             | Wed, 03 Dec 2025    | Prob (F-statistic): | 0.157     |
| Time:             | 08:34:43            | Log-Likelihood:     | -20600.   |
| No. Observations: | 2733                | AIC:                | 4.121e+04 |
| Df Residuals:     | 2729                | BIC:                | 4.123e+04 |
| Df Model:         | 3                   |                     |           |
| Covariance Type:  | nonrobust           |                     |           |

OLS Regression Results

|                      |          |         |        |          |         |         |
|----------------------|----------|---------|--------|----------|---------|---------|
|                      | coef     | std err | t      | P\>\|t\| | \[0.025 | 0.975\] |
| const                | 131.4128 | 74.194  | 1.771  | 0.077    | -14.068 | 276.894 |
| EngagementScore      | 0.2792   | 0.325   | 0.860  | 0.390    | -0.357  | 0.916   |
| AverageSessionLength | -4.0050  | 3.396   | -1.179 | 0.238    | -10.664 | 2.654   |
| SessionCount         | -0.7405  | 7.075   | -0.105 | 0.917    | -14.613 | 13.132  |

|                |          |                   |            |
|----------------|----------|-------------------|------------|
| Omnibus:       | 3676.366 | Durbin-Watson:    | 1.971      |
| Prob(Omnibus): | 0.000    | Jarque-Bera (JB): | 526728.279 |
| Skew:          | 7.810    | Prob(JB):         | 0.00       |
| Kurtosis:      | 69.193   | Cond. No.         | 1.97e+03   |

<br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.97e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.

The regression model further confirms that behavioral metrics explain
almost none of the variation in spending. All coefficients are close to
zero and statistically insignificant, and the R-squared value is only
0.002.

### Question 2: Do spending levels vary across game genres?

#### Summarizing

To compare monetization patterns across different types of games, I
summarized spending and engagement by game genre.

``` python
genre_summary = purchases.groupby('GameGenre').agg(
    avg_spend=('InAppPurchaseAmount','mean'),
    total_spend=('InAppPurchaseAmount','sum'),
    count=('InAppPurchaseAmount','count'),
    avg_engagement=('EngagementScore','mean')
)

genre_summary.sort_values('avg_spend', ascending=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|               | avg_spend  | total_spend | count | avg_engagement |
|---------------|------------|-------------|-------|----------------|
| GameGenre     |            |             |       |                |
| Battle Royale | 158.517399 | 27423.51    | 173   | 211.215491     |
| Racing        | 147.753750 | 24822.63    | 168   | 193.014583     |
| Strategy      | 144.622514 | 26465.92    | 183   | 190.315519     |
| Fighting      | 135.794091 | 23899.76    | 176   | 200.157386     |
| MOBA          | 130.473471 | 22180.49    | 170   | 204.399882     |
| MMORPG        | 98.527486  | 18030.53    | 183   | 204.307596     |
| Adventure     | 94.721592  | 14871.29    | 157   | 209.337070     |
| Role Playing  | 93.354888  | 16617.17    | 178   | 205.535562     |
| Card          | 85.536576  | 15738.73    | 184   | 214.205054     |
| Simulation    | 80.907379  | 16666.92    | 206   | 207.906602     |
| Sports        | 77.636908  | 16070.84    | 207   | 190.349130     |
| Casual        | 76.984649  | 14242.16    | 185   | 194.739243     |
| Puzzle        | 75.051630  | 13809.50    | 184   | 205.532989     |
| Action RPG    | 60.738655  | 10386.31    | 171   | 197.114327     |
| Sandbox       | 55.947356  | 11637.05    | 208   | 196.684375     |

</div>

According to the result, battle Royale players spend the most on
average, approximately \$158 per player and also generate the highest
total spending. Strategy and Racing genres follow as the next highest
spenders.

Simultaneously, card games have the highest engagement score but
relatively moderate spending, suggesting that high engagement does not
necessarily translate into higher spending.

#### Visualizing

``` python
plt.figure(figsize=(10,6))
sns.barplot(
    data=genre_summary.reset_index(),
    x="avg_spend",
    y="GameGenre"
)

plt.xlabel("Average Spending")
plt.ylabel("Game Genre")
plt.title("Average In-App Spending by Game Genre")
plt.show()
```

![](readme_files/figure-commonmark/cell-9-output-1.png)

The bar chart clearly shows these differences, with Battle Royale and
Strategy leading all other genres in average spending.

### Conclusion

Across both research questions, the analysis suggests that player
spending is not strongly tied to basic engagement behaviors.
Correlations between spending and activity metrics are extremely weak,
and the regression model confirms that these variables explain almost
none of the variation in in-app purchase amounts. This indicates that
simple activity levels are not reliable predictors of how much a player
spends.

However, spending patterns do vary substantially across game genres.
Genres with competitive or fast-paced gameplay, such as Battle Royale,
show the highest average spending, while genres like Strategy and Racing
also rank among top spenders. In contrast, genres with high engagement
scores like Card games do not necessarily exhibit high spending levels,
highlighting an important distinction between playing more and spending
more.

Taken together, these findings suggest that monetization behavior is
shaped less by how intensely players engage with a game, and more by the
type of game they choose to play. Understanding genre-level spending
tendencies may provide developers with a more reliable starting point
when designing pricing strategies or in-game purchase systems.
