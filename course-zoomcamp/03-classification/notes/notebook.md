---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# 3. Machine Learning for Classification

We'll use logistic regression to predict churn


## 3.1 Churn prediction project

* Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
* https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv

<!-- #endregion -->

## 3.2 Data preparation
[Markdown version](notes/3.2-data-preparation.md)

* Download the data, read it with pandas
* Look at the data
* Make column names and values look uniform
* Check if all the columns read correctly
* Check if the churn variable needs any preparation

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```

```python
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
```

```python
# $data lets you use python variables
!wget $data -O data-week-3.csv 
```

```python
df = pd.read_csv('data-week-3.csv')
df.head()
```

It's uncomfortable to look at all the columns at the same time. We can transpose the dataframe (or head) and read it transposed.

```python
df.head().T
```

This way we can quickly see every variable in the data and without scrolling or hiding.

Notice capitalization are not consistent. Recall the last session; turn the columns lowercase

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
```

```python
df.head().T
```

All column names are lowercase and spaces in the data are now underscores

Now we check datatypes

```python
df.dtypes
```

Some things to note:
- seniorcitizen is not a string; int64 0 or 1
- monthlycharges should be a number but it's not

```python
df.totalcharges
```

Let's try to convert this to a number

```python
pd.to_numeric(df.totalcharges)
```

We see that there was an issue; it does not just contain numbers.

Missing data was denoted with a space in the data and we replaced all spaces with underscores. This created a string object, replacing all value with strings.


Pandas has a way to handle them, we can tell Pandas to replace things it cannot handle with NaN

```python
tc = pd.to_numeric(df.totalcharges, errors='coerce')
```

Are any values null?

```python
tc.isnull().sum()
```

```python
df[tc.isnull()][['customerid', 'totalcharges']]
```

Indeed, we can see every totalcharges value that is not in index. Can replace with 0s

```python
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
```

```python
df.totalcharges = df.totalcharges.fillna(0)
```

Perhaps not the best approach. Customer may have actually spent money, but we cannot know anyway.

0 not always the best approach in terms of common sense. But in practice its usually okay.


Now to look at churn

```python
df.churn.head()
```

For machine learning, in classification, we are not interested in yes or no, but rather numbers. Let's convert to 0 (no) and 1 (yes.)

```python
(df.churn == 'yes').astype(int).head()
```

Now write it back to churn

```python
df.churn = (df.churn == 'yes').astype(int)
df.churn
```

Summary:
- Loaded data, looked at it
- Spotted error with totalcharges
- Fixed error with totalcharges
- Converted churn variable (target) to binary clone (0s and 1s)


## 3.3 Setting up the validation framework

* Perform the train/validation/test split with Scikit-Learn


Recall that we split into train/validation/split in NumPy/Pandas. Now we will do so in **Scikit-Learn**.

Scikit-Learn is one of the most popular machine learning libraries and implements many popular algorithms and common utilities. We'll use it for train-test splitting.

```python
from sklearn.model_selection import train_test_split
```

Can use **?** in notebooks to see documentation

```python
train_test_split?
```

We will use test_size to specify the test size split (20%) and random_state for repeatability.

Will split it two times; once to get the test set and once more to get train and validation sets.

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
```

Let's see the test set size

```python
len(df_full_train), len(df_test)
```

```python
# Note that we need 25% split to get 20% of the original dataset size.
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)
```

We can see test/val datasets have the same size, and train dataset has 4225 examples.


Now let us get the target variables.

```python
df_train
```

Do not have to do this (does not affect ML model), but it is nicer when the indices are not shuffled.

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

Again, use underlying NumPy array to grab values then delete from DataFrames

```python
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```

Helps us not accidentally use target variable when training the model.

May notice we did not delete target variable from the full dataframe (df_full_train). The reason for that is that we will look at the target variable more in the next lesson.


## 3.4 EDA

* Check missing values
* Look at the target variable (churn)
* Look at numerical and categorical variables

```python
# Drop indices for readability
df_full_train = df_full_train.reset_index(drop=True)
df_full_train.head().T
```

Check for missing values.

```python
df_full_train.isnull().sum()
```

No additional data preparation here, we already checked for missing values.

Let's look at the target variable.

```python
df_full_train.churn.value_counts(normalize=True)
```

We see that the number of churned users is almost 1/3rd of non-churned users. **0.269968** is our **churn rate**.

Just getting the mean() gives us the churn rate.

```python
global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)
```

This works because, for a binary variable, the negative value is 0 and the positive value is 1, so only positive values contribute to the mean, giving us the same value as the proportion. Essentially:
- (# of 1s)/*n* = Churn Rate


Let's look at the other variables now. Let's check numerical and categorical variables

```python
df_full_train.dtypes
```

Only tenure, monthlycharges, and totalcharges are numerical (seniorcitizen is a binary value.)

```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']
```

```python
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]
```

Let's take a look of the number of unique values for categorical variables.

```python
df_full_train[categorical].nunique()
```

## 3.5 Feature importance: Churn rate and risk ratio

Feature importance analysis (part of EDA) - identifying which features affect our target variable

* Churn rate
* Risk ratio
* Mutual information - later

Now we can look at churn rate by demographic.


#### Churn rate

```python
df_full_train.head()
```

For example, gender. Let's see churn rate among female customers

```python
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female
```

Churn rate for female customers is pretty close to the global churn rate.

```python
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male
```

Churn rate for female customers is pretty close to the global churn rate as well.

```python
global_churn = df_full_train.churn.mean()
global_churn
```

```python
global_churn - churn_female
```

```python
global_churn - churn_male
```

Let's see for churn rate for customers who live with partners.

```python
df_full_train.partner.value_counts()
```

```python
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner
```

```python
global_churn - churn_partner
```

This is noticably less than the global rate.

```python
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner
```

```python
global_churn - churn_no_partner
```

This is noticably *more* than the global rate. It seems to be about 6% less for those with partners and 6% more for those without partners.

It seems that *gender* does not matter for churn, where as having a *partner* does matter. This brings us to our first point.


### 1. **Churn rate** 
Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.
- If group rate churn rate < global churn rate
    - Group is less likely to churn
- If group rate churn rate > global churn rate
    - Group is *more* likely to churn


Instead of comparing them, we can divide one by another, bringing us to our second point.


#### Risk ratio


Dividing the group churn rate by the global churn rate.

```python
churn_no_partner / global_churn
```

The risk ratio for the no-partner group is greater than 1. They are *more* likely to churn.

```python
churn_partner / global_churn
```

The risk ratio for the partner group is less than 1. They are *less* likely to churn. To explain risk ratio:


### 2. **Risk ratio** 
Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms. 
- Risk = group / global
    - \> 1: more likely to churn
    - < 1: less likely to churn


We can see that the no-partner group is about 22% higher, the partner group is about 24% lower. Let's visualize this.

```python
labels = ['Example', 'No-Partner', 'Partner']

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/4, [round(global_churn-0.01, 2), round(churn_no_partner, 2), round(churn_partner, 2)], width, label='Group')
rects2 = ax.bar(x + width/4, [round(global_churn, 2), round(global_churn, 2), round(global_churn, 2)], width, label='Global')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.margins(0.25)
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Partner Status')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=4)
ax.bar_label(rects2, padding=4)

fig.tight_layout()

plt.show()
```

- For the control group, we see there is about the same risk as every other user.
- For the no-partner group, the risk ratio is 22%. They are **high risk**.
- For the partner group, the risk ratio is -6%. They are **low risk**.


We can implement something like this in SQL.


```
SELECT
    gender,
    AVG(churn),
    AVG(churn) - global_churn AS diff,
    AVG(churn) / global_churn AS risk
FROM
    data
GROUP BY
    gender;
```


Let's translate this to Pandas

```python
from IPython.display import display
```

```python
df_full_train.groupby('gender').churn.mean()
```

Need multiple statistics. Can use **.agg()** for this:

```python
df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn
df_group['risk'] = df_group['mean'] / global_churn
df_group
```

Want to do this for every categorical variable

```python
# Need ipython display to make this look nice.
for category in categorical:
    print(category)
    df_group = df_full_train.groupby(category).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()
```

- Senior citizens are more likely to churn
- People who have no partner are more likely to churn (vs less likely for partnered)
    - Important: Predictive power
- People who have no dependents are more likely to churn (vs less likely for with dependents)
    - Similar to partner
- Approximately the same for people with phone service
- People who have no phone service are much less likely, one line slightly less, multiple lines slightly likelier
- People with no internet are very likely to stay
- People with fiber optic internet are at high risk of churn
- People without online backup, device protection or tech support are highly likely to churn
- People with month-to-month are more likely to churn (people with long contracts are very unlikely to churn)
- People with paperless billing are more likely to churn
- People who pay with electronic check are highly likely to churn

A customer with, say, no partner, no kids, and a month-to-month plan are very likely to churn.
- Very unlikely for the opposite


Variables with a high risk ratio are the kind we want to use for ML algorithms.

It would be useful to have a number to describe how important a variable is overall.


## 3.6 Feature importance: Mutual information

Mutual information - concept from information theory, it tells us how much 
we can learn about one variable if we know the value of another

* https://en.wikipedia.org/wiki/Mutual_information


We are using this to measure the importance of a categorical variable.

Previously: we looked at risk ratio to see the importance of categorical variables. Applies to each value within a variables.
- E.g. variable `contract`
    - Can see that people on month-to-month contracts are more likely to churn than those on plans.

We can see that `contract` is important but not if it is more or less important than others. 


Intuition here: the higher the mutual information is, the more we learn about `churn` from a variable.

```python
from sklearn.metrics import mutual_info_score
```

```python
mutual_info_score?
```

Order does not matter.

```python
mutual_info_score(df_full_train.churn, df_full_train.contract)
```

Significant mutual information. If we know the contract type, we do learn a lot about potential churn.

```python
mutual_info_score(df_full_train.gender, df_full_train.churn)
```

Very low mutual information. If we know the gender, we do not learn much about potential churn.

```python
mutual_info_score(df_full_train.partner, df_full_train.churn)
```

Noticable mutual information. Not as much as contract type but much more than gender.


These are numbers are hard to interpet on their own, but we can tell the differences.

What we can do, is check the mutual information of every variable and order them.

```python
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
```

```python
# .apply() variable allows us to run a function on a Pandas Series
mi = df_full_train[categorical].apply(mutual_info_churn_score)  # only applies to categorical variables
mi.sort_values(ascending=False)
```

- `contract` very important
- `onlinesecurity` to `dependents` are decently important.
- Notice a drop in order of magnitude at `partner`, and another for `multiplelines`
- We thought `partner` was relatively important, but was not important in the larger scheme of things


These useful variables are why ML actually works. Variables like contract, onlinesecurity, and techsupport actually give information on churn. 

These are the signals ML models use while training and allow them to make inference on unseen examples.


## 3.7 Feature importance: Correlation

How about numerical columns?

* Correlation coefficient - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient


Also known as Pearson's Correlation. Way to measure degree of dependency between two variables.
- Denoting correlation as *r*, -1 <= *r* <= 1
    - Positive correlation means they are proportional
    - Negative correlation means they are inversely proportional

E.g. variable *x* and *y*.
- Positive correlation means that as *x* grows, *y* grows as well
    - Negative correlation means that as *x* grows, *y* shrinks
- When correlation is between 0 and abs(0.1), the correlation is **low**
    - Increase in *x* rarely leads to an increase in *y*
    - When correlation is between 0.2 and 0.5 or -0.2 and 0.5, the correlation is **moderate**
        - Increase in *x* sometimes leads to an increase in *y*
    - When correlation is between 0.6 and 1.0 or -0.6 and 1.0, the correlation is **strong**
        - Increase in *x* often/always leads to an increase in *y*


In this case
- *y<sub>i</sub>* ∈ {0, 1}
- -∞ < *x* < ∞, *x* ∈ ℝ

E.g. *x* is tenure, *y* is churn.
- 0 < *x* <= 72
- For a positive correlation, if tenure ↑ then churn ↑
    - If *x* ↑ then *y* ↑.
- For a negative correlation, if tenure ↑ then churn ↓
    - If *x* ↑ then *y* ↓.
- For zero correlation, tenure does not really affect churn
    - *x* does not really affect *y*.

```python
df_full_train.tenure.max()
```

Select numerical values, check the correlation between those and churn.

```python
df_full_train[numerical].corrwith(df_full_train.churn)#.abs
```

The longer tenure or totalcharges, the less likely churn is (negative correlation).
- These are correlated, the longer you stay with a company the higher the total charges are.

With high monthly charges, the more likely one is to churn (positive correlation).


Can show this with tenure:

```python
# People with the company for 0, 1 or 2 months
tenure_low = df_full_train[df_full_train.tenure <= 2].churn.mean()
tenure_low
```

Churn rate is high for people with the company 2 months or less. But between 2 and 12 months:

```python
tenure_med = df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()
tenure_med
```

Churn rate is still pretty high but lower between 2 and 12 months. For more than a year:

```python
tenure_long = df_full_train[df_full_train.tenure > 12].churn.mean()
tenure_long
```

Churn rate is much lower after a year.

```python
labels = ['<2 Months', '2-12 Months', '>1 Year']

x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
rects = ax.bar(x, 
    [
        round(tenure_low, 4),
        round(tenure_med, 4),
        round(tenure_long, 4),
    ], 
    width = 0.75,
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.margins(0.25)
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Tenure')
ax.set_xticks(x, labels)

ax.bar_label(rects, padding=4)

fig.tight_layout()

plt.show()
```

Now let's look for monthly charges


For monthly charges less than $20:

```python
charges_low = df_full_train[df_full_train.monthlycharges <= 20].churn.mean()
charges_low
```

Churn rate is just less than 9%. Between $20 and $50:

```python
charges_med = \
    df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean()
charges_med
```

Churn rate is 18%. For charges above $50:

```python
charges_high = df_full_train[df_full_train.monthlycharges > 50].churn.mean()
charges_high
```

Churn rate is 32.5%. This is much more important

```python
labels = ['<$20/Month', '\$20-\$50/Month', '>$50/Month']

x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()
rects = ax.bar(x, 
    [
        round(charges_low, 4),
        round(charges_med, 4),
        round(charges_high, 4),
    ], 
    width = 0.75,
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.margins(0.25)
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Monthly Charges')
ax.set_xticks(x, labels)

ax.bar_label(rects, padding=4)

fig.tight_layout()

plt.show()
```

Overall we can see that tenure has a **negative correlation**, monthly charges has a **positive correlation**.

Also shows how much the variable affects churn. Can simply see the absolute value to just see effect:

```python
df_full_train[numerical].corrwith(df_full_train.churn).abs()
```

## 3.8 One-hot encoding

* Use Scikit-Learn to encode categorical features

```python
from sklearn.feature_extraction import DictVectorizer
```

```python
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
```

## 3.9 Logistic regression

* Binary classification
* Linear vs logistic regression

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

```python
z = np.linspace(-7, 7, 51)
```

```python
sigmoid(10000)
```

```python
plt.plot(z, sigmoid(z))
```

```python
def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result
```

```python
def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result
```

## 3.10 Training logistic regression with Scikit-Learn

* Train a model with Scikit-Learn
* Apply it to the validation dataset
* Calculate the accuracy

```python
from sklearn.linear_model import LogisticRegression
```

```python
model = LogisticRegression(solver='lbfgs')
# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
model.fit(X_train, y_train)
```

```python
model.intercept_[0]
```

```python
model.coef_[0].round(3)
```

```python
y_pred = model.predict_proba(X_val)[:, 1]
```

```python
churn_decision = (y_pred >= 0.5)
```

```python
(y_val == churn_decision).mean()
```

```python
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
```

```python
df_pred['correct'] = df_pred.prediction == df_pred.actual
```

```python
df_pred.correct.mean()
```

```python
churn_decision.astype(int)
```

## 3.11 Model interpretation

* Look at the coefficients
* Train a smaller model with fewer features

```python
a = [1, 2, 3, 4]
b = 'abcd'
```

```python
dict(zip(a, b))
```

```python
dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))
```

```python
small = ['contract', 'tenure', 'monthlycharges']
```

```python
df_train[small].iloc[:10].to_dict(orient='records')
```

```python
dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')
```

```python
dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
```

```python
dv_small.get_feature_names()
```

```python
X_train_small = dv_small.transform(dicts_train_small)
```

```python
model_small = LogisticRegression(solver='lbfgs')
model_small.fit(X_train_small, y_train)
```

```python
w0 = model_small.intercept_[0]
w0
```

```python
w = model_small.coef_[0]
w.round(3)
```

```python
dict(zip(dv_small.get_feature_names(), w.round(3)))
```

```python
-2.47 + (-0.949) + 30 * 0.027 + 24 * (-0.036)
```

```python
sigmoid(_)
```

## 3.12 Using the model

```python
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
```

```python
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
```

```python
y_full_train = df_full_train.churn.values
```

```python
model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)
```

```python
dicts_test = df_test[categorical + numerical].to_dict(orient='records')
```

```python
X_test = dv.transform(dicts_test)
```

```python
y_pred = model.predict_proba(X_test)[:, 1]
```

```python
churn_decision = (y_pred >= 0.5)
```

```python
(churn_decision == y_test).mean()
```

```python
y_test
```

```python
customer = dicts_test[-1]
customer
```

```python
X_small = dv.transform([customer])
```

```python
model.predict_proba(X_small)[0, 1]
```

```python
y_test[-1]
```

## 3.13 Summary

* Feature importance - risk, mutual information, correlation
* One-hot encoding can be implemented with `DictVectorizer`
* Logistic regression - linear model like linear regression
* Output of log reg - probability
* Interpretation of weights is similar to linear regression

<!-- #region -->
## 3.14 Explore more

More things

* Try to exclude least useful features


Use scikit-learn in project of last week

* Re-implement train/val/test split using scikit-learn in the project from the last week
* Also, instead of our own linear regression, use `LinearRegression` (not regularized) and `RidgeRegression` (regularized). Find the best regularization parameter for Ridge

Other projects

* Lead scoring - https://www.kaggle.com/ashydv/leads-dataset
* Default prediction - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


<!-- #endregion -->
