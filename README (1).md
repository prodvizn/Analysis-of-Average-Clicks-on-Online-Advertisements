
# Ad Click Prediction

In this project, we will build both a Logistic Regression & a Random Forest model to indicate whether or not a particular internet user clicked on an Advertisement. 

This model will predict whether or not a user will click on an ad based off the features of that user.


## Data

This data set contains the following features:

* **'Daily Time Spent on Site':** consumer time on site in minutes
* **'Age':** cutomer age in years
* **'Area Income':** Avg. Income of geographical area of consumer
* **'Daily Internet Usage':** Avg. minutes a day consumer is on the internet
* **'Ad Topic Line':** Headline of the advertisement
* **'City':** City of consumer
* **'Male':** Whether or not consumer was male
* **'Country':** Country of consumer
* **'Timestamp':** Time at which consumer clicked on Ad or closed window
* **'Clicked on Ad':** 0 or 1 indicated clicking on Ad
## Load Libraries/Data Extraction

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

```python
data = pd.read_csv("advertising.csv")
data.head()
```


## Exploratory Data Analysis

**Check for missing values**

```python
data.isnull().sum()
```

**Visualize target variable**

```python
plt.figure(figsize = (14, 6)) 
plt.subplot(1,2,1)            
sns.countplot(x = 'Clicked on Ad', data = data)
plt.subplot(1,2,2)
sns.distplot(data["Clicked on Ad"], bins = 20)
plt.show()
```
![image](https://user-images.githubusercontent.com/87580423/145317625-692bf00e-fd58-4a78-99fa-2ff4e37bf2d9.png)

**Used pairplot to show the relationship between our target variable and features.**

```python
sns.pairplot(data, hue='Clicked on Ad')
```
![image](https://user-images.githubusercontent.com/87580423/145315467-84cdf438-c176-4733-80d1-b39d8de315bf.png)

**Correlation between variables**

```python
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True)
```

![image](https://user-images.githubusercontent.com/87580423/145315394-ecbae9ba-958d-4034-891e-380a9bf7ff38.png)

## Extracted Features Visualizations

**Clicks vs Months**
```python
f,ax=plt.subplots(1,2,figsize=(14,5))
data['Month'][data['Clicked on Ad']==1].value_counts().sort_index().plot(ax=ax[0])
ax[0].set_title('Months Vs Clicks')
ax[0].set_ylabel('Count of Clicks')
pd.crosstab(data["Clicked on Ad"], data["Month"]).T.plot(kind = 'bar',ax=ax[1])

plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/87580423/145317190-03dcab62-61d3-4808-997d-d988efadcb3c.png)

**Clicks vs Hour/Day of the week**

```python
f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(data["Clicked on Ad"], data["Hour"]).T.plot(style = [], ax = ax[0])
pd.pivot_table(data, index = ['Weekday'], values = ['Clicked on Ad'],aggfunc= np.sum).plot(kind = 'bar', ax=ax[1]) # 0 - Monday
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/87580423/145317760-3d72e813-8b62-41dd-81ae-329d42edfb56.png)

**Clicked vs Not Clicked**

```python
data.groupby('Clicked on Ad')['Clicked on Ad', 'Daily Time Spent on Site', 'Age', 'Area Income', 
                            'Daily Internet Usage'].mean()
```

![image](https://user-images.githubusercontent.com/87580423/145318068-ba9293a0-285b-47b1-82a6-0c4199165189.png)

**Clicked vs Not Clicked by Gender & Clicked Vs Not Clicked by Day of the Week**

```python
f,ax=plt.subplots(1,2,figsize=(14,5))
sns.set_style('whitegrid')
sns.countplot(x='Male',hue='Clicked on Ad',data=data,palette='bwr', ax = ax[0]) # Overall distribution of Males and females count
table = pd.crosstab(data['Weekday'],data['Clicked on Ad'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax[1], grid = False) # 0 - Monday
ax[1].set_title('Stacked Bar Chart of Weekday vs Clicked')
ax[1].set_ylabel('Proportion by Day')
ax[1].set_xlabel('Weekday')
plt.tight_layout()
plt.show()
```

![image](https://user-images.githubusercontent.com/87580423/145318339-5add270b-e405-4bd2-a85c-0d8d7c6cbb9f.png)

**Clicked vs Not Clicked by Month**

```python
sns.factorplot('Month', 'Clicked on Ad', hue='Male', data = data)
plt.show()
```

![image](https://user-images.githubusercontent.com/87580423/145318862-c5098c1e-009b-4b73-ad6a-13357687200c.png)

## Basic Model Building Based on the actual data

![image](https://user-images.githubusercontent.com/87580423/145319303-3cf7d92f-9653-409a-a30e-b9fc7fdf81c5.png)

## Building a Basic Model

![image](https://user-images.githubusercontent.com/87580423/145319655-e750713d-aff7-4d4b-adf4-d91e09eafa08.png)

## Predictions

![image](https://user-images.githubusercontent.com/87580423/145319794-169b8eb8-c83c-4e46-b22c-4d450f13a8a4.png)

## Performance Metrics

**Now we need to see how far our predictions met the actual test data (y_test) by performing evaluations using classification report & confusion matrix on the target variable and the predictions.**

![image](https://user-images.githubusercontent.com/87580423/145320103-fcb1b819-25f5-411d-933f-9dcaef33ead8.png)

## Feature Engineering

![image](https://user-images.githubusercontent.com/87580423/145321384-2ebf0377-f059-4d0c-b506-580b4d0ac05c.png)

There does not seem to be any effect of month, day, weekday and hour on the target variable.

**Dummy encode on Month, Weekday Columns then create buckets for hour columns.**

**Also, feature engineering on Age column**
![image](https://user-images.githubusercontent.com/87580423/145322813-5c1b55ac-f8b6-4897-aaca-66a2159b56e4.png)

**Checking bins for Age Column:**

![image](https://user-images.githubusercontent.com/87580423/145323417-06cceee8-b324-4b05-9c49-f23476e94fd8.png)

**Creating Bins on Age Column/Dummy encoding**

![image](https://user-images.githubusercontent.com/87580423/145324251-b66e3607-5d2f-435e-a15e-885744519d5c.png)

**Removing Redundent and no predictive power features**

![image](https://user-images.githubusercontent.com/87580423/145324418-8d704fe7-b1f5-4641-8e51-39968f5b66e8.png)

## Building a Logistic Regression Model

![image](https://user-images.githubusercontent.com/87580423/145325413-9ce708fa-d74b-4bee-be9d-056474eb1d90.png)

***We can see that the feature Male(Gender) does not contribute to the model (i.e., see x4 since the P-value is >= .05) so we can actually remove that variable from our model.***

After removing the variable if the Adjusted R-squared has not changed from the previous model. Then we could conclude that the feature indeed was not contributing to the model. 

**Looks like the contributing features for the model are (P-value < .05):**

* Daily Time Spent on site
* Daily Internet Usage
* Age
* Country
* Area income

**Accuracy Score, K fold Cross-Validation, Confusion Matrix**

![image](https://user-images.githubusercontent.com/87580423/145325673-bd23e64a-20b5-4a93-9e28-57cb9e44e8d3.png)

## Random Forest Model

**Accuracy Score, K fold Cross-Validation, Confusion Matrix**

![image](https://user-images.githubusercontent.com/87580423/145325972-7a72a5cc-683f-469e-b110-565419724385.png)

## Test Model Performance

![image](https://user-images.githubusercontent.com/87580423/145326252-cd98d26f-cea2-47d4-a897-26b1493f27cd.png)

**We can observe that random forest has higher accuracy compared to logistic regression model in both test and train data sets.**

## ROC Graph

![image](https://user-images.githubusercontent.com/87580423/145326564-db411857-c45f-4a84-a340-e496610dda04.png)

## Random Forest Feature Importance

![image](https://user-images.githubusercontent.com/87580423/145326783-65f233e0-2c3a-4474-9ae4-2f8a91f7ba3c.png)

![image](https://user-images.githubusercontent.com/87580423/145326877-532a104f-94eb-4418-95fd-bfab828e50be.png)

**Recommendations**

Above are the dominant features our model is predicting so our target population are the people:

* Who Spends less time on the internet
* Who spends less time on the website
* Who has lower income
* Who are older than our average sample (mean around 40 years old)