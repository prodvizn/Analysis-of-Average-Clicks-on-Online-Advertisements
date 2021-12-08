
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

Check for missing values

```python
data.isnull().sum()
```

Visualize target variable

```python
plt.figure(figsize = (14, 6)) 
plt.subplot(1,2,1)            
sns.countplot(x = 'Clicked on Ad', data = data)
plt.subplot(1,2,2)
sns.distplot(data["Clicked on Ad"], bins = 20)
plt.show()
```

Used pairplot to show the relationship between our target variable and features.

```python
sns.pairplot(data, hue='Clicked on Ad')
```

Correlation between variables

```python
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True)
```

