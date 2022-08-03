# Python Machine Learning

## Basics

#### Data Types
- Numerical
  - Discrete
  - Continuous
- Categorical
  : Unmeasurable up gainst each other : color, yes/no value
- Ordinal
  : Nonnumerical & measurable against each other
<pre>
import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))
</pre>

#### Parameters
- <strong>Mean, Median, and Mode</strong> : With numpy array
- <strong>Standard Deviation</strong> : numpy.std(...)
- <strong>Variance (SD^2)</strong> : numpy.var(...)
- <strong>Percentiles</strong> : numpy.percentile(array, data_point)
- <strong>R-Squared</strong>
  : Determine relationship between x,y {-1:1}

#### Plot {matplotlib.pyplot as plt}
- <strong>Histogram</strong> : plt.hist(array, seperated_period)
- <strong>Scatter</strong> : plt.scatter(x,y)
- <strong>Linear</strong> Regression <br>
  : from <strong>scipy</strong> import <strong>stats</strong> <br>
<pre>
<strong>slope, intercept, relation, p, std_err = stats.linregress(x, y)</strong>
- <strong>relation</strong> : {-1,1 : x,y are 100% related, 0 : unrelated}
</pre>
- <strong>Polynomial Regression</strong> <br>
<pre>
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3)) #Fitted y value <br> #mymodel(FV x) for FV y 
</pre>
    myline = numpy.linspace(1, x[-1], y[-1]) #Create x value to plot
- <strong>Multiple Regression</strong> <br>
  : Multiple vary parameter {x1, x2, ...} to y <br>
  from <strong>sklearn</strong> import <strong>linear_model</strong>
<pre>
Store x,y as <strong>dataFrame</strong>

from sklearn import linear_model

x = df[['Weight', 'Volume']] #Cars
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x, y) #Get a regression object
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

</pre>
<br>

## Machine Learning

#### Train & Test
----------------------------------------------------------------
: Split data into 2 for 
- Train Set like 80%
- Test Set like 20%
<pre>
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

#Train 80%, Test 20%
train_x = x[:80] 
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

#Fit the regression
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0, 6, 100)

#Check how well the fit is
#With Train Set data
r2 = r2_score(train_y, mymodel(train_x))

#With Test Set data
r2test = r2_score(test_y, mymodel(test_x))
</pre>

#### Decision Tree
----------------------------------------------------------------
: A flow chart helping we make decisions based onprevious experience. <br>
: All Data has to be numerical

![dataset](https://user-images.githubusercontent.com/108507768/182319743-4a06e55c-afbe-43eb-948f-3c50b23cb294.jpg)
<br>
<strong>Import Modules</strong>
<pre>
#import modules we need

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
</pre>
<br>
<strong>Set Data</strong>
<pre>
#Data Setting
df = pandas.read_csv("shows.csv")

#Use map() to set numerical value to non-numerical value in the df {'UK': 0, 'USA': 1, 'N': 2}
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

#feature columns : columns we try to predict from
features = ['Age', 'Experience', 'Rank', 'Nationality']
x = df[features]

#target column : column we try to predict
y = df['Go']
</pre>
<br>
<strong>Decision Tree Create</strong>
<pre>
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)

#Make an image to observe
tree.plot_tree(dtree)
</pre>
<pre>
print(dtree.predict([[40, 10, 7, 1]]))
</pre>

#### Confusion Matrix
----------------------------------------------------------------
: It is a table that is used in classification problems to assess where errors in the model were made.

<pre>
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1, 0.9, size = 1000) #Actual data set
predicted = numpy.random.binomial(1, 0.9, size = 1000) #Predicted data set

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

import matplotlib.pyplot as plt

cm_display.plot()
plt.show()
</pre>
    
Accuracy <br>
: Accuracy measures how often the model is correct.
<pre>
Accuracy = metrics.accuracy_score(actual, predicted)
</pre>

Precision
Of the positives predicted, what percentage is truly positive?
<pre>
Precision = metrics.precision_score(actual, predicted)
</pre>

#### Hierarchical Clustering
----------------------------------------------------------------
Unsupervied learning method for clustering data points<br>
- Unsupervied : does not have to be trained, do not need a "target" variable
<pre>
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 
#compute the ward linkage using euclidean distance, and visualize it using a dendrogram
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)
plt.show()
</pre>

#### Logistic Regression
----------------------------------------------------------------
: Solve classification problems by predicting categorical outcomes
- Binomial : Predicts 2 Outcomes
- Multinomial : > 2 Outcomes
<pre>
import numpy

#X represents the size of a tumor in centimeters.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.

#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

from sklearn import linear_model

logr = linear_model.LogisticRegression()
logr.fit(X,y)

predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
</pre>

<strong>Probability</strong>
<pre>
import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logr, X))
</pre>

#### Grid Search
----------------------------------------------------------------
    
    









