# Python Machine Learning

### Basics
----------------------------------------------------------------
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

----------------------------------------------------------------

### Machine Learning
----------------------------------------------------------------
#### Train & Test
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
: A flow chart helping we make decisions based onprevious experience. <br>
: All Data has to be numerical



<pre>
#import modules we need

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
</pre>

<pre>

</pre>
    
    
    
    
    
    
