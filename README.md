# decision trees: scikit-learn + pandas

Understanding Decision Tree with scikit-learn and using Iris dataset for the same. 


## Requirements

Need python installed on computer. 
I used IDLE IDE to run python scripts

## Running the script in analyze_dt.py

$ python analyze_dt.py

This:

* Fetches the data using pandas, or grabs the local copy.
* Outputs the *head* of the pandas data frame.
* Fits the decision tree and outputs the *pseudo code* for the decision tree.
* Uses pandas to show that the first branch at *PetalLength <= 2.45* is easily
  verified.

The resulting output is:

```
-- get data:
-- iris.csv found locally

-- df.head():
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa


-- get_code:
if ( PetalLength <= 2.45000004768 ) {
    return Iris-setosa ( 50 examples )
}
else {
    if ( PetalWidth <= 1.75 ) {
        if ( PetalLength <= 4.94999980927 ) {
            if ( PetalWidth <= 1.65000009537 ) {
                return Iris-versicolor ( 47 examples )
            }
            else {
                return Iris-virginica ( 1 examples )
            }
        }
        else {
            return Iris-versicolor ( 2 examples )
            return Iris-virginica ( 4 examples )
        }
    }
    else {
        if ( PetalLength <= 4.85000038147 ) {
            return Iris-versicolor ( 1 examples )
            return Iris-virginica ( 2 examples )
        }
        else {
            return Iris-virginica ( 43 examples )
        }
    }
}

-- look back at original data using pandas
-- df[df['PetalLength'] <= 2.45]]['Name'].unique():  ['Iris-setosa']
```

### 2. Use interactively with IDLE


```python
>>> df=get_iris_data()
-- iris.csv found locally
>>> df2, targets  =encode_target(df, "Name")
>>>features = list(df2.columns[:4])
>>> features
['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
>>> y=df2["Target"]
>>> X=df2[features]
>>> dt=DecisionTreeClassifier(min_samples_split=20, random_state=99)
>>> dt.fit(X,y)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            presort=False, random_state=99, splitter='best')
>>> visualize_tree(dt,features)
```

This will create a dot file which can be opened in Graph File Editor to view the Decision tree