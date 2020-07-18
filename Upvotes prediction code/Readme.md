

```python
# DataFrame
import pandas as pd
# Linear Algebra
import numpy as np
# Viz
import seaborn as sns
import matplotlib.pyplot as plt
# Stats
import scipy.stats as sp
# Predictive
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, ShuffleSplit
# Misc
import warnings
```


```python
warnings.filterwarnings('ignore')
```

### Read Data


```python
df = pd.read_csv("train.csv")
```


```python
test = pd.read_csv("test.csv")
```

### Basic checks


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Tag</th>
      <th>Reputation</th>
      <th>Answers</th>
      <th>Username</th>
      <th>Views</th>
      <th>Upvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>52664</td>
      <td>a</td>
      <td>3942.0</td>
      <td>2.0</td>
      <td>155623</td>
      <td>7855.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>327662</td>
      <td>a</td>
      <td>26046.0</td>
      <td>12.0</td>
      <td>21781</td>
      <td>55801.0</td>
      <td>1175.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>468453</td>
      <td>c</td>
      <td>1358.0</td>
      <td>4.0</td>
      <td>56177</td>
      <td>8067.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>96996</td>
      <td>a</td>
      <td>264.0</td>
      <td>3.0</td>
      <td>168793</td>
      <td>27064.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>131465</td>
      <td>c</td>
      <td>4271.0</td>
      <td>4.0</td>
      <td>112223</td>
      <td>13986.0</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (330045, 7)




```python
df.describe(include="all")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Tag</th>
      <th>Reputation</th>
      <th>Answers</th>
      <th>Username</th>
      <th>Views</th>
      <th>Upvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>330045.000000</td>
      <td>330045</td>
      <td>3.300450e+05</td>
      <td>330045.000000</td>
      <td>330045.000000</td>
      <td>3.300450e+05</td>
      <td>330045.000000</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>top</td>
      <td>NaN</td>
      <td>c</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>NaN</td>
      <td>72458</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>235748.682789</td>
      <td>NaN</td>
      <td>7.773147e+03</td>
      <td>3.917672</td>
      <td>81442.888803</td>
      <td>2.964507e+04</td>
      <td>337.505358</td>
    </tr>
    <tr>
      <td>std</td>
      <td>136039.418471</td>
      <td>NaN</td>
      <td>2.706141e+04</td>
      <td>3.579515</td>
      <td>49215.100730</td>
      <td>8.095646e+04</td>
      <td>3592.441135</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>117909.000000</td>
      <td>NaN</td>
      <td>2.820000e+02</td>
      <td>2.000000</td>
      <td>39808.000000</td>
      <td>2.594000e+03</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>235699.000000</td>
      <td>NaN</td>
      <td>1.236000e+03</td>
      <td>3.000000</td>
      <td>79010.000000</td>
      <td>8.954000e+03</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>353620.000000</td>
      <td>NaN</td>
      <td>5.118000e+03</td>
      <td>5.000000</td>
      <td>122559.000000</td>
      <td>2.687000e+04</td>
      <td>107.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>471493.000000</td>
      <td>NaN</td>
      <td>1.042428e+06</td>
      <td>76.000000</td>
      <td>175738.000000</td>
      <td>5.231058e+06</td>
      <td>615278.000000</td>
    </tr>
  </tbody>
</table>
</div>



Let us check for missing values


```python
df.isnull().sum()
```




    ID            0
    Tag           0
    Reputation    0
    Answers       0
    Username      0
    Views         0
    Upvotes       0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 330045 entries, 0 to 330044
    Data columns (total 7 columns):
    ID            330045 non-null int64
    Tag           330045 non-null object
    Reputation    330045 non-null float64
    Answers       330045 non-null float64
    Username      330045 non-null int64
    Views         330045 non-null float64
    Upvotes       330045 non-null float64
    dtypes: float64(4), int64(2), object(1)
    memory usage: 17.6+ MB



```python
df['Tag'].value_counts()
```




    c    72458
    j    72232
    p    43407
    i    32400
    a    31695
    s    23323
    h    20564
    o    14546
    r    12442
    x     6978
    Name: Tag, dtype: int64



There are no missing values in this dataset.

### Feature addition and new dataframe generation

We also break the dataset into X and Y.


```python
def username_freq(df):
    # Append the frequency of user in `new_df`
    username_df = pd.DataFrame(df['Username'].value_counts().reset_index())
    username_df.columns = ['Username', 'Freq']
    new_df = df.set_index('Username').join(username_df.set_index('Username'), how = 'left').reset_index()
    X = new_df[["Reputation","Answers","Views","Freq"]]
    try:
        Y = new_df[["Upvotes"]]
    except (UnboundLocalError, KeyError):
        Y = 0
        pass
    return new_df,X, Y
```


```python
new_df, X, y = username_freq(df)
```

### Dealing with categorical variables

OneHotEncoder cannot process string values directly. If your nominal features are strings, then you need to first map them into integers.

pandas.get_dummies is kind of the opposite. By default, it only converts string columns into one-hot representation, unless columns are specified.

However, OHE has an option to handle unknown values.

Hence, we use OneHotEncoder.


#### OHE


```python
def ohe_cat_var(new_df, X):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe.fit_transform(new_df[['Tag']]).toarray())
    ohe_df.columns = ohe.get_feature_names(['Tag']).tolist()
    # merge with main df bridge_df on key values
    X_ohe = X.join(ohe_df)
    return X_ohe
```


```python
X_ohe = ohe_cat_var(new_df, X)
```


```python
X_ohe.shape
```




    (330045, 14)



### Generate combined dataframe called new_df_1


```python
new_df_1 = X_ohe.join(y, how = "outer")
```

### Feature importance

Before proceeding with our main analysis, let us check what features are important using chisq test. We see which variable has the strongest relationship with our target variable and rank them accordingly.


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_ohe,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_ohe.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
```

             Specs         Score
    2        Views  5.048328e+10
    0   Reputation  1.514921e+10
    3         Freq  2.390643e+06
    1      Answers  3.428539e+05
    8        Tag_j  1.086871e+04
    6        Tag_h  8.571940e+03
    4        Tag_a  7.286273e+03
    10       Tag_p  7.219698e+03
    12       Tag_s  7.209124e+03
    5        Tag_c  7.099724e+03


We learn that Views, Reputation, Freq and Answers have a strong relationship with our target variable in that order. This information will be used to prioritise features for data manipulation

Looking at all distributions without a filter on Tag


```python
fig = plt.figure(figsize = (15,10))
ax = fig.gca()
hist = new_df_1.hist(ax = ax)
```


![png](output_31_0.png)


## Exploration

### Pairplot
sns.pairplot(new_df_1)
### Correlation


```python
sns.set(style="white")

# Compute the correlation matrix
corr = round(new_df_1.corr(),1)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 30))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2f8164628d0>




![png](output_36_1.png)


There is mild correlation in the dataset but that should not stop us from proceeding with the models.

## Models

### Functions


```python
def stratified_cross_val_results(reg, df,target , no_of_splits):
    rmse = []
    skf = StratifiedKFold(n_splits=no_of_splits, random_state=42)
    #skf.get_n_splits(df, target)
    for train_index, test_index in skf.split(df, target):
        #print("Train:", train_index, "Validation:", test_index)
        X1_train, X1_test = df.iloc[train_index], df.iloc[test_index]
        y1_train, y1_test = target.iloc[train_index], target.iloc[test_index]
        reg.fit(X1_train, y1_train)
        prediction = reg.predict(X1_test)
        score = np.sqrt(mean_squared_error(y1_test, prediction))#accuracy_score(prediction, y1_test)
        rmse.append(score)
    print("Stratified cross val Mean RMSE: ", np.mean(rmse))
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
    #np.sqrt(mean_squared_error(y_test, pred))
    #return reg, np.mean(rmse)
```


```python
def validation_curve_function(fn,parameter_range,parameter_name,xlab):
    param_range = parameter_range
    train_scores, test_scores = validation_curve(
        fn, X_train, y_train, param_name=parameter_name, param_range=param_range,
        cv=4, scoring="neg_root_mean_squared_error", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #plt.figure(figsize=(17,9))
    plt.figure(figsize=(20,5))
    plt.title("Validation Curve")
    plt.xlabel(xlab)
    plt.ylabel("Score")
    #plt.ylim(0.4, 1.1)
    lw = 1
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
```


```python
def cross_val_score_calculation(reg, X,y,cv):
    scores =  cross_val_score(reg, X, y, cv=cv, n_jobs=-1, scoring= "neg_root_mean_squared_error")*-1
    print("Min score: ", scores.min())
    print("25th percentile: ", np.quantile(scores, 0.25))
    print("Mean score: ", scores.mean())
    print("75th percentile: ", np.quantile(scores, 0.75))
    print("Max score: ", scores.max())
```

### Determine X and y variables


```python
X = new_df_1.drop(['Upvotes'], axis=1)
```


```python
y = new_df_1["Upvotes"]
```

### Train test split

We divide the dataset in 70:30 ratio


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.metrics import mean_squared_error
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7,random_state = 42)
```

### Base models

#### Decision Tree


```python
from sklearn.tree import DecisionTreeRegressor
```

##### Hyperparameter estimation


```python
validation_curve_function(DecisionTreeRegressor(),list(range(5,21)),"max_depth","max_depth")
```


![png](output_55_0.png)



```python
validation_curve_function(DecisionTreeRegressor(max_depth = 12),list(range(8,50,2)),"max_leaf_nodes","max_leaf_nodes")
```


![png](output_56_0.png)


##### cross val score


```python
dtr = DecisionTreeRegressor(max_depth = 12, max_leaf_nodes=35)
```


```python
cross_val_score_calculation(dtr, X, y, cv=10)
```

    Min score:  1212.9518902564116
    25th percentile:  1234.5269952033384
    Mean score:  1601.5007472324207
    75th percentile:  1834.1410705456497
    Max score:  2579.6878000702786



```python
cross_val_score_calculation(dtr, X, y, cv=20)
```

    Min score:  712.3408753243086
    25th percentile:  1197.4351974659457
    Mean score:  1510.722063569337
    75th percentile:  1664.5277340441949
    Max score:  2682.957759602357



```python
cross_val_score_calculation(dtr, X, y, cv=30)
```

    Min score:  646.4517576126489
    25th percentile:  951.0505792440201
    Mean score:  1510.8299712301716
    75th percentile:  1714.2245460655581
    Max score:  4051.918433366182


##### Stratified k-fold


```python
stratified_cross_val_results(dtr, X_train, y_train, 10)
```

    Stratified cross val Mean RMSE:  1757.009754336984
    Prediction RMSE:  1527.6748850340532



```python
stratified_cross_val_results(dtr, X_train, y_train, 20)
```

    Stratified cross val Mean RMSE:  1687.9112521336933
    Prediction RMSE:  1565.7043056590173



```python
stratified_cross_val_results(dtr, X_train, y_train, 30)
```

    Stratified cross val Mean RMSE:  1530.1084124083475
    Prediction RMSE:  1565.7043056590173


##### Modeling


```python
dtr.fit(X_train, y_train)
pred = dtr.predict(X_test)
print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
```

    Prediction RMSE:  2256.4630291746007


### Ensemble methods

We are going to try different ensemble models and decide which model performs the best.
The models are
1. Random Forest
2. Light GBM
3. Gradient Boosting Regressor
4. XGBoost

#### Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
```

##### Hyperparameter estimation


```python
validation_curve_function(RandomForestRegressor(),list(range(6,34,2)),"max_depth","max_depth")
```


![png](output_73_0.png)



```python
validation_curve_function(RandomForestRegressor(max_depth=10),list(range(8,34,2)),"max_leaf_nodes","max_leaf_nodes")
```


![png](output_74_0.png)



```python
rf = RandomForestRegressor(max_depth=10, max_leaf_nodes=22)
```

##### cross val score


```python
cross_val_score_calculation(rf, X, y, cv=10)
```

    Min score:  1080.188935530931
    25th percentile:  1203.2449501410858
    Mean score:  1430.1046554617249
    75th percentile:  1609.5099969125686
    Max score:  2196.7537708897435



```python
cross_val_score_calculation(rf, X, y, cv=20) #2511 max
```

    Min score:  663.2472935502562
    25th percentile:  1043.8879115883697
    Mean score:  1322.8461045701317
    75th percentile:  1541.5913718040515
    Max score:  2742.586504002012



```python
cross_val_score_calculation(rf, X, y, cv=30) #1156 mean/median
```

    Min score:  607.9860277601267
    25th percentile:  815.2714252659008
    Mean score:  1281.405078621515
    75th percentile:  1410.061527139581
    Max score:  3557.3905530808647


##### Stratified k-fold


```python
stratified_cross_val_results(rf, X_train, y_train, 10)
```

    Stratified cross val Mean RMSE:  1364.571247777451
    Prediction RMSE:  1220.665352478296



```python
stratified_cross_val_results(rf, X_train, y_train, 20)
```

    Stratified cross val Mean RMSE:  1295.6788701235519
    Prediction RMSE:  1201.4145659253454



```python
stratified_cross_val_results(rf, X_train, y_train, 30)
```

    Stratified cross val Mean RMSE:  1258.097310951358
    Prediction RMSE:  1230.722433996946


##### Modeling


```python
rf = RandomForestRegressor(max_depth=10, max_leaf_nodes=22, oob_score=True)
```


```python
rf.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=10, max_features='auto', max_leaf_nodes=22,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=True,
                          random_state=None, verbose=0, warm_start=False)




```python
pred = rf.predict(X_test)
print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
```

    Prediction RMSE:  1210.562066788874

#### LightGBMfrom lightgbm import LGBMRegressorlgb = LGBMRegressor()#X_train_train, X_valid, y_train_train, y_valid = train_test_split(X_train, y_train, train_size = 0.8, random_state = 42)# lgbm_train = X_train_train.join(y_train_train, how = 'outer')# lgbm_valid =  X_valid.join(y_valid, how = 'outer')##### Using default parameterslgb.fit(X_train, y_train)pred = lgb.predict(X_test)
print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))import lightgbm as lgbparams = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 10, 
    'learning_rate': 0.1,
    'verbose': 0, 
    'early_stopping_round': 20,
    'n_estimators' : [200, 400, 500]}#n_estimators = 100
n_iters = 15
preds_buf = []
err_buf = []
for i in range(n_iters):
    X_train_train, X_valid, y_train_train, y_valid = train_test_split(X_train, y_train, train_size = 0.9, random_state = i)
    #x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=i)
    d_train = lgb.Dataset(X_train_train, label=y_train_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid)
    watchlist = [d_valid]

    model = lgb.train(params, d_train,  watchlist, verbose_eval=1)
    
    pred = model.predict(X_test)
    err = np.sqrt(mean_squared_error(y_test, pred))
    err_buf.append(err)
    print('RMSE = ' + str(err))
    
    pred = model.predict(X_test)
    preds_buf.append(pred)

print('Mean RMSE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))err_bufX_train_train, X_valid, y_train_train, y_valid = train_test_split(X_train, y_train, train_size = 0.9, random_state = 11)
d_train = lgb.Dataset(X_train_train, label=y_train_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)watchlist = [d_valid]

model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)pred = model.predict(X_test)np.sqrt(mean_squared_error(y_test, pred))
#### Gradient Boosting Regressor


```python
from sklearn.ensemble import GradientBoostingRegressor 
```

##### Hyperparameter estimation


```python
validation_curve_function(GradientBoostingRegressor(),list(range(6,34,2)),"max_depth","max_depth")
```


![png](output_108_0.png)



```python
validation_curve_function(GradientBoostingRegressor(max_depth=12),list(range(8,34,2)),"max_leaf_nodes","max_leaf_nodes")
```


![png](output_109_0.png)



```python
gbr = GradientBoostingRegressor(max_depth=12, max_leaf_nodes=12)
```

##### cross val score


```python
cross_val_score_calculation(gbr, X, y, cv=10)
```

    Min score:  827.8122043633111
    25th percentile:  1121.98182858242
    Mean score:  1190.6475021203219
    75th percentile:  1352.2699880318348
    Max score:  1473.2462636218984



```python
cross_val_score_calculation(gbr, X, y, cv=20)
```

    Min score:  430.5441761398115
    25th percentile:  840.5278121295528
    Mean score:  1116.8207639265988
    75th percentile:  1373.477838444866
    Max score:  2195.7552672579063



```python
cross_val_score_calculation(gbr, X, y, cv=30)
```

    Min score:  422.924662618468
    25th percentile:  654.7450722910226
    Mean score:  1081.7800480686385
    75th percentile:  1265.486022088334
    Max score:  2327.6956535967847



```python
cross_val_score_calculation(gbr, X, y, cv=40)
```

    Min score:  295.7146603670245
    25th percentile:  638.334165607519
    Mean score:  1048.7535203862408
    75th percentile:  1155.5002767169885
    Max score:  3089.12891005014



```python
cross_val_score_calculation(gbr, X, y, cv=50)
```

    Min score:  113.5027622575795
    25th percentile:  595.177038794851
    Mean score:  999.5341734554868
    75th percentile:  1199.9426437930165
    Max score:  3076.6036431213934


##### Stratified k-fold


```python
stratified_cross_val_results(gbr, X_train, y_train, 10)
```

    Stratified cross val Mean RMSE:  1224.4146018440729
    Prediction RMSE:  1165.7458325932448



```python
stratified_cross_val_results(gbr, X_train, y_train, 20)
```

    Stratified cross val Mean RMSE:  1067.6092366047692
    Prediction RMSE:  1168.5243545404567



```python
stratified_cross_val_results(gbr, X_train, y_train, 30)
```

    Stratified cross val Mean RMSE:  1069.8884921608096
    Prediction RMSE:  1153.3558193595136



```python
stratified_cross_val_results(gbr, X_train, y_train, 40)
```

    Stratified cross val Mean RMSE:  1092.2235412597572
    Prediction RMSE:  1168.939541098438



```python
stratified_cross_val_results(gbr, X_train, y_train, 50)
```

    Stratified cross val Mean RMSE:  1081.468683253745
    Prediction RMSE:  1160.9295646947842


##### Modeling


```python
gbr.fit(X_train, y_train)
```


```python
pred = rf.predict(X_test)
print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
```

    Prediction RMSE:  1210.562066788874

#### Stacking classifierfrom sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFolddef evaluate_model(model):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('rf', RandomForestRegressor()))
	level0.append(('gbr', GradientBoostingRegressor()))
    #	level0.append(('lgbm', ))
	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model# get a list of models to evaluate
def get_models():
	models = dict()
	models['rf'] = RandomForestRegressor()
	models['gbr'] = GradientBoostingRegressor()
    # models['svm'] = SVR()
	models['stacking'] = get_stacking()
	return modelsmodels = get_models()
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
#### CatboostRegressor


```python
from catboost import CatBoostRegressor
```

##### Hyperparameter estimation


```python
validation_curve_function(CatBoostRegressor(learning_rate=0.1, loss_function = "MAPE",eval_metric="MAPE", boost_from_average = True),list(range(6,20,2)),"max_depth","max_depth")
```


![png](output_135_0.png)



```python
ctb = CatBoostRegressor(learning_rate=0.1, loss_function = "MAPE",eval_metric="MAPE", boost_from_average = True, max_depth = 8)
```

##### cross val score


```python
cross_val_score_calculation(ctb, X, y, cv=10)
```

    Min score:  2251.766953858753
    25th percentile:  2690.265217759018
    Mean score:  3431.375948841194
    75th percentile:  4197.568145301786
    Max score:  4472.551202661819



```python
cross_val_score_calculation(ctb, X, y, cv=20)
```

    Min score:  1164.8667935668996
    25th percentile:  2369.2070783008776
    Mean score:  3320.8706243360784
    75th percentile:  3970.5845175550403
    Max score:  5353.482261036186


##### Stratified k-fold


```python
stratified_cross_val_results(ctb, X_train, y_train, 10)
```

    0:	learn: 0.8523575	total: 119ms	remaining: 1m 59s
    1:	learn: 0.8351395	total: 175ms	remaining: 1m 27s
    2:	learn: 0.8197881	total: 237ms	remaining: 1m 18s
    3:	learn: 0.8052796	total: 306ms	remaining: 1m 16s
    4:	learn: 0.7861025	total: 358ms	remaining: 1m 11s
    5:	learn: 0.7802464	total: 427ms	remaining: 1m 10s
    6:	learn: 0.7681655	total: 500ms	remaining: 1m 10s
    7:	learn: 0.7574203	total: 644ms	remaining: 1m 19s
    8:	learn: 0.7436521	total: 701ms	remaining: 1m 17s
    9:	learn: 0.7391258	total: 765ms	remaining: 1m 15s
    10:	learn: 0.7329496	total: 825ms	remaining: 1m 14s
    11:	learn: 0.7277381	total: 878ms	remaining: 1m 12s
    12:	learn: 0.7253701	total: 938ms	remaining: 1m 11s
    13:	learn: 0.7207735	total: 999ms	remaining: 1m 10s
    



```python
stratified_cross_val_results(ctb, X_train, y_train, 20)
```

    Stratified cross val Mean RMSE:  1067.6092366047692
    Prediction RMSE:  1168.5243545404567


## Test dataset


```python
def stratified_cross_val_results_test(reg, df,target , no_of_splits):
    rmse = []
    skf = StratifiedKFold(n_splits=no_of_splits, random_state=42)
    for train_index, test_index in skf.split(df, target):
        X1_train, X1_test = df.iloc[train_index], df.iloc[test_index]
        y1_train, y1_test = target.iloc[train_index], target.iloc[test_index]
        reg.fit(X1_train, y1_train)
        prediction = reg.predict(X1_test)
        score = np.sqrt(mean_squared_error(y1_test, prediction))#accuracy_score(prediction, y1_test)
        rmse.append(score)
    print("Stratified cross val Mean RMSE: ", np.mean(rmse))
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
    #np.sqrt(mean_squared_error(y_test, pred))
    #return reg, np.mean(rmse)
```


```python
test_df, X_t, y_t = username_freq(test)
```


```python
test_df_1 = ohe_cat_var(test_df, X_t)
```


```python
test_df_1.shape
```




    (141448, 14)




```python
X_t.shape
```




    (141448, 4)




```python
X.shape
```




    (330045, 14)




```python
test_df_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reputation</th>
      <th>Answers</th>
      <th>Views</th>
      <th>Freq</th>
      <th>Tag_a</th>
      <th>Tag_c</th>
      <th>Tag_h</th>
      <th>Tag_i</th>
      <th>Tag_j</th>
      <th>Tag_o</th>
      <th>Tag_p</th>
      <th>Tag_r</th>
      <th>Tag_s</th>
      <th>Tag_x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>172.0</td>
      <td>2.0</td>
      <td>1061.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>188.0</td>
      <td>2.0</td>
      <td>29625.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1616.0</td>
      <td>5.0</td>
      <td>52095.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>159561.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>441687.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df_1.Freq.value_counts()
```




    1     57651
    2     23452
    3     12444
    4      8000
    5      5195
          ...  
    51       51
    47       47
    46       46
    42       42
    40       40
    Name: Freq, Length: 99, dtype: int64



Predictions


```python
#model = DecisionTreeRegressor(random_state=42, max_depth=14)


#reg = DecisionTreeRegressor(random_state=42, max_depth=8,min_samples_leaf = 5,min_samples_split = 9)
#stratified_cross_val_results(reg, X_train, y_train, 10)
reg = DecisionTreeRegressor(random_state=42, max_depth=8,min_samples_leaf = 5,min_samples_split = 9)
reg.fit(X, y)
pred = reg.predict(test_df_1)
#print("Prediction RMSE: ", np.sqrt(mean_squared_error(y_test, pred)))
```


```python
rmse = []
skf = StratifiedKFold(n_splits=20, random_state=42)
for train_index, test_index in skf.split(X, y):
    X1_train, X1_test = X.iloc[train_index], X.iloc[test_index]
    y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X1_train, y1_train)
    prediction = model.predict(X1_test)
    score = np.sqrt(mean_squared_error(y1_test, prediction))#accuracy_score(prediction, y1_test)
    rmse.append(score)
print("Stratified cross val Mean RMSE: ", np.mean(rmse))
model.fit(X, y)
pred = model.predict(test_df_1)
```

    Stratified cross val Mean RMSE:  1192.7059743364216



```python
model.fit(X, y)
pred = model.predict(test_df_1)
```


```python
np.round(pred,0)
```




    array([15., 60., 60., ..., 15., 15., 15.])




```python
df1 = (pd.DataFrame(test["ID"]))
```


```python
df2 = pd.DataFrame(np.round(pred,0), columns=["Upvotes"])
```


```python
final = df1.join(df2, how = "outer")
```


```python
final[final <0]=0
```


```python
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Upvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>366953</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>71864</td>
      <td>60.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>141692</td>
      <td>60.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>316833</td>
      <td>95.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>440445</td>
      <td>175.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>141443</td>
      <td>47187</td>
      <td>41.0</td>
    </tr>
    <tr>
      <td>141444</td>
      <td>329126</td>
      <td>41.0</td>
    </tr>
    <tr>
      <td>141445</td>
      <td>282334</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>141446</td>
      <td>386629</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>141447</td>
      <td>107271</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
<p>141448 rows × 2 columns</p>
</div>




```python
final.to_csv("Submission.csv", index=False)
```
