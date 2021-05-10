
# coding: utf-8

# # Project Name : Bike Renting
# 
# Problem Statement : The objective of this Case is to Predication of bike rental count on daily based on the environmental and seasonal settings. 

# # Importing Standard Libraries

# In[1]:


# importing libraries for data pre-processing steps
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
get_ipython().run_line_magic('matplotlib', 'inline')

# importing libraries for model development, Performance, Evaluation and Optimization
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


# Setting up the working directory
os.chdir("E:\DataScienceEdwisor\PythonScripts")

#Checking the current working directory
os.getcwd()


# In[3]:


# Loading the dataset which is in '.CSV' format i.e; (Comma-Seperated-Values)
Bike_Actual = pd.read_csv('day.csv')

data = Bike_Actual.copy()


# #  Exploratory Data Analysis  

# In[4]:


# Lets see few observations of the dataset
data.head()


# In[5]:


# Lets Rename Column names with proper names
data = data.rename(columns = {'dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather',
                                        'hr':'hour','hum':'humidity','cnt':'count'})


# In[6]:


# Lets see the dimensions of the dataset
print(data.shape)


# In[7]:


# Information about the data and its data-types
data.info()


# In[8]:


# Lets see the descriptive statistics of the data
data.describe()


# In[9]:


# Lets Check the COlumn Names
data.columns


# In[10]:


# Lets seperate numeric and categorical columns
num_cols = ['temp','atemp','humidity','windspeed','casual','registered','count']
cat_cols = ['season','year','month','holiday','weekday','workingday','weather']


# In[11]:


# Lets check the counts in each categorical columns
for i in cat_cols:
    print(i)
    print(data[i].value_counts())
    print("*****************************")


# In[12]:


# Lets Check unique values in the whole data
for i in data.columns:
    print(i,' ======================>', len(data[i].value_counts()))


# <b> Missing value analysis <b>

# In[13]:


# Lets if there are any missing values
data.isnull().sum()


# <b> No missing values are there in the dataset <b> 

# <b> Data Visualization  <b>
#     
# <b> W.K.T target variable 'count' = 'casual' + 'registered' <b>

# In[14]:


# Average Count of all Bikes Distribution 
for i in cat_cols:
    fig = plt.figure(figsize=(15,6))
    fig = sns.barplot(x=i, y="count", data=data)
    fig.set(xlabel=i, ylabel='count')
    plt.suptitle("'{X}' average wise Total  '{Y}' ".format(X=i,Y='count'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# In[15]:


# Count of seasons with labels 1:spring, 2-summer, 3-fall, 4-winter
fig, ax = plt.subplots()
sns.barplot(data=data[['season','count']],
            x='season',
            y='count',
            ax=ax, palette = 'Set2')

plt.title('count by Season')
plt.ylabel('count')
plt.xlabel('Season')

tick_val=[0, 1, 2, 3]
tick_lab=['Spring','Summer','Fall',"winter"]
plt.xticks(tick_val, tick_lab)

plt.show()


# In[16]:


# Count of particular seasons with labels 1: clear, 2: mist, 3: light snow
fig, ax = plt.subplots()
sns.countplot(x= 'weather', hue = 'season' , data=data, palette = 'Set2')

plt.title('count by Weather')
plt.ylabel('count')
plt.xlabel('Weather')

tick_val=[0, 1, 2]
tick_lab=['clear','mist','light snow']
plt.xticks(tick_val, tick_lab)

plt.show()


# In[17]:


# Lets check the count of the target variable cnt and registered and casual
fig = plt.subplots(figsize=(7,5))
sns.boxplot(data=data[['count', 'casual', 'registered']], palette='Set2')


# In[18]:


# Lets see the distribution of the target variable cnt
data['count'].hist(figsize=(15,10), color='Blue', alpha=0.7)


# In[19]:


# Lets see the distribution of the target variable casual
data['casual'].hist(figsize=(15,10), color='Blue', alpha=0.7)


# In[20]:


# Lets see the distribution of the target variable registered
data['registered'].hist(figsize=(15,10), color='Blue', alpha=0.7)


# In[21]:


# Distribution of weather in all four different seasons
plot= sns.FacetGrid(data=data,
               col='season',
               row='weather',hue='season', palette='Set2')
plot.map(plt.hist,'count')

plt.subplots_adjust(top=1.0)
plot.fig.suptitle('count vs weather')

plot.set_xlabels('count')
plot.set_ylabels('Frequency')

plt.show()


# <b> Scatter Plot Analysis to See Relationship between Features for Numeric Columns VS Count Variable  <b>

# In[22]:


# Scatter Plot For Cheking the Variables Scatter VS Count
for i in ['temp','atemp','humidity','windspeed']:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="count", data=data)
    fig.set(xlabel=i, ylabel='count')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot ".format(X=i,Y='count'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# <b> Scatter Plot Analysis to See Relationship between Features for Numeric Columns VS Casual Variable  <b>

# In[23]:


# Scatter Plot For Cheking the Variables Scatter VS Casual
for i in ['temp','atemp','humidity','windspeed']:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="casual", data=data)
    fig.set(xlabel=i, ylabel='casual')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot ".format(X=i,Y='casual'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# <b> Scatter Plot Analysis to See Relationship between Features for Numeric Columns VS Registered Variable  <b>

# In[24]:


# Scatter Plot For Cheking the Variables Scatter VS Registered
for i in ['temp','atemp','humidity','windspeed']:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="registered", data=data)
    fig.set(xlabel=i, ylabel='registered')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot & Fitted Regression Line ".format(X=i,Y='registered'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# <b> Scatter Plot Analysis to See Relationship between Features for Categorical Columns <b>

# <b> All Categorical Features With respect to target variable Count <b> 

# In[25]:


# Scatter Plot For Cheking the Variables Scatter VS Count
for i in cat_cols:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="count", data=data,fit_reg=False)
    fig.set(xlabel=i, ylabel='count')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot ".format(X=i,Y='count'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# <b> All Categorical Features With respect to variable Casual <b> 

# In[26]:


# Scatter Plot For Cheking the Variables Scatter VS Casual
for i in cat_cols:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="casual", data=data,fit_reg=False)
    fig.set(xlabel=i, ylabel='casual')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot ".format(X=i,Y='casual'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# <b> All Categorical Features With respect to variable Registered <b> 

# In[27]:


# Scatter Plot For Cheking the Variables Scatter VS Registered
for i in cat_cols:
    fig = plt.figure(figsize=(15,6))
    fig = sns.lmplot(x=i, y="registered", data=data,fit_reg=False)
    fig.set(xlabel=i, ylabel='registered')
    plt.suptitle("'{X}' VS '{Y}' Scatter Plot ".format(X=i,Y='registered'),y = 1.02,fontsize=15)
    plt.tight_layout()
    plt.show()


# In[28]:


#Joint plot For all numeric column
for i in num_cols:
    fig = plt.figure(figsize=(10,7))
    fig = sns.jointplot(x=i, y="count", data=data)
    fig.set_axis_labels(xlabel=i,ylabel='count',fontsize=14)
    plt.suptitle("'{X}' VS '{Y}' Joint Plot".format(X=i,Y='count'),y = 1.02,fontsize=15)
    plt.show()


# In[29]:


# Jointplots For temp, atemp VS humidity and winsped
plot = sns.jointplot(x = 'temp' , y = 'atemp', data=data)
plot = sns.jointplot(x = 'temp' , y = 'humidity', data=data)
plot = sns.jointplot(x = 'temp' , y = 'windspeed', data=data)
plot = sns.jointplot(x = 'atemp' , y = 'humidity', data=data)
plot = sns.jointplot(x = 'atemp' , y = 'windspeed', data=data)
plot = sns.jointplot(x = 'humidity' , y = 'windspeed', data=data)
plt.subplots_adjust(top=1.0)
plt.show()


# In[30]:


# Lets see the Distribution in weather season wise 
fig = plt.figure()
fig = sns.countplot(x='weather', hue = 'season', data=data, palette='Set2' )


# <b> Outlier Analysis using Boxplot method <b>

# In[31]:


# Lets Check Outlier in Numeric Columns
for i in num_cols:
    plt.figure()
    sns.boxplot(data[i],palette="Blues")
    plt.title(i)
    plt.show()


# In[32]:


# Lets Check Outlier in Categorical Columns
for i in cat_cols:
    plt.figure()
    sns.boxplot(x=i, y='count', data=data)
    plt.title(i)
    plt.show()


# <b> As we can see most of the outliers are in continous / numeric variables itself Lets Detect and Remove Them <b>

# In[33]:


# Here we are not considering casual registered and count features 
# Detecting and revoing outliers and then replacing the values with nan
for i in ['temp','atemp','humidity','windspeed']:
    print(i)
    q75, q25 = np.percentile(data.loc[:,i], [75, 25])
    # iqr-Inter Quartile Range
    iqr = q75 - q25
    min = q25 - (iqr * 1.5)
    max = q75 + (iqr * 1.5)
    # Replace the values with np.nan    
    data.loc[data.loc[:,i] < min,i] = np.nan
    data.loc[data.loc[:,i] > max,i] = np.nan
    #Calculate missing value
    print('{numeric}==========>: {x} Missing_Values'.format(numeric = i, x = (data.loc[:,i].isnull().sum())))


# In[34]:


# Lets Use fillna method to impute values into the missing values using the mean method
data.humidity = data.humidity.fillna(data.humidity.mean())
data.windspeed = data.windspeed.fillna(data.windspeed.mean())


# <b> Fetaure Selection <b>

# <b> Correlation Analysis <b>

# In[35]:


# Generate the Correlation matrix 
corr = data[['temp','atemp','humidity','windspeed','casual', 'registered','count']].corr()

# Ploting using seaborn library
sns.heatmap(corr, annot=True, cmap="YlGnBu", linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,16)
plt.title("Heatmap analysis For All Numeric columns")
plt.show()


# <b> As we can see that 'temp' & 'atemp' features are hihgly positively correlated with each other and 'registered' & 'casual' are also highly correlated with each other and we can also see there is multicollinearity between 'registred' & 'count' also <b>

# In[36]:


# Lets analyze through pair plot on numeric columns
sns.pairplot(data[['temp','atemp','humidity','windspeed','casual', 'registered','count']],palette ="husl")
plt.tight_layout()
plt.show()


# <b> Lets Analyze Chi2-Square Test of Independence for Categorical Variables <b>

# In[37]:


# Null Hypothesis H0 = Variables are Not Independent
# ALternate Hypothesis H1 = Variables are Independent

cat_cols = ['season','year','month','holiday','weekday','workingday','weather']
# Creating all combinational pairs because we have different levels in all categorical columns
chi_pairs = [(i,j) for i in cat_cols for j in cat_cols ]

# Extracting all combinations for chi2-test
p_vals = []
for i in chi_pairs:
    if i[0] != i[1]:  
    # here Chi2-Square test compares two variables in contigency table
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(data[i[0]], data[i[1]]))
    
    # Lets round the decimal upto 3
        p_vals.append(p.round(3))
    else:
        p_vals.append('*')

p_vals = np.array(p_vals).reshape((7,7))
p_vals = pd.DataFrame(p_vals, index=cat_cols, columns=cat_cols)
# Here we are Printing only the P-Value 
print(p_vals)

# Here Chi2 - is the observed/actual value
# Here P -  is the pearson correlation value i.e: (p<0.05)
# Here DOF( Degrees of Freedom) = (no. of rows-1) - (no. of columns-1)
# Here EX - is the expected value
# Here Conclusion is that if the P-Value is less than (p<0.05) we reject Null Hypothesis saying that 2 values depend
# on each other else we accept alternate hypothesis saying that these 2 values are independent of each other


# <b> Anova Test <b>

# <b> Analysis of variance which is nothing but a numerical variable, in case of anova we use “1 categorical and 1 numeric variable”. <b>
# <b> Anova is a statistical technique used to compare the means of 2 or more groups of observations. <b>

# <b> Here We are doing Anova Test For 3 Numerical Vars i.e, ('count','casual','registered') VS all Categorical Variables because we know that target variable  count = casual + reistered <b>

# In[38]:


# Lets See the Analysis of Variance Between Count Numerical Variable and All categorical Variables First
aov_test_count = ols('count~season+year+month+holiday+weekday+workingday+weather', data=data).fit()
anova_table_count = sm.stats.anova_lm(aov_test_count, typ=1)
anova_table_count


# In[39]:


# Lets See the Analysis of Variance Between Casual Numerical Variable and All categorical Variables First
aov_test_casual = ols('casual~season+year+month+holiday+weekday+workingday+weather', data=data).fit()
anova_table_casual = sm.stats.anova_lm(aov_test_casual, typ=1)
anova_table_casual


# In[40]:


# Lets See the Analysis of Variance Between Registered Numerical Variable and All categorical Variables First
aov_test_registered = ols('registered~season+year+month+holiday+weekday+workingday+weather', data=data).fit()
anova_table_registered = sm.stats.anova_lm(aov_test_registered, typ=1)
anova_table_registered


# In[41]:


# Checking Features Importances
# Lets drop some of the columns
drop_cols = ['instant','datetime','casual','registered','count']

from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=300)
X = data.drop(columns=drop_cols)
y = data['count']
regressor.fit(X,y)
feat_imp = pd.DataFrame({'Features':data.drop(columns=drop_cols).columns,
                         'importance':regressor.feature_importances_})
feat_imp.sort_values(by = 'importance', ascending=False).reset_index(drop=True)


# <b> MultiCollinearity Check <b>

# \begin{equation*}
# (V.I.F = 1 / (1-R^2)
# \end{equation*}

# <b> Variance inflation factors measure how much the variance of the estimated regression coefficients are inflated as compared to when the predictor variables are not linearly related. It is used to explain how much multi-collinearity, correlation between predictors exists in an analysis.
# To interpret the Variance Inflation factors (VIF) =1 (Not correlated)
# If 1 < VIF < 5 (Moderately correlated)
# VIF >=5 (Highly Correlated) <b> 

# In[42]:


# For Numeric COlumns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
numeric_cols = add_constant(data[['temp','atemp','humidity','windspeed']]) 
VIF = pd.Series([vif(numeric_cols.values, i) for i in range (numeric_cols.shape[1])],
                 index = numeric_cols.columns)
VIF.round(2)


# In[43]:


# For Categorical Columns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
cat_names = add_constant(data[['season','year','month','holiday','workingday','weekday','weather']]) 
VIF = pd.Series([vif(cat_names.values, i) for i in range (cat_names.shape[1])],
                 index = cat_names.columns)
VIF.round(2)


# <b> Dimension Reduction <b>

# In[44]:


# Dropping the fetures highly correlated and not useful features 
data.drop(columns=['instant', 'datetime', 'holiday', 'atemp', 'casual', 'registered'], inplace=True)


# In[45]:


# Dimesnion of the dataset after dimension reduction
print(data.shape)


# In[46]:


# Lets Create Dummy Variables For season and weather columns
season_dv = pd.get_dummies(data['season'], drop_first=True, prefix='season')
data = pd.concat([data, season_dv],axis=1)
data = data.drop(columns = ['season'])
weather_dv = pd.get_dummies(data['weather'], drop_first=True, prefix= 'weather')
data = pd.concat([data, weather_dv],axis=1)
data = data.drop(columns= ['weather'])


# In[47]:


# lets See the Dataset
data.head()


# <b> Feature Scaling <b>

# In[48]:


#Checking distribution of data pandas inbuilt histogram visualization
names = ['temp','humidity','windspeed']
data[names].hist(figsize=(15,6), alpha=0.7)
plt.show()


# <b> Data is Uniformly distributed and already data is also scaled so we are not going to scale the data <b>

# <b> Sampling Data  Splitting the Data to Train and Test sets <b>

# In[49]:


X = data.drop(columns=['count'])
y = data['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# <b> Model Development <b>

# In[50]:


# Defining function to fit the model and predict the results
from sklearn.metrics import explained_variance_score
def reg_models(regression_models, features, comparison):
    regression_models.fit(features, comparison)
    y_pred = regression_models.predict(X_test)
    KFCV = cross_val_score(estimator=regression_models, X=features, y=comparison, cv=10,
                           scoring = 'explained_variance' )
    KVC_Performance = KFCV.mean()
    print("K-Fold Cross Validation Explained Variance")
    print('<=========================================>')
    print(KVC_Performance)
    print()
    print("Explained Variance on Train Data")
    print('<================================>')
    print(regression_models.score(features, comparison))
    print()
    print("Explained Variance on Test Data")
    print('<==============================>')
    print(regression_models.score(X_test, y_test))


# In[51]:


# Defining Function to Evaluate models on different error metrics
def eval_model(actual_vals, pdict_vals):
    print("Root Mean Squared Error --------->", round(np.sqrt(metrics.mean_squared_error(actual_vals,pdict_vals)), 2))
    print('<================================>')
    print(('Mean Absolute Percentage Error-->  {} % ').format(round(np.mean(np.abs((actual_vals - pdict_vals)/actual_vals))*100, 2)))
    print('<================================>')
    print("R2 Score ------------------------>", round(metrics.r2_score(actual_vals,pdict_vals), 2))


# <b> Linear Regression Model <b>

# In[52]:


# Linear Regression model variance explained
LR_Model = LinearRegression()
reg_models(LR_Model, X_train, y_train)


# In[53]:


# Predict new test cases Linear Regression model
LR_Predict = LR_Model.predict(X_test) 


# In[54]:


# Evaluation of linera Regression Model
eval_model(y_test, LR_Predict)


# <b> K Nearest Neighbor Model <b>

# In[55]:


# KNN model variance explained
KNN_Model = KNeighborsRegressor(n_neighbors=5)
reg_models(KNN_Model, X_train, y_train)


# In[56]:


# Predict new test case for KNN model
KNN_Predict = KNN_Model.predict(X_test)


# In[57]:


# Evaluation of KNN Model
eval_model(y_test, KNN_Predict)


# <b> Support Vector Regressor Model <b>

# In[58]:


# SVR model variance explained
SVR_Model = SVR()
reg_models(SVR_Model, X_train, y_train)


# In[59]:


# Predict new test case for SVR model
SVR_Predict = SVR_Model.predict(X_test)


# In[60]:


# Evaluation of SVR Model
eval_model(y_test, SVR_Predict)


# <b> Decision Tree Model <b>

# In[61]:


# Decision Tree model variance explained
DT_Model = DecisionTreeRegressor(max_depth=2, random_state=100)
reg_models(DT_Model, X_train, y_train)


# In[62]:


# Predict new test case for Decision Tree model
DT_Predict = DT_Model.predict(X_test)


# In[63]:


# Evaluation of Decision Tree Model
eval_model(y_test, DT_Predict)


# <b> Random Forest Model <b>

# In[64]:


# Random Forest Tree model variance explained
RF_Model = RandomForestRegressor(n_estimators=500, random_state=100)
reg_models(RF_Model, X_train, y_train)


# In[65]:


# Predict new test case for Random Forest model
RF_Predict = RF_Model.predict(X_test)


# In[66]:


# Evaluation of Random Forest Model
eval_model(y_test, RF_Predict)


# <b> XGB Regressor Model <b>

# In[67]:


# Xtreme Gradient Boosting Regressor model variance explained
XGB_Model = XGBRegressor()
reg_models(XGB_Model, X_train, y_train)


# In[68]:


# Predict new test case for XGB model
XGB_Predict = XGB_Model.predict(X_test)


# In[69]:


# Evaluation of XGB Model
eval_model(y_test, XGB_Predict)


# <b> Hyper Parameter Tuning & Optimization <b>

# <b>Lets Tune Both Random Forest Model and XGB Model and Then we will finalize the Outperformed Model from these both models<b>

# In[70]:


# Hyper Paramter Tuning XGB MOdel to Find Optimum parameters
#xgb_model = XGBRegressor()

#params = [{'n_estimators':[250,350,450,550], 'max_depth':[2,3,5,7],
#           'learning_rate':[0.01, 0.045, 0.05, 0.055, 0.1], 'gamma':[0, 0.001, 0.01, 0.03, 0.05],
#           'subsample':[1, 0.5, 0.7, 0.8, 0.9]}]

#grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv = 5,
#                           scoring = 'explained_variance', n_jobs=-1)

#grid_search = grid_search.fit(X_train, y_train)
#print('Best Score ====>', grid_search.best_score_)
#print('Best Params ===>', grid_search.best_params_)

# Best Score ====> 0.897321837387
# Best Params ===> {'gamma': 0, 'learning_rate': 0.045, 'max_depth': 3, 'n_estimators': 550, 'subsample': 0.5}


# <b> Tuned XGB Regressor Model <b>

# In[71]:


# Tuned Xtreme Gradient Boosting Regressor model variance explained

xgb_model = XGBRegressor(learning_rate=0.045, max_depth=3, n_estimators=550, 
                         gamma = 0, subsample = 0.5)

reg_models(xgb_model, X_train, y_train)


# In[72]:


# Predict new test case for Tuned XGB model
xgb_predict = xgb_model.predict(X_test)


# In[73]:


# Evaluation of Tuned XGB Model
eval_model(y_test, xgb_predict)


# <b> Tuned Random Forest Model <b>

# In[74]:


# Hyper Paramter Tuning Random Forest Model to Find Optimum parameters
#rf_model = RandomForestRegressor(random_state=1)
#params = [{'n_estimators':[200,300,400,500,600,700,800,1000],'max_features':['auto','sqrt','log2'],
#           'min_samples_split':[2,4,6],'max_depth':[2,4,6,8,10,12,14,16],'min_samples_leaf':[2,3,5]}]

#grid_search = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5,scoring = 'explained_variance', n_jobs=-1)

#grid_search = grid_search.fit(X_train, y_train)
#print('Best Score ====>', grid_search.best_score_)
#print('Best Params ===>', grid_search.best_params_)

# Best Score ====> 0.865267759445
# Best Params ===> {'max_depth': 12, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 800}


# In[75]:


# Tuned Random Forest model variance explained
rf_model = RandomForestRegressor(n_estimators=800, max_depth=12, max_features='sqrt', min_samples_leaf=2, 
                                 min_samples_split=2)

reg_models(rf_model, X_train, y_train)


# In[76]:


# Predict new test case for Random Forest model
rf_predict = rf_model.predict(X_test)


# In[77]:


# Evaluation of Random Forest Model
eval_model(y_test, rf_predict)


# In[78]:


# Lets Plot Scatter Graph for predicted values for tuned XGBRegressor Model
XGB_Regressor = XGBRegressor(random_state=1, learning_rate=0.045, max_depth=3, n_estimators=550, 
                         gamma = 0, subsample=0.5)
XGB_Regressor.fit(X_train, y_train)
y_pred = XGB_Regressor.predict(X_test)


# In[79]:


# Plotting Scatter plot for Tuned XGB Regressor for Actual and Predicted values
fig , ax = plt.subplots(figsize=(7,5))
ax.scatter(y_test, y_pred)
ax.plot([0,8000],[0,8000], 'r--', label = 'Perfect Prediction')
ax.legend()
plt.title("Scatter graph for Actual and Predicted")
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.tight_layout()
plt.show()


# In[80]:


# Saving results back to hard disk
# Setting up the working directory
#os.chdir("E:\DataScienceEdwisor\PROJECT-2\Python")

#data.to_csv("final_day.csv", index = False)

