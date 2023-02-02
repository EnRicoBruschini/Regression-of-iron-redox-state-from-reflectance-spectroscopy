#!/usr/bin/env python
# coding: utf-8

# ### Import necessary modules and libraries

# In[1]:


import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import pprint
from scipy import stats
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn import neighbors


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor


# ### Define ancillary functions

# #### Set a wavelength range for a dataframe containing reflectance data

# In[2]:


# Define a fuction to select a sub-range of wavelengths for each dataframe to be used in the calculation
# Each dataframe needs to have the same WL range of the others
def df_bounds(wl_1, wl_2, df):
    
    
    b_df = df[(df.WL_nm >= wl_1) & (df.WL_nm <= wl_2)]
    
    # Reset the index column of the datframe
    b_df = b_df.reset_index()
    
    # Delete the first index column (to avoid indicization problems)
    b_df.pop('index')
    return b_df


# #### Slope calculation

# In[3]:


def slope(wl_1, wl_2, df):
    
    n_col = df.shape[1]
    
    x1 = wl_1
    x2 = wl_2
    
    
    sl = []
    
    for i in range(1, n_col):

    
        refl_i = df[(df.WL_nm == wl_1)].iloc[0, i]
        refl_f = df[(df.WL_nm == wl_2)].iloc[0, i]
        
        slope = (refl_f - refl_i) / (wl_2 - wl_1)
        sl.append(slope)
        
    return (sl)


# #### Average reflectance

# In[4]:


# The function calculates the average value of reflectance between two given wavelengths

def av_refl(wl_1, wl_2, df):
    
    
    b_df = df[(df.WL_nm >= wl_1) & (df.WL_nm <= wl_2)]
    averRefl = df.mean(axis=0)
    #averRefl = averRefl.iloc[: , 1:]
    return (averRefl)


# ### Import DB_info and DB_spectra

# #### Import information database

# In[6]:


# To red the xlsx files put them in the same folder of the present notebook otherwise define a path and pass it to
# read_excel function

df_glasses = pd.read_excel('ds02.xlsx')
df_glasses


# #### Import spectra database

# In[7]:



filename = 'ds03.xlsx'

df_carli = pd.read_excel(filename, sheet_name='Carli_2016')
df_cannon = pd.read_excel(filename, sheet_name='Cannon_2017')
df_pisello = pd.read_excel(filename, sheet_name='Pisello_2022')
df_stockW = pd.read_excel(filename, sheet_name='Stockstill_Well')
df_minitti = pd.read_excel(filename, sheet_name='Minitti_2002')
df_moroz = pd.read_excel(filename, sheet_name='Moroz_2009')
df_farrand = pd.read_excel(filename, sheet_name='Farrand_2016')
df_bruschini = pd.read_excel(filename, sheet_name='Bruschini_2022')


# #### Create sub-set of the dataframes

# In[8]:


df_carli_bounds   = df_bounds(350, 2400, df_carli)
df_cannon_bounds  = df_bounds(350, 2400, df_cannon).drop(columns="WL_nm")
df_pisello_bounds = df_bounds(350, 2400, df_pisello).drop(columns="WL_nm")
df_stockW_bounds  = df_bounds(350, 2400, df_stockW).drop(columns="WL_nm")
df_minitti_bounds = df_bounds(350, 2400, df_minitti).drop(columns="WL_nm")
df_moroz_bounds   = df_bounds(350, 2400, df_moroz).drop(columns="WL_nm")
df_farr_bounds    = df_bounds(350, 2400, df_farrand).drop(columns="WL_nm")
df_brus_bounds    = df_bounds(350, 2400, df_bruschini).drop(columns="WL_nm")


# #### Concatenates the dataframes

# In[9]:


# Concatenate the df for each dataset into a new dataframe called df

frames = [df_carli_bounds, df_cannon_bounds, df_pisello_bounds, df_stockW_bounds, df_minitti_bounds, df_moroz_bounds, df_farr_bounds, df_brus_bounds]



df = pd.concat(
    frames,
    axis=1,
    verify_integrity=True)


# In[10]:


df.shape


# In[11]:


df


# #### Slope(s) calculation

# In[12]:


# Calculate a few spectral slopes for later use

sl_400_600 = slope(400, 600, df)
sl_600_800 = slope(600, 800, df)
sl_900_1200 = slope(900, 1200, df)
sl_500_1000 = slope(500, 1000, df)


# ### Spectral binning

# In[13]:


## Bin reflectance data using the mean as statistic

# Define the number of bins
bins = 30

# Extract from the dataframe the column with the wavelengths --> x_wl is now  a series
x_wl = df.iloc[:, 0]

# Create an empty list to be populated in the following for loop
bin_sp = []

# Loop troughth the dataframe columns (spectra), bin them using the mean and store the results in the "bin_sp" variable
for i in range(1, df.shape[1]):
    df_n = df.iloc[:, i]
    bin_means, bin_edges, binnumber = stats.binned_statistic(x_wl, df_n, bins=bins)
    bin_sp.append(bin_means)


# In[14]:


# Create an empty list (names of the binned spectra columns) to be populated in the following for loop
binname = []

# Loop troughth the number of bins to create a name for each bin. Allocate the bin names in the variable "binname"
for j in range(1, bins+1):

    b_nms = 'bin_' + str(j)
    binname.append(b_nms)
    
    
# Create a new dataframe with the binned spectra    
df_bnsp = pd.DataFrame(bin_sp)

# Assign a name to each column
df_bnsp.columns = binname

# Show the results
df_bnsp


# ### Merge binned spectra to the DB_info

# In[15]:


# Add the bins of each spectrum to the dataframe df_glasses

frames_b = [df_glasses, df_bnsp]
df_glasses = pd.concat(
    frames_b,
    axis=1,
    verify_integrity=True)

df_glasses


# In[16]:


# Check the names of the df_glasses columns

df_glasses.columns.values


# In[17]:


# Check the dimensions of the df_glasses dataframe
df_glasses.shape


# In[18]:


# Get detailed information of the glasses dataframe
df_glasses.info()


# ### Clean the dataset and prepare it for the $Fe^{3+} / Fe_{TOT}$ prediction

# #### Drop the unnecessary columns

# In[19]:


# Some features were considered not useful for Fe3+/Fe_tot prediction and hence they were removed from the df
df_g2 = df_glasses.drop(df_glasses.iloc[:, 20:28],axis = 1)

# Get information of the new dataframe
df_g2.info()


# #### Add new features (columns)

# In[20]:


# Calculate new features and add them to the df

df_g2['SiAl'] = df_g2['SiO2'] + df_g2['Al2O3']
df_g2['CaAl'] = df_g2['CaO'] + df_g2['Al2O3']

df_g2.insert(52, 'Sl_500_1000', sl_500_1000)

df_g2.info()


# #### Drop the rows with ND values in the "Fe3_Fe_tot" column

# In[21]:


# Retrieve the indices of the rows without the Fe3+_Fetot ratio (Fe3_Fe_tot = ND)
ND_indcs = df_g2.index[df_g2['Fe3_Fe_tot'] == 'ND'].tolist()

# Store the retrieved data into a new dataframe for later use
no_Fe_ratio = df_g2[df_g2['Fe3_Fe_tot'] == 'ND']

# Drop the lines with the 'ND' values in the 'Fe3_Fe_tot' column of the original dataframe
df_g2.drop(ND_indcs, inplace=True)
df_g2.info()


# In[22]:


# Convert the 'Fe3_Fe_tot' column into float
df_g2['Fe3_Fe_tot'] = df_g2['Fe3_Fe_tot'].astype(float)
df_g2.info()


# In[23]:


# Remove the column with spectra names from the dataframe
sp_names = df_g2.pop('sptr_name')


# ### Exploratory Data Analysis (EDA)

# In[24]:


def multiplot(data, features, plottype, nrows, ncols, figsize, y=None, colorize=False):
    n=0
    plt.figure(1)
    fig, axes=plt.subplots(nrows, ncols, figsize=figsize)
    
    if colorize:
        colors=sns.color_palette(n_colors=(nrows*ncols))
    else:
        colors=[None]*(nrows*ncols)
    
    for row in range(ncols):
        for col in range(nrows):
            
            if plottype=='regplot':
                if y==None:
                    raise ValueError('y value is needed with regplot type')
                
                sns.regplot(data=data, x=features[n],y=y,ax=axes[row,col],color=colors[n])
                correlation=np.corrcoef(data[features[n]],data[y])[0,1]
                axes[row,col].set_title('Correlation {:.2f}'.format(correlation))
                
            elif plottype=='distplot':
                sns.distplot(a=data[features[n]],ax=axes[row,col],color=colors[n])
            
    
            n+=1
        
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()


feats = ["FeO","MgO","Fe3_Fe_tot","SiO2"]

# multiplot(data = df_g2,features = feats,plottype = "distplot",nrows = 2, ncols = 2,
#           figsize = (10,6),y = "Bin_15", colorize = True);

multiplot(data = df_g2,features = feats,plottype = "distplot",nrows = 2, ncols = 2,
          figsize = (10,6),y = "Bin_1", colorize = True);


# These are hystograms to show chemical information about the samples in the DB. Solid lines are PDF (Partial Density Function)

# In[25]:


# Scatter plots FeO vs MnO - shades represent SiO2 content of the samples
ax2 = df_g2.plot.scatter(x='FeO',
                       y='MnO',
                       c='SiO2',
                       colormap='plasma')


# In[26]:


# Scatter plots MgO vs MnO - shades represent SiO2 content of the samples
ax3 = df_g2.plot.scatter(x='MgO',
                       y='MnO',
                       c='SiO2',
                       colormap='plasma')


# In[27]:


# Scatter plots SiO2 vs MnO - shades represent Al2O3 content of the samples
ax3 = df_g2.plot.scatter(x='SiO2',
                       y='MnO',
                       c='Al2O3',
                       colormap='plasma')


# In[28]:


# Scatter plots Al2O3 vs SiO2 - shades represent grainsize
ax4 = df_g2.plot.scatter(x='Al2O3',
                       y='SiO2',
                       c='GS_um',
                       colormap='plasma')


# In[29]:


# Scatter plots SiO2 vs CaO+Al2O3 - shades represent K2O content of the samples
ax5 = df_g2.plot.scatter(x='SiO2',
                       y='CaAl',
                       c='K2O',
                       colormap='plasma')


# In[30]:


ax6 = df_g2.plot.scatter(x='avrg_refl',
                       y='FeO+TiO2',
                       c='GS_um',
                       colormap='plasma')


# In[31]:


# Select only the data with GS < 63um and plot the results FeO+TiO2 vs avrg_refl vs slope

df_fine = df_g2[(df_g2.GS_um <= 65)]
ax8 = df_fine.plot.scatter(x='FeO+TiO2',
                       y='avrg_refl',
                       c='GS_um',
                       colormap='plasma')


# In[37]:


# This command outputs an xlsx file with all the data of the DB
df_g2.to_excel("output.xlsx") 


# ### Build the ML models

# #### Define the feature matrix and the label vector

# In[32]:


# 'Labels' vector
y = df_g2.pop('Fe3_Fe_tot')

# 'Features' matrix
X = df_g2.to_numpy()

# 'Labels' vector
y = y.to_numpy()


# #### Test and train splitting

# In[33]:


# Test and train splitting
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.20, random_state=16)


# In[34]:


y_test


# #### Features scaling

# In[35]:


scaler  = MinMaxScaler()

X_tr_sc = scaler.fit_transform(X_train)
X_tr    = pd.DataFrame(X_tr_sc)

X_ts_sc = scaler.fit_transform(X_test)
X_ts    = pd.DataFrame(X_ts_sc)


# #### Random forest

# In[36]:


# Define the values to be put in the grid
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 50)]
max_features = [30]
max_depth = [int(x) for x in np.linspace(5, 30, num = 20)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [6, 8, 10, 12]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[37]:


# Define the model to work with
rf = RandomForestRegressor()

# The hyperparameters are optimized through a randomized search (less computationally intensive)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 4, verbose=2, random_state=16, n_jobs = -1)

# Fit the model
rf_random.fit(X_tr_sc, y_train)


# In[38]:


# Output the best parameters fit by the model
rf_random.best_params_


# In[39]:


# Feed the model with the best-fit parameters and fir the training set
selected_rf=RandomForestRegressor(n_estimators=41, min_samples_split=5, min_samples_leaf=6, max_features = 30, max_depth=10)
selected_rf.fit(X_tr_sc,y_train)

# Predict the values according to the optimed model and calculate the metrics (MAE, MSE, RMSE)
rf_preds=selected_rf.predict(X_ts_sc)
mean_absolute_error(y_test,rf_preds),mean_squared_error(y_test,rf_preds),np.sqrt(mean_squared_error(y_test,rf_preds)),(y_test,rf_preds)


# In[40]:


# Print the test set
y_test


# In[41]:


# Print the predictions
rf_preds


# In[42]:


# Creata a scatter plot of observed vs predicted values
plt.scatter(rf_preds, y_test)

x_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
y_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

plt.plot(x_l, y_l)


# In[43]:


# Create a df with the y-test set ('Actual') and the predictions ('Predicted')
df_preds=pd.DataFrame({'Actual':y_test, 'Predicted':rf_preds})
df_preds


# In[44]:


# Display the results with an histogram and superimposed PDF 
import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y_test, hist=True, color="r", label="Actual Value")
sns.distplot(rf_preds, hist=True, color="b", label="Fitted Values" , ax=ax)
ax.legend()


plt.title('actual vs fitted values')


plt.show()
plt.close()


# In[45]:


# Display the R2 score
r2_score(y_test, rf_preds)


# #### Gradient boosting

# In[46]:


params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
}


# In[47]:


from sklearn import ensemble
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train,y_train)


# In[48]:


mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[49]:


mae = mean_absolute_error(y_test,reg.predict(X_test))
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))


# In[50]:


GB_preds = reg.predict(X_test)

plt.scatter(GB_preds, y_test)

x_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
y_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
plt.plot(x_l, y_l)


# In[51]:


df_GBpreds=pd.DataFrame({'Actual':y_test, 'Predicted':GB_preds})
df_GBpreds


# In[52]:


import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y_test, hist=True, color="r", label="Actual Value")
sns.distplot(GB_preds, hist=True, color="b", label="Fitted Values" , ax=ax)
ax.legend()


plt.title('actual vs fitted values')


plt.show()
plt.close()


# In[53]:


selected_GB=RandomForestRegressor(n_estimators=191, min_samples_split=5, min_samples_leaf=6, max_features = 20, max_depth=23)
selected_rf.fit(X_tr_sc,y_train)
rf_preds=selected_rf.predict(X_ts_sc)
mean_absolute_error(y_test,rf_preds),mean_squared_error(y_test,rf_preds),np.sqrt(mean_squared_error(y_test,rf_preds)),r2_score(y_test,rf_preds)


# #### Support vector regressor

# In[56]:


#score_fnc=mean_squared_error

score_fnc=mean_absolute_error

param_grid = [
                #{'kernel':['linear'],
                # 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]},

                {'kernel':['poly'],
                 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                 'degree':[2,3,5,7,9,10,11,12],
                 'gamma' : [1e-1, 1, 5, 10],
                },

                {'kernel':['rbf'],
                 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                 'gamma' : [1e-1, 1, 10, 100, 1000]}
            
            ]
cv=KFold(n_splits=5,shuffle=True,random_state=0)
svm=SVR()

model_selection=GridSearchCV(svm,param_grid=param_grid,scoring=make_scorer(score_fnc),cv=cv)


model_selection.fit(X_tr_sc,y_train)
model_selection.best_estimator_


# In[113]:


score_fnc=mean_squared_error
param_grid = [
                #{'kernel':['linear'],
                # 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]},

                {'kernel':['poly'],
                 'C':[0.01,0.05,0.1,0.5,1,1.5,2,5,10,15,20,30,50,100],
                 'degree':[2,3,5,7,9,10,11,12],
                 'gamma' : [1e-1, 1, 5, 10],
                },

                {'kernel':['rbf'],
                 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                 'gamma' : [1e-1, 1, 10, 100, 1000]}
            
            ]
cv=KFold(n_splits=5,shuffle=True,random_state=0)
svm=SVR()

model_selection=GridSearchCV(svm,param_grid=param_grid,scoring=make_scorer(score_fnc),cv=cv)


model_selection.fit(X_tr_sc,y_train)
model_selection.best_estimator_


# In[ ]:




model_selection.best_estimator_
# In[ ]:





# In[57]:


selected_svm=SVR(kernel='poly',gamma=5,degree=12,C=0.001)
selected_svm.fit(X_tr_sc,y_train)
svm_preds=selected_svm.predict(X_ts_sc)


# In[58]:


mean_absolute_error(y_test,svm_preds),mean_squared_error(y_test,svm_preds),np.sqrt(mean_squared_error(y_test,svm_preds))


# In[59]:


selected_svm=SVR(kernel='poly',C=0.1, gamma = 5, degree=2)
selected_svm.fit(X_tr_sc,y_train)
svm_preds=selected_svm.predict(X_ts_sc)


# In[60]:


y_test


# In[61]:


svm_preds


# In[62]:


plt.scatter(svm_preds, y_test)

x_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
y_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.plot(x_l, y_l)


# In[63]:


import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y_test, hist=True, color="r", label="Actual Value")
sns.distplot(svm_preds, hist=True, color="b", label="Fitted Values" , ax=ax)
ax.legend()


plt.title('actual vs fitted values')


plt.show()
plt.close()


# In[123]:


df_preds=pd.DataFrame({'Actual':y_test, 'Predicted':svm_preds})
df_preds


# In[ ]:





# In[ ]:





# In[ ]:


######### SVR - 2ND TEST ########


# In[60]:


regressor = SVR(kernel='rbf')
regressor.fit(X_tr_sc,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Prediction of new values

# #### Scale the dataframe containing the values to be predicted

# In[54]:


# Delete the Fe3_Fe_tot column (all ND values)
no_Fe_ratio.drop(columns="Fe3_Fe_tot", inplace=True)

sp_names_regr = no_Fe_ratio.pop('sptr_name')

# 'Features' matrix for the "regression" dataset
X_regr = no_Fe_ratio.to_numpy()


# In[55]:


sp_names_regr


# In[43]:


no_Fe_ratio.info()


# In[56]:


# Regress the values of Fe3+/Fe_TOT using the gradient boosting model
regressed_Fe_rat = reg.predict(no_Fe_ratio)


# In[57]:


regressed_Fe_rat


# In[58]:


no_Fe_ratio


# In[ ]:





# In[43]:


sp_names_regr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


# To use this function the module stats has to be imported
def bin_df(df, bins):
    x_wl = df.iloc[:, 0]
    df_n = df.iloc[:, 1]
    bin_means, bin_edges, binnumber = stats.binned_statistic(x_wl, df_n, bins=bins)
    return bin_means, bin_edges, binnumber
    
    
a = bin_df(df, 30)
# plt.plot(x, yhat, 'r-', label='smoothed_refl')
# plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
#            label='binned_refl')

x = df.iloc[:, 0]
y = df.iloc[:, 1]


plt.plot(x, y, 'r-', label='spectrum')
plt.hlines(a[0], a[1][:-1], a[1][1:], colors='g', lw=2,
            label='binned_refl')


# In[ ]:




