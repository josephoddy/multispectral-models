# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:41:39 2022

@author: Joe
"""

#%% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
import os
from plotnine import ggplot, aes, geom_boxplot, theme_bw, theme, geom_violin, xlab, ylab


#%% set directory
os.getcwd()
os.chdir('C:\\Users\\Joe\\OneDrive - Rothamsted Research\\PhD\\2020 to 2021 field trial\\TEC5 data')

#%% import data
data_full = pd.read_csv('220721 all spectral readings.csv')
data_full = pd.DataFrame(data_full)
data = data_full.iloc[:, 4:408]
data = data[data['ln.asn.average'].notna()]

#%% define predictor and response variables for asn and yield
X = data.iloc[:, 2:405]
y = data[["ln.asn.average"]]
yyield = data[["Grain85%"]]

#%% define cross-validation method for model testing
cv = RepeatedKFold(n_splits=5, n_repeats=10)

#%% Calculate MSE using cross-validation for asn model, adding one component at a time
mse = []

for i in np.arange(1, 20):
    pls = PLSRegression(n_components=i)
    score = -1*model_selection.cross_val_score(pls, X, y, cv=cv,
               scoring='neg_root_mean_squared_error').mean()
    mse.append(score)

#plot mse vs. number of components
plt.plot(mse)
plt.xlabel('Number of PLS Components')
plt.ylabel('MSE')
plt.title('Asn')

#%% Calculate MSE using cross-validation for yield model, adding one component at a time
mse = []

for i in np.arange(1, 20):
    pls = PLSRegression(n_components=i)
    score = -1*model_selection.cross_val_score(pls, X, yyield, cv=cv,
               scoring='neg_root_mean_squared_error').mean()
    mse.append(score)

#plot mse vs. number of components
plt.plot(mse)
plt.xlabel('Number of PLS Components')
plt.ylabel('MSE')
plt.title('Yield')

#%% Calculate Rsq using cross-validation for asn model, adding one component at a time
rsq = []

for i in np.arange(1, 20):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, X, y, cv=cv,
               scoring='r2').mean()
    rsq.append(score)

#plot R2 vs. number of components
plt.plot(rsq)
plt.xlabel('Number of PLS Components')
plt.ylabel('Rsq')
plt.title('hp')

#%% Calculate Rsq using cross-validation for yield model, adding one component at a time
rsq = []

for i in np.arange(1, 10):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, X, yyield, cv=cv,
               scoring='r2').mean()
    rsq.append(score)

#plot R2 vs. number of components
plt.plot(rsq)
plt.xlabel('Number of PLS Components')
plt.ylabel('Rsq')
plt.title('Yield')


#%% Repeat five fold cross validation 1000 times with selected number of PLS components for asn model
cv2 = RepeatedKFold(n_splits=5, n_repeats=1000)

pls = PLSRegression(n_components = 10)
asnscore = model_selection.cross_val_score(pls, X, y, cv=cv2, scoring='r2')

#%% Repeat five fold cross validation x number of times with selected number of PLS components for yield model
cv2 = RepeatedKFold(n_splits=5, n_repeats=1000)

yieldpls = PLSRegression(n_components = 3)
yieldscore = model_selection.cross_val_score(pls, X, yyield, cv=cv2, scoring='r2')

#%% extract accuracy scores from asn and yield models
asndf = pd.DataFrame(asnscore)
asndf.columns = ['Asn']
yielddf = pd.DataFrame(yieldscore)
yielddf.columns = ['Yield']
fulldata = pd.concat([asndf, yielddf], axis=1)

#%% reshape data
fulldata2 = fulldata.melt(var_name = 'variable', value_name = 'rsq')

#%% plot accuracy scores as boxplots for asn and yield
(ggplot(fulldata2, aes(x = 'variable', y = 'rsq'))
 + geom_violin(fill = "mediumseagreen") + geom_boxplot(outlier_colour = '', width = 0.25, color = "black", fill = "lavender")
 + theme_bw() + ylab("R-Squared") + xlab("")
 + theme(figure_size=(2, 4))
 )

#%% get means for asn and yield r squared results
print(fulldata2)
fulldata2.groupby('variable')['rsq'].mean()
