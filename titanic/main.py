# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:07:40 2019

@author: aj4g2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.base import TransformerMixin
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import graphviz
from sklearn.tree import export_graphviz
import csv


def plot_histgrams(x,y):
    fig = plt.figure('Feature Exploration')
    fig.clf()
    axa = fig.subplots(2,4).flatten()
    for i,col in enumerate(x.columns):
        ax = axa[i]
        ax.cla()
        ax.set_title(col)
        if not col in catVars:
            ax.hist((x[col].iloc[y==1]).dropna(),color='green',alpha=0.5,label='Survived',weights=np.ones(len((x[col].iloc[y==1]).dropna()))/len((x[col].iloc[y==1]).dropna()))
            ax.hist((x[col].iloc[y==0]).dropna(),color='blue',alpha=0.5,label='Not Survived',weights=np.ones(len((x[col].iloc[y==0]).dropna()))/len((x[col].iloc[y==0]).dropna()))
            ax.legend(loc='upper right')    
        else:
            (x[col].iloc[y==1].value_counts(sort=False)/sum(y==1)).plot(kind='bar',ax=ax,color='green',alpha=0.5,label='Survived')
            (x[col].iloc[y==0].value_counts(sort=False)/sum(y==0)).plot(kind='bar',ax=ax,color='blue',alpha=0.5,label='Not Survived')
            ax.legend(loc='upper right')    


'''From stack overflow'''
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def transform_variables(x,catVars):
    column_trans = make_column_transformer(
            (OneHotEncoder(),catVars),
            remainder = StandardScaler() )
    xt = column_trans.fit_transform(x)
    col_names_trans = []
    for transformers in column_trans.transformers_:
            for ci,cols in enumerate(transformers[2]):
                if transformers[0] == 'onehotencoder':
                    for cats in transformers[1].categories_[ci]:
                        col_names_trans.append((cols+'_'+str(cats)))
                else:
                    col_names_trans.append(x.columns[cols])
                    
    xt = pd.DataFrame(xt,columns=col_names_trans)
    return xt,column_trans

class feature_selection:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def rfe(self,method = 'svm',kernel='linear'):
        '''Rank the features basd on  recurive features elimination '''
        if method == 'svm':
            estimator = SVC(kernel=kernel)
        elif method == 'logisticRegression':
            estimator = LogisticRegression(solver='lbfgs')
        selector = RFE(estimator, 1, step=1,verbose=0)
        selector = selector.fit(self.X, self.y)
        return (selector.ranking_-1)

    def rank_features(self,ranking,featNames):
        '''Return the feature names based in input numerical ranking'''
        return [featNames[i] for i in ranking]

def crossValidate(X,y,k,model='svm',kernel='linear'):
    acc = []
    if model == 'svm':
        classifier = SVC(kernel=kernel,gamma='scale')    
    elif model == 'neuralNet':
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8,4), random_state=1)
    elif model == 'logisticRegression':
        classifier = LogisticRegression(solver='lbfgs')
    kf = KFold(n_splits=k,shuffle=True)
    for train_index, test_index in kf.split(X):
        classifier.fit(X[train_index,],y[train_index,])
        acc.append(classifier.score(X[test_index],y[test_index]))
    return acc

def crossValidate_perFeature(X,y,ranking,k=3,model='svm',kernel='linear',ax=plt.axes(),color='red'):
    acc = []
    for i,r in enumerate(ranking):
        acc.append(np.mean(crossValidate(np.asarray(X[ranking[:i+1]]),y,k,model=model,kernel=kernel)))
    ax.plot(ranking,acc,color=color,marker='.',linestyle='-')
    ax.set_title(model)    
    return acc

def performGreedyClassification(X,y,ranking_svm,ranking_lr):
    fig = plt.figure('Accuracy')
    fig.clf()
    ax = fig.subplots(4,1)
    acc_svm = crossValidate_perFeature(X,y,ranking_svm,k=3,model='svm',kernel='linear',ax=ax[0],color='red')
    acc_lr = crossValidate_perFeature(X,y,ranking_lr,k=3,model='logisticRegression',ax=ax[1],color='red')
    acc_nn = crossValidate_perFeature(X,y,ranking_lr,k=3,model='neuralNet',ax=ax[2],color='red')
    acc_svm2 = crossValidate_perFeature(X,y,ranking_svm,k=3,model='svm',kernel='rbf',ax=ax[3],color='red')


def draw_decision_tree(x,y,catVars):
    '''Draw decision tree with original features (before normalization and one hot coding)'''
    xtn = x.copy()
    for var in catVars:
        for i,name in enumerate(xtn[var].unique()):
            xtn.loc[xtn[var]==name,var] = i
            
    clf = DecisionTreeClassifier(max_depth=4,min_samples_split=10,criterion='entropy')
    clf.fit(xtn,y)
    #tree.plot_tree(clf.fit(xtn,y))
    dot_data = export_graphviz(clf, out_file=None, 
                          feature_names=list(xtn.columns),  
                          class_names=['Not Survived','Survived'],  
                          filled=True, rounded=True,  
                          special_characters=True)
    graph = graphviz.Source(dot_data)
    graph
    graph.render("titanic")
    

df = pd.read_csv(r'D:\Work\kaggleProjects\titanic\data\train.csv')
df.set_index(keys='PassengerId',drop=True,inplace=True)
y = np.asarray(df['Survived'])
x = df.drop(columns=['Survived','Name','Ticket','Cabin','SibSp','Parch'])
catVars = ['Sex','Embarked','Pclass']

#plot_histgrams(x,y)

'''Feature ranking with SVM and Logistic Regression'''
x_im = DataFrameImputer().fit_transform(x)
x_im_t,column_trans = transform_variables(x_im,catVars)
featSel = feature_selection(x_im_t,y)
ranking_svm = featSel.rank_features(featSel.rfe(method='svm',kernel='linear'),x_im_t.columns)
ranking_lr = featSel.rank_features(featSel.rfe(method='logisticRegression'),x_im_t.columns)

'''Compute accuracy using multiple methods by increamentally adding features according to the ranking'''
performGreedyClassification(x_im_t,y,ranking_svm,ranking_lr)

'''Draw decision tree with original features (before normalization and one hot coding)'''
draw_decision_tree(x_im,y,catVars)

'''perform_gridSearch_for_all'''

pipe_lr = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',LogisticRegression(solver='lbfgs'))])
param_grid = {'clf__C':[0.1,1,10,100]}
grid_search = GridSearchCV(pipe_lr, param_grid=param_grid,cv=3)
clf_lr = grid_search.fit(x_im,y)
print('Best Score for Logistic Regression = ',clf_lr.best_score_,' for ',clf_lr.best_params_)

pipe_svm = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',SVC(gamma='scale'))])
param_grid = {'clf__C':[0.1,1,10,100],'clf__kernel':['linear','rbf']}
grid_search = GridSearchCV(pipe_svm, param_grid=param_grid,cv=3)
clf_svm = grid_search.fit(x_im,y)
print('Best Score for SVM = ',clf_svm.best_score_,' for ',clf_svm.best_params_)


pipe_nn = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1))])
param_grid = {'clf__hidden_layer_sizes':[(8,4),(4,2)]}
grid_search = GridSearchCV(pipe_nn, param_grid=param_grid,cv=3)
clf_nn = grid_search.fit(x_im,y)
print('Best Score for Neural Net = ',clf_nn.best_score_,' for ',clf_nn.best_params_)


xtn = x_im.copy()
for var in catVars:
    for i,name in enumerate(xtn[var].unique()):
        xtn.loc[xtn[var]==name,var] = i
param_grid = {'max_depth':[2,4,6,8,10,16],'min_samples_split':[2,5,10,15],'criterion':['entropy','gini']}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid,cv=3)
clf_dt = grid_search.fit(xtn,y)
print('Best Score for Decision Tree = ',clf_dt.best_score_,' for ',clf_dt.best_params_)

pipe_dt = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',DecisionTreeClassifier())])
param_grid = {'clf__max_depth':[2,4,6,8,10,16],'clf__min_samples_split':[2,5,10,15],'clf__criterion':['entropy','gini']}
grid_search = GridSearchCV(pipe_dt, param_grid=param_grid,cv=3)
clf_dt2 = grid_search.fit(x_im,y)
print('Best Score for Decision Tree with one Hot & Norm = ',clf_dt2.best_score_,' for ',clf_dt2.best_params_)


eclf1 = VotingClassifier(estimators=[
        ('lr', clf_lr.best_estimator_[1]), ('svm', clf_svm.best_estimator_[1]), ('nn', clf_nn.best_estimator_[1]), ('dt', clf_dt2.best_estimator_[1])], voting='hard')

pipe_en = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',eclf1)])

print('Ensemble of LR, SVM, NN & DT Accuracy = ',np.mean(cross_val_score(pipe_en, x_im, y, cv=3)))

'''Best is svm'''
pipe_best = Pipeline(steps=[('preprocessor',column_trans),
              ('clf',clf_svm.best_estimator_[1])])
q = pipe_best.fit(x_im,y)
df_test = pd.read_csv(r'D:\Work\kaggleProjects\titanic\data\test.csv')
df_test.set_index(keys='PassengerId',drop=True,inplace=True)
x_test = df_test.drop(columns=['Name','Ticket','Cabin'])
x_test_im = DataFrameImputer().fit_transform(x_test)
y_test = pipe_best.predict(x_test_im)
q = x_test_im.index.values

with open('y_test.csv', mode='w',newline='') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerow(['PassengerId','Survived'])
    [wr.writerow([x_test_im.index.values[i],y_test[i]]) for i in range(len(y_test))]

print('Percentage of Male Survivals in training data = ',  (sum((y==1) & (list(x_im.Sex=='male'))))/(sum(x_im.Sex=='male')))    
print('Percentage of Female Survivals in training data = ',  (sum((y==1) & (list(x_im.Sex=='female'))))/(sum(x_im.Sex=='female')))    

print('Percentage of Male Survivals in testing data = ',  (sum((y_test==1) & (list(x_test_im.Sex=='male'))))/(sum(x_test_im.Sex=='male')))    
print('Percentage of Female Survivals in testing data = ',  (sum((y_test==1) & (list(x_test_im.Sex=='female'))))/(sum(x_test_im.Sex=='female')))    

y_train_pred = pipe_best.predict(x_im)
print('Accuracy on training data = ',sum(y_train_pred==y)/len(y))






    