import numpy as np
import pandas as pd
import iplHelper as util


#import package for mean square error
from sklearn.metrics import mean_squared_error

#import package for SVR 
#from sklearn.svm import SVR
from sklearn.linear_model import Ridge
#import custom cross validation
#import crossvalidation as cv

#import custom featureselection
#import featureselection as fs

#import package for exhaustive grid search
from sklearn.grid_search import GridSearchCV

from sklearn import cross_validation


dataset = pd.read_csv("dataset-final.csv")


def read_record(i):
    temp = []
    temp.append(dataset.iloc[i]['toss'])
    temp.append(dataset.iloc[i]['home/away'])
    temp.append(dataset.iloc[i]['over'])
    temp.append(dataset.iloc[i]['runs'])
    temp.append(dataset.iloc[i]['wicket'])
    temp.append(dataset.iloc[i]['batsman_runs'])
    temp.append(dataset.iloc[i]['non_striker_runs'])
    temp.append(dataset.iloc[i]['batsman_strikerate'])
    temp.append(dataset.iloc[i]['non_striker_strikerate'])
    temp.append(dataset.iloc[i]['seasonal_batsman_runs'])
    temp.append(dataset.iloc[i]['seasonal_non_striker_runs'])
    temp.append(dataset.iloc[i]['seasonal_batsman_average'])
    temp.append(dataset.iloc[i]['seasonal_non_striker_average'])
    temp.append(dataset.iloc[i]['seasonal_batsman_strikerate'])
    temp.append(dataset.iloc[i]['seasonal_non_striker_strikerate'])
    temp.append(dataset.iloc[i]['bowler_seasonal_economy'])
    temp.append(dataset.iloc[i]['bowler_seasonal_strikerate'])
    temp.append(dataset.iloc[i]['recent_runrate'])
    temp.append(dataset.iloc[i]['top_bowler1_curr'])
    temp.append(dataset.iloc[i]['top_bowler1_prev'])
    temp.append(dataset.iloc[i]['wick1_curr'])
    temp.append(dataset.iloc[i]['top_bowler2_curr'])
    temp.append(dataset.iloc[i]['top_bowler2_prev'])
    temp.append(dataset.iloc[i]['wick2_curr'])
    temp.append(dataset.iloc[i]['top_bowler3_curr'])
    temp.append(dataset.iloc[i]['top_bowler3_prev'])
    temp.append(dataset.iloc[i]['wick3_curr'])
    temp.append(dataset.iloc[i]['wickets'])
    temp.append(dataset.iloc[i]['run-rate'])
    return temp

x_array = np.empty((0,29))
y_array = []
for i in range(0, 19805):
    x_array = np.append(x_array, [read_record(i)], axis=0)
    y_array.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))
y_array = np.asarray(y_array)
''' 
CArray = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
EArray = [0.2,0.4,0.6,0.8,1]
model = SVR()
param_grid =  [{'C': CArray, 'kernel': ['rbf'], 'epsilon' : EArray},]
grid_search = GridSearchCV(model, param_grid=param_grid)
grid_search.fit(x_array, y_array)

print grid_search.best_params_
print grid_search.best_score_
'''
print "FIN"

x_test = np.empty((0,29))
y_test = []
for i in range(19805, len(dataset)):
    x_test = np.append(x_test, [read_record(i)], axis=0)
    y_test.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))
y_test = np.asarray(y_test)

print "BEST MODEL"
best_model = Ridge(alpha=787)
best_model.fit(x_array, y_array)
trianpred=best_model.predict(x_array)
predictions = best_model.predict(x_test)
print best_model.score(x_array,y_array)
print 'TEST'
print best_model.score(x_test,y_test)
for i in range(0, len(predictions)):
    print predictions[i], y_test[i]

x_array=np.array(x_array)
y_array=np.array(y_array).reshape((y_array.shape[0],1))
x_test=np.array(x_test)
y_test=np.array(y_test)
trianpred=np.array(trianpred).reshape((trianpred.shape[0],1))
print x_array.shape
print y_array.shape
x=np.concatenate((x_array,y_array),axis=1)
x=np.concatenate((x,trianpred),axis=1)
print x[0]
'''  

#print x_array
#print y_array

'''