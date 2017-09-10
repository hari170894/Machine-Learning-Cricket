import numpy as np
import pandas as pd
import iplHelper as util


#import package for mean square error
from sklearn.metrics import mean_squared_error

#import package for SVR 
from sklearn.svm import SVR

#import custom cross validation
#import crossvalidation as cv

#import custom featureselection
#import featureselection as fs

#import package for exhaustive grid search
from sklearn.grid_search import GridSearchCV

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


dataset = pd.read_csv("dataset-final.csv")
season_st = pd.read_csv("file_to_write.csv")

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
    #temp.append(dataset.iloc[i]['bowler_seasonal_economy'])
    #temp.append(dataset.iloc[i]['bowler_seasonal_strikerate'])
    match_id = dataset.iloc[i]['match']
    innings = dataset.iloc[i]['innings']
    match_records = season_st[season_st['match'] == match_id]
    ing_records = match_records[match_records['innings'] == innings]
    #print match_id, innings, ing_records.iloc[0]['seasonal_we']#, ing_records[0]['seasonal_we'], ing_records[0]['seasonal_we']
    temp.append(ing_records.iloc[0]['seasonal_we'])
    temp.append(ing_records.iloc[0]['seasonal_ws'])
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


print "Running"
x_array = np.empty((0,29))
y_array = []
for i in range(37, 17625):
    if(dataset.iloc[i]['over'] == 10):
        x_array = np.append(x_array, [read_record(i)], axis=0)
        y_array.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))



y_array = np.asarray(y_array)

x_test = np.empty((0,29))
y_test = []
for i in range(17625, len(dataset)):
    if(dataset.iloc[i]['over'] == 10):
        x_test = np.append(x_test, [read_record(i)], axis=0)
        y_test.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))
y_test = np.asarray(y_test)

        
CArray = [0.0005, 0.0006, 0.0007, 0.0008,0.0009, 0.001,0.002,0.003, 5, 5.5, 6, 6.5, 7]
CArray = [0.0001, 0.001,0.01, 0.1, 1, 2, 3, 4, 5, 6]
EArray = [0.2,0.4,0.6,0.8,1]
AlphaArray = [10000, 1000, 100 , 10 , 1, 0.1, 0.01, 0.001, 0.0001]


for i in range(0, len(AlphaArray)):
    #model = SVR(C= CArray[i], epsilon=0.5, kernel='linear')
    model = Ridge(alpha = AlphaArray[i])
    #model = Lasso(alpha = AlphaArray[i], max_iter=10000)
    model.fit(x_array, y_array)
    #print model.coef_
    print model.score(x_test, y_test)
    pred = model.predict(x_test)
    #for j in range(0,len(y_test)):
    #    print y_test[j], pred[j]
    print mean_squared_error(y_test, pred)


'''

model = SVR()
param_grid =  [{'C': CArray, 'kernel': ['linear'], 'epsilon' : EArray},]
grid_search = GridSearchCV(model, param_grid=param_grid)
grid_search.fit(x_array, y_array)

print grid_search.best_params_
print grid_search.best_score_

x_test = np.empty((0,29))
y_test = []
for i in range(19805, len(dataset)):
    if(dataset.iloc[i]['over'] == 10):
        x_test = np.append(x_test, [read_record(i)], axis=0)
        y_test.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))
y_test = np.asarray(y_test)

best_model = SVR(C=10, epsilon=1, kernel='rbf')
best_model.fit(x_array, y_array)

predictions = best_model.predict(x_test)

for i in range(0, len(predictions)):
    print predictions[i], y_test[i]

#'''

#print x_array
#print y_array

