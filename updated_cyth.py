import numpy as np
import pandas as pd
import iplHelper as util

import matplotlib.pyplot as plt
import random
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
from sklearn.neural_network import MLPRegressor

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

valuesRidgelist=[]
valuesLassoList=[]
valuesEnsemblelist=[]


for overval in range(1,20): 
    print "Running"+str(overval)
    x_array = np.empty((0,29))
    y_array = []
    for i in range(37, 19806):
        if(dataset.iloc[i]['over'] == overval):
            x_array = np.append(x_array, [read_record(i)], axis=0)
            y_array.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))

    y_array = np.asarray(y_array)

    x_val = np.empty((0,29))
    y_val = []
    for i in range(17625, 19806):
        if(dataset.iloc[i]['over'] == overval):
            x_val = np.append(x_val, [read_record(i)], axis=0)
            y_val.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))

    y_val=np.asarray(y_val)

    x_test = np.empty((0,29))
    y_test = []
    for i in range(19806, len(dataset)):
        if(dataset.iloc[i]['over'] == overval ):
            x_test = np.append(x_test, [read_record(i)], axis=0)
            y_test.append(float(util.get_run_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / ( float(util.get_ball_count_in_innings(dataset.iloc[i]['match'],dataset.iloc[i]['innings'])) / 6 ))
    y_test = np.asarray(y_test)

            
    CArray = [0.0005, 0.0006, 0.0007, 0.0008,0.0009, 0.001,0.002,0.003, 5, 5.5, 6, 6.5, 7]
    CArray = [0.0001, 0.001,0.01, 0.1, 1, 2, 3, 4, 5, 6]
    EArray = [0.2,0.4,0.6,0.8,1]
    AlphaArray = [10000, 1000, 100 , 10 , 1, 0.1, 0.01, 0.001, 0.0001]

    '''
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

    model=MLPRegressor(max_iter=100000000)
    model.fit(x_array,y_array)
    print model.score(x_test,y_test)
    '''
    AlphaArray=[767]
    CArray=[0.01]
    C=[0.001]
    EArray=[0.5]
    model1=SVR(C=C[0],epsilon=EArray[0],kernel='linear')
    model2=Ridge(alpha=AlphaArray[0])
    model3=Lasso(alpha=CArray[0], max_iter=100000)
    model1.fit(x_array,y_array)
    model2.fit(x_array,y_array)
    model3.fit(x_array,y_array)
    #print model1.score(x_test, y_test)
    #print model2.score(x_test, y_test)
    #print model3.score(x_test, y_test)
    p1=model1.predict(x_test)
    p2=model2.predict(x_test)
    p3=model3.predict(x_test)
    '''
    print mean_squared_error(p1,y_test)
    print mean_squared_error(p2,y_test)
    print mean_squared_error(p3,y_test)
    print mean_squared_error((p1+p2)/2,y_test)
    print mean_squared_error((p2+p3)/2,y_test)
    print mean_squared_error((p1+p3)/2,y_test)
    print mean_squared_error((p1+p2+p3)/3,y_test)
    print 'Mean differences'
    print np.average(np.absolute(p1-y_test))
    print np.average(np.absolute(p2-y_test))
    print np.average(np.absolute(p3-y_test))
    print np.average(np.absolute((p1+p2)/2-y_test))
    print np.average(np.absolute((p2+p3)/2-y_test))
    print np.average(np.absolute((p1+p3)/2-y_test))
    print np.average(np.absolute((p1+p2+p3)/3-y_test))
    print 'Max differences'
    print np.max(np.absolute(p1-y_test))
    print np.max(np.absolute(p2-y_test))
    print np.max(np.absolute(p3-y_test))
    print np.max(np.absolute((p1+p2)/2-y_test))
    print np.max(np.absolute((p2+p3)/2-y_test))
    print np.max(np.absolute((p1+p3)/2-y_test))
    print np.max(np.absolute((p1+p2+p3)/3-y_test))
    '''
    valuesRidgelist.append(mean_squared_error(p2,y_test))
    valuesLassoList.append(mean_squared_error(p3,y_test))
    valuesEnsemblelist.append(np.max(np.absolute((p2+p3)/2-y_test)))
    #z= (p1+p2)/2
    #print z[10],y_test[10]
print valuesRidgelist
print valuesLassoList
print valuesEnsemblelist

def createLineGraph(values,name):
    #initialize values
    accvalues=[]
    trainvalues=[]
    for i in range(len(values)):
        accvalues.append(20*(values[int(i)]))
    inds   = np.arange(len(accvalues))+1
    #labels =["Validation Accuracy","Training Loss"]

    #Plot a line graph
    plt.figure(random.randint(1,50000))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,accvalues,linewidth=1) #Plot the first series in red 

    #This plots the data
    plt.ylabel("Absolute Score Difference") #Y-axis label
    plt.xlabel("Overs") #X-axis label
    #plt.title("Average Predicted Score Error Over Time(Overs)") #Plot title
    #plt.xlim(-1,len(accvalues)+1) #set x axis range
    #plt.yscale('linear') #Set yaxis scale
    #plt.legend(labels,loc="best")

    #Make sure labels and titles are inside plot area
    #plt.tight_layout()

    #Save the chart
    plt.savefig("Figures/Line"+name+".jpeg")
def createLineGraph1(values,name):
    #initialize values
    accvalues=[]
    trainvalues=[]
    for i in range(len(values)):
        accvalues.append(20*(values[int(i)]))
    inds   = np.arange(len(accvalues))+1
    #labels =["Validation Accuracy","Training Loss"]

    #Plot a line graph
    plt.figure(random.randint(1,50000))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,accvalues,linewidth=1) #Plot the first series in red 

    #This plots the data
    plt.ylabel("Maximum Score Error") #Y-axis label
    plt.xlabel("Overs") #X-axis label
    #plt.title("Maximum Score Error Over Time(Overs)") #Plot title
    #plt.xlim(-1,len(accvalues)+1) #set x axis range
    #plt.yscale('linear') #Set yaxis scale
    #plt.legend(labels,loc="best")

    #Make sure labels and titles are inside plot area
    #plt.tight_layout()

    #Save the chart
    plt.savefig("Figures/Line"+name+".pdf")


createLineGraph(valuesRidgelist,"Ridge")
createLineGraph(valuesLassoList,"Lasso")
createLineGraph1(valuesEnsemblelist,"Ensemble")


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

