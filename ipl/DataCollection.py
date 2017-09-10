import pandas
import numpy as np 

data=pandas.read_csv('deliveries.csv')
#print data.head(3)
data['bowler_penalty_runs']=data['wide_runs']+data['noball_runs']
data['team_penalty_runs']=data['bye_runs']+data['legbye_runs']+data['penalty_runs']
del data['wide_runs']
del data['noball_runs']
del data['bye_runs']
del data['legbye_runs']
del data['penalty_runs']
#print data.columns.values
matchdata= data.groupby(['match_id','batting_team','over'])
#print matchdata.head()
#print matchdata.columns.values
print matchdata.sum()[['batsman_runs','bowler_penalty_runs','team_penalty_runs','total_runs']]