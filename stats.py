import numpy as np
import pandas as pd

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("del-modified.csv")

# Match utility functions
def get_runs_in_over(match_id, innings, overid):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    over_balls = innings_balls[innings_balls["over"] == overid]
    runs = over_balls["total_runs"]
    return runs.sum()

def get_runs_by_bowler_in_match(match_id, innings, bowler):
    over_list = get_over_list_by_bowler(match_id, bowler, 20)
    count = 0
    for i in range(0,len(over_list)):
        count += get_runs_in_over(match_id, innings, over_list[i])
    return count

def get_wickets_by_bowler_in_match(match_id, innings, bowler):
    over_list = get_over_list_by_bowler(match_id, bowler, 20)
    count = 0
    for i in range(0,len(over_list)):
        count += get_wickets_by_bowler_in_over(match_id, innings, over_list[i])
    return count


def get_over_list_by_bowler(match_id, bowler, overlimit):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    bowler_balls = match_balls[match_balls["bowler"] == bowler]
    over_balls = bowler_balls[bowler_balls["over"] <= overlimit]
    over_balls = over_balls["over"]
    over_list = over_balls.unique()
    return over_list

def get_wickets_in_over(match_id, innings, over_id):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    over_balls = innings_balls[innings_balls["over"] == over_id]
    wick = over_balls[over_balls["dismissal_kind"] != "0"]
    return len(wick)

def get_wickets_by_bowler_in_over(match_id, innings, over_id):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    over_balls = innings_balls[innings_balls["over"] == over_id]
    wick = over_balls[over_balls["dismissal_kind"] != "0"]
    wick = wick[wick["dismissal_kind"] != "run out"]
    return len(wick)

def get_wickets_upto_over(match_id, innings, over_id):
    wick = 0
    for i in range(1,over_id+1):
        wick += get_wickets_in_over(match_id, innings, i)

    return wick

def get_run_rate_upto_over(match_id, innings, over_id):
    runs = 0
    for i in range(1,over_id+1):
        runs += get_runs_in_over(match_id, innings, i)

    return float(runs)/over_id

def get_ball_count_in_innings(match_id,innings):
    ball = 0
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    no_wides = innings_balls[innings_balls["wide_runs"] == 0]
    no_noball = no_wides[no_wides["noball_runs"] == 0]
    return len(no_noball)

def get_run_count_in_innings(match_id, innings):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    runs = innings_balls["total_runs"]
    return runs.sum()

#
#
# Match helper functions functions
def get_bowler_list(match_id, innings):
    bowlerlist = []
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    tot_bolwers = innings_balls["bowler"]
    bowlerlist = tot_bolwers.unique()
    return bowlerlist

def get_innings_of_bowler(match_id, bowler):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    bowler_balls = match_balls[match_balls["bowler"] == bowler]
    ing = bowler_balls["inning"].unique()
    if len(ing) > 0 :
        return ing[0]
    else:
        return 0

#
#
# prior statiistics functions given a bowler
def get_previous_match_list(bowler,matchlimit):
    bowler_balls = deliveries[deliveries["bowler"] == bowler]
    matchlimit = bowler_balls[bowler_balls["match_id"] < matchlimit]
    matchlist = matchlimit["match_id"]
    return matchlist.unique()
    
def get_previous_runs_by_bowler(bowler, match_id):
    match_list = get_previous_match_list(bowler, match_id)
    ing = 0
    run = 0
    for i in range(0,len(match_list)):
        ing = get_innings_of_bowler(match_list[i], bowler)
        run += get_runs_by_bowler_in_match(match_list[i], ing, bowler)
    return run

def get_previous_wickets_by_bowler(bowler, match_id):
    match_list = get_previous_match_list(bowler, match_id)
    ing = 0
    wick = 0
    for i in range(0,len(match_list)):
        ing = get_innings_of_bowler(match_list[i], bowler)
        wick += get_wickets_by_bowler_in_match(match_id, ing, bowler)
    return wick

def get_previous_over_by_bowler(bowler,match_id):
    match_list = get_previous_match_list(bowler, match_id)
    over_count = 0
    for i in match_list:
        over_count += len(get_over_list_by_bowler(i, bowler, 20))
    return over_count

def get_previous_strike_rate_by_bowler(bowler, match_id):
    wick = get_previous_wickets_by_bowler(bowler, match_id)
    balls = get_previous_over_by_bowler(bowler, match_id)*6
    if(wick == 0):
        return get_prior_strike_rate(match_id)

    return float(balls)/wick

def get_previous_economy_by_bowler(bowler, match_id):
    run = get_previous_runs_by_bowler(bowler, match_id)
    over = get_previous_over_by_bowler(bowler, match_id)
    ec  = 0
    if(over == 0):
        ec = get_prior_economy(match_id)
    else:
        ec = float(run)/over
    return ec
#
#
#prior statistics for all bowlers
def get_prior_economy(match_id):
    if(match_id == 1):
        return 7.986

    actual_balls = deliveries[deliveries["match_id"] < match_id]
    not_wide = actual_balls[actual_balls["wide_runs"] == 0]
    valid_deliveries = not_wide[not_wide["noball_runs"] == 0]
    ball_count = len(valid_deliveries)
    over_count = float(ball_count) / 6

    run_column = actual_balls["total_runs"]
    run_count = run_column.sum()
    economy = float(run_count)/over_count
    return economy

def get_prior_strike_rate(match_id):
    if(match_id == 1):
        return 19.6
    
    actual_balls = deliveries[deliveries["match_id"] < match_id]

    not_wide = actual_balls[actual_balls["wide_runs"] == 0]
    valid_deliveries = not_wide[not_wide["noball_runs"] == 0]
    ball_count = len(valid_deliveries)

    wick = actual_balls[actual_balls["dismissal_kind"] != "0"]
    wick_count = len(wick)
    strike_rate = float(ball_count)/wick_count
    return strike_rate

def get_prior_bowling_average(match_id):
    if(match_id == 1):
        return 27.53

    actual_balls = deliveries[deliveries["match_id"] < match_id]
    valid_runs = actual_balls["batsman_runs"]
    run_count = valid_runs.sum()

    wick = actual_balls[actual_balls["dismissal_kind"] != "0"]
    valid_wick = wick[wick["dismissal_kind"] != "run out"]
    wick_count = len(valid_wick)
    average = float(run_count)/wick_count
    return average
#
#
# seasonal statistics
def get_seasonal_match_list(bowler, matchlimit):
    season = get_season(matchlimit)
    min_limit = get_min_limit_of_season(season)

    season_balls = deliveries[deliveries["match_id"] >= min_limit]
    past_balls = season_balls[season_balls["match_id"] < matchlimit]
    bowler_balls = past_balls[past_balls["bowler"] == bowler]
    matchlist = bowler_balls["match_id"]
    return matchlist.unique()

def get_season(match_id):
    if(match_id < 59):
        return 1
    if(match_id < 116) & (match_id > 58):
        return 2
    if(match_id < 176) & (match_id > 116):
        return 3
    if(match_id < 250) & (match_id > 176):
        return 4
    if(match_id < 323) & (match_id > 250):
        return 5
    if(match_id < 399) & (match_id > 323):
        return 6
    if(match_id < 459) & (match_id > 399):
        return 7
    if(match_id < 517) & (match_id > 459):
        return 8
    else:
        return 9

def get_min_limit_of_season(season):
    if(season == 1):
        return 1
    if(season == 2):
        return 59
    if(season == 3):
        return 116
    if(season == 4):
        return 176
    if(season == 5):
        return 250
    if(season == 6):
        return 323
    if(season == 7):
        return 399
    if(season == 8):
        return 459
    if(season == 9):
        return 517


def get_seasonal_economy(bowler, match_id):
    matchlist = get_seasonal_match_list(bowler, match_id)
    if len(matchlist) == 0:
        return get_previous_economy_by_bowler(bowler, match_id)
    run = 0
    over = 0
    for et in matchlist:
        ing = get_innings_of_bowler(et,bowler)
        run += get_runs_by_bowler_in_match(et, ing, bowler)
        over += len(get_over_list_by_bowler(et, bowler, 20))

    ec = float(run)/over
    return ec

def get_seasonal_wicket_count(bowler, match_id):
    matchlist = get_seasonal_match_list(bowler, match_id)
    if len(matchlist) == 0:
        return 0

    wick = 0
    for et in matchlist:
        ing = get_innings_of_bowler(et, bowler)
        wick += get_wickets_by_bowler_in_match(et, ing, bowler) 
    return wick

def get_seasonal_strike_rate(bowler, match_id):
    matchlist = get_seasonal_match_list(bowler, match_id)
    if len(matchlist) == 0:
        return get_previous_strike_rate_by_bowler(bowler, match_id)
    ball = 0.0
    wick = get_seasonal_wicket_count(bowler, match_id)
    for et in matchlist:
        ing = get_innings_of_bowler(et, bowler)
        ball += len(get_over_list_by_bowler(et, bowler, 20))*6

    if wick == 0:
        return get_previous_strike_rate_by_bowler(bowler, match_id)

    sr = float(ball)/ wick
    return sr





def get_seasonal_weighted_economy(match_id, innings):
    bowler_list = get_top_bowlers(match_id,innings)
    ec = 0.0
    if(len(bowler_list) > 5):
        bowler_list = bowler_list[:5]
    for et in bowler_list:
        ec += get_seasonal_economy(et, match_id)
    
    ec = float(ec)/len(bowler_list)
    return ec

def get_seasonal_weighted_strike_rate(match_id, innings):
    bowler_list = get_bowler_list(match_id,innings)
    sr = 0.0
    if(len(bowler_list) > 5):
        bowler_list = bowler_list[:5]
    for et in bowler_list:
        sr += get_seasonal_strike_rate(et, match_id)
    
    sr = float(sr)/len(bowler_list)
    return sr
    
    

#
#
#Get weighted economy of the bowlers
def get_weighted_economy_of_bowlers(match_id,innings):
    bowler_list = get_bowler_list(match_id,innings)
    ec = 0.0
    for et in bowler_list:
        pre_match_list = get_previous_match_list(et, match_id)
        if(len(pre_match_list) == 0):
            ec += get_prior_economy(match_id)
        else:
            runs = get_previous_runs_by_bowler(et,match_id)
            overs = get_previous_over_by_bowler(et, match_id)
            ec += float(runs)/overs
    ec = ec/len(bowler_list)
    return ec

def get_weighted_strike_rate_of_bowlers(match_id, innings):
    bowler_list = get_bowler_list(match_id,innings)
    sr = 0.0
    for et in bowler_list:
        sr += get_previous_strike_rate_by_bowler(et, match_id)
    sr = sr/len(bowler_list)
    return sr


def get_toss_value(match_id, innings):
    t = 0
    match_record = matches[matches["id"] == match_id ]
    dec =  match_record.get_value(match_record.index[0], 'toss_decision')
    if(dec == "bat"):
        t = 1
    else:
        t = 2
    if(t ==  innings):
        return 1
    else:
        return 0


def validate_pair(teamName, location):
    t = 0
    if(teamName == 'Kolkata Knight Riders') & (location == 'Kolkata'):
        t = 1
    if(teamName == 'Chennai Super Kings') & (location == 'Chennai'):
        t = 1
    if(teamName == 'Rajasthan Royals') & (location == 'Jaipur'):
        t = 1
    if(teamName == 'Mumbai Indians') & (location == 'Mumbai'):
        t = 1
    if(teamName == 'Deccan Chargers') & (location == 'Hyderabad'):
        t = 1
    if(teamName == 'Kings XI Punjab') & (location == 'Chandigarh'):
        t = 1
    if(teamName == 'Royal Challengers Bangalore') & (location == 'Bangalore'):
        t = 1
    if(teamName == 'Delhi Daredevils') & (location == 'Delhi'):
        t = 1
    if(teamName == 'Kochi Tuskers Kerala') & (location == 'Kochi'):
        t = 1
    if(teamName == 'Pune Warriors') & (location == 'Pune'):
        t = 1
    if(teamName == 'Sunrisers Hyderabad') & (location == 'Hyderabad'):
        t = 1
    if(teamName == 'Rising Pune Supergiants') & (location == 'Pune'):
        t = 1
    if(teamName == 'Gujarat Lions') & (location == 'Rajkot'):
        t = 1
    
    return t

    

def check_homeground(match_id,innings):
    teamName = ""
    match_record = matches[matches["id"] == match_id ]
    if(innings == 1):
        teamName = match_record.get_value(match_record.index[0], 'team1')
    else:
        teamName = match_record.get_value(match_record.index[0], 'team2')
    
    location = match_record.get_value(match_record.index[0], 'city')
    return validate_pair(teamName, location)




def get_recent_runrate(match_id, innings, over_id):
    if(over_id <=3):
        return get_run_rate_upto_over(match_id, innings, over_id)
    else:
        runs = 0
        runs += get_runs_in_over(match_id, innings, over_id-1)
        runs += get_runs_in_over(match_id, innings, over_id -2)
        runs += get_runs_in_over(match_id, innings, over_id-3)
        return float(runs)/3   

def compare(bowler1,bowler2, match_id, innings):
    wick1 = get_seasonal_wicket_count(bowler1, match_id)
    wick2 = get_seasonal_wicket_count(bowler2, match_id)

    if(wick1 > wick2):
        return 1
    if (wick2 > wick1):
        return -1
    else:
        ec1 = get_seasonal_economy(bowler1, match_id)
        ec2 = get_seasonal_economy(bowler2, match_id)
        if(int(ec1*100) < int(ec2*100)):
            return 1
        if(int(ec1*100) > int(ec2*100)):
            return -1
        else:
            pr1 = get_previous_economy_by_bowler(bowler1, match_id)
            pr2 = get_previous_economy_by_bowler(bowler2, match_id)
            if(pr1 < pr2):
                return 1
            else:
                return -1 



def get_current_match_stats_upto_over(bowler, match_id, innings, over_id):
    over_list = get_over_list_by_bowler(match_id, bowler, over_id)
    wick = 0
    runs = 0
    over_count = 0

    for i in over_list:
        if i <= over_id:
            over_count += 1
            runs += get_runs_in_over(match_id, innings, i)
            wick += get_wickets_by_bowler_in_over(match_id, innings, i)
    
    if(over_count == 0):
        return get_seasonal_economy(bowler, match_id), wick
    else:
        return float(runs)/over_count, wick


def get_top_bowlers(match_id, innings):
    bowler_list = get_bowler_list(match_id, innings)
    for i in range(len(bowler_list)):
        for j in range(len(bowler_list)-1, i, -1):
            if( compare(bowler_list[j], bowler_list[j-1] , match_id, innings) > 0 ):
                temp = bowler_list[j]
                bowler_list[j] = bowler_list[j-1]
                bowler_list[j-1] = temp
    return bowler_list

def get_top_bowlers_stats(match_id, innings, over_id):
    bowler_list = get_top_bowlers(match_id, innings)

    default_ec = get_prior_economy(match_id)
    curr_stats = []
    prev_stats = []
    wick_stats = []

    for i in range(3):
        curr_stats.append(default_ec)
        prev_stats.append(default_ec)
        wick_stats.append(0)

    for i in range(3):
        if(i < len(bowler_list)):
            temp = get_current_match_stats_upto_over(bowler_list[i], match_id, innings,over_id)
            curr_stats[i] = temp[0]
            wick_stats[i] = temp[1]
            prev_stats[i] = get_seasonal_economy(bowler_list[i], match_id)
    
    return curr_stats, prev_stats, wick_stats


def get_last_over(match_id, innings):
    match_balls = deliveries[deliveries["match_id"] == match_id]
    innings_balls = match_balls[match_balls["inning"] == innings]
    overs = innings_balls["over"]
    return overs.max()

def get_last_ten_over_economy(match_id, innings):
    max1 = get_last_over(match_id, innings)
    min = 0
    if(max1 > 10):
        min = max1 - 10
        ub = get_run_rate_upto_over(match_id,innings, max1)
        eb = get_run_rate_upto_over(match_id,innings,min)
        return float(ub*20 - eb*10)/10

    else:
        return get_run_rate_upto_over(match_id, innings, max1)

'''
def get_match_list_for_location(match_id):
    match_record = matches[matches["id"] == match_id ]
    location =  match_record.get_value(match_record.index[0], 'city')

    match_records = matches[matches["city"] == location]
    valid_records = match_records[match_records["id"] < match_id]

    if(len(valid_records) > 0 ):
        count = 0.0
        sum = 0.0
        for i in range(0,len(valid_records)):
            sum += get_last_ten_over_economy(valid_records.iloc[i]['id'], 1)
            count += 1

'''

def get_feature_vec(match_id, innings, over_id):
    curr_ec, prev_ec, wick = get_top_bowlers_stats(match_id, innings, over_id)
    return get_recent_runrate(match_id, innings, over_id), curr_ec[0], prev_ec[0], wick[0], curr_ec[1], prev_ec[1], wick[1], curr_ec[2], prev_ec[2], wick[2], get_wickets_upto_over(match_id, innings ,over_id), get_run_rate_upto_over(match_id, innings, over_id) ,get_toss_value(match_id, innings), check_homeground(match_id, innings), get_seasonal_weighted_economy(match_id,innings), get_seasonal_weighted_strike_rate(match_id, innings)

print get_seasonal_weighted_economy(577, 1) , get_seasonal_weighted_strike_rate(577, 1)
print get_seasonal_weighted_economy(577, 2) , get_seasonal_weighted_strike_rate(577, 2)

#with open('file_to_write.csv', 'w') as f:
#    for i in range(1,577):
#        if(i == 242) | (i == 487):
#            continue
        
#        w1 = get_seasonal_weighted_economy(i, 1)
#        w2 = get_seasonal_weighted_economy(i, 2)
#        w3 = get_seasonal_weighted_strike_rate(i,1)
#        w4 = get_seasonal_weighted_strike_rate(i,2)
    
#        print i, 1, w1, w3
#        print i, 2, w2, w4

#       f.write(str(i) + ", " + "1" + ", " + str(w1) + ", " + str(w3)+ "\n")
#        f.write(str(i) + ", " + "2" + ", " + str(w2) + ", " + str(w4)+ "\n")

    


