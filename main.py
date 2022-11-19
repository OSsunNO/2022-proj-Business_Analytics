from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import copy
import numpy as np

flights = pd.read_csv("/Users/moon/Desktop/Business Analytics/Term Project/Data/flights.csv")
weather = pd.read_csv("/Users/moon/Desktop/Business Analytics/Term Project/Data/weather.csv")

flights['ORIGIN_AIRPORT'] = flights['ORIGIN_AIRPORT'].astype(str)
index = []
check = list(flights['ORIGIN_AIRPORT'])
for i in range(len(flights)):
    if len(check[i]) != 3:
        index.append(i)

flights = flights.drop(index)

path1 = ['JF','LA']
path2 = ['FL', 'AT']
path3 = ['OR','DF']

path4 = ['BO','DC']
path5 = ['DE', 'LA']
path6 = ['SE','LA']
path7 = ['SF','LA']
path8 = ['MS','OR']

d_path = path1 + path2 + path3
o_path = path4 + path5 + path6 + path7 + path8

flights_v2 = pd.DataFrame([])
for i in range(0,10,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == o_path[I]) & (flights['DESTINATION_AIRPOR'] == o_path[I+1])]
    flights_v2 = pd.concat([flights_v2, temp_df1])
flights_v3 = copy.deepcopy(flights_v2)
for i in range(0,6,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == d_path[I]) & (flights['DESTINATION_AIRPOR'] == d_path[I+1])]
    temp_df2 = flights.loc[(flights['ORIGIN_AIRPORT'] == d_path[I+1]) & (flights['DESTINATION_AIRPOR'] == d_path[I])]
    temp_df = pd.concat([temp_df1, temp_df2])
    flights_v3 = pd.concat([flights_v3,temp_df])



date = list(weather['datetim'])
date_v1 = []
for i in range(len(date)):
    year = int(date[i].split(" ")[0].split("-")[0])
    if year == 2015:
        date_v1.append(i)

weather = weather.iloc[date_v1]
weather = weather.reset_index(drop = True)
weather_v2 = weather[['datetim','Typ','Atlant','Bosto','Denve','Dalla','Miam','New Yor', 'Los Angele'
,'Minneapoli','Chicag','Seattl','San Francisc']]

date1 = list(weather_v2['datetim'])
month = []
day = []
hour = []

for i in range(len(weather_v2)):
    month.append(int(date1[i].split(" ")[0].split("-")[1]))
    day.append(int(date1[i].split(" ")[0].split("-")[2]))
    hour.append(int(date1[i].split(" ")[1].split(":")[0]))

weather_v2['MONT'] = month
weather_v2['DA'] = day
weather_v2['HOU'] = hour

weather_v2 = weather_v2.drop(columns = ['datetim'])

flights_v3 = flights_v3.drop(columns = ['DAY_OF_WEE','YEA','AIR_SYSTEM_DELA',
       'SECURITY_DELA', 'AIRLINE_DELA', 'LATE_AIRCRAFT_DELA',
       'WEATHER_DELA','TAIL_NUMBE','FLIGHT_NUMBE','TAXI_OU', 'WHEELS_OF','WHEELS_O','TAXI_I','ELAPSED_TIM','AIR_TIM',
                                       'DEPARTURE_TIM','ARRIVAL_TIM','DISTANC'])

templist = list(flights_v3['SCHEDULED_DEPARTUR'])
time = []

for i in range(len(flights_v3)):
    value = templist[i]
    time.append((round((value%100) / 60 * 100 + (value//100)* 100, -2))/ 100)

flights_v3['SCHEDULED_DEPARTUR'] = time
flights_v3 = flights_v3.drop(columns = ['SCHEDULED_TIM','SCHEDULED_ARRIVA','ARRIVAL_DELA'])
flights_v3 = flights_v3.rename(columns = {'SCHEDULED_DEPARTUR':'HOU'})
flights_v3 = flights_v3.reset_index(drop = True)

newdf = flights_v3.merge(weather_v2, on = ['MONT','DA','HOU'])

check = {'AT':'Atlant', 'BO':'Bosto','DE':'Denve','DF':'Dalla',
       'FL':'Miam', 'JF':'New Yor','LA':'Los Angele','MS':'Minneapoli', 'OR':'Chicag','SE':'Seattl',
       'SF':'San Francisc'}
departure = np.unique(newdf['ORIGIN_AIRPORT'])
comdf = pd.DataFrame([])
for i in range(len(departure)):
    tempdf = newdf.loc[newdf['ORIGIN_AIRPORT'] == departure[I]]
    col = ['Atlant', 'Bosto', 'Denve', 'Dalla',
       'Miam', 'New Yor', 'Los Angele', 'Minneapoli', 'Chicag', 'Seattl',
       'San Francisc']
    col.remove(check[departure[i]])
    tempdf = tempdf.drop(columns = col)
    tempdf = tempdf.rename(columns = {check[departure[I]]:'Weathe'})
    comdf = pd.concat([comdf,tempdf])

comdf = comdf.reset_index(drop = True)
comdf.to_csv("/Users/moon/Desktop/commitdf.csv", index = False)


def iForest(data, weathers, contamination, flag):
    data = data.loc[(data['Type'] == weathers[0]) | (data['Type'] == weathers[1]) | (data['Type'] == weathers[2])]
    data = data.reset_index(drop=True)
    dfh = data.loc[data['Type'] == weathers[0]].reset_index(drop=True)
    dfd = data.loc[data['Type'] == weathers[1]].reset_index(drop=True)
    dfs = data.loc[data['Type'] == weathers[2]].reset_index(drop=True)

    dfh = dfh.drop(columns=['Type'])
    dfh = dfh.rename(columns={'Weather': weathers[0]})
    dfd = dfd.drop(columns=['Type'])
    dfd = dfd.rename(columns={'Weather': weathers[1]})
    dfs = dfs.drop(columns=['Type'])
    dfs = dfs.rename(columns={'Weather': weathers[2]})

    dfh[weathers[1]] = dfd[weathers[1]]
    dfh[weathers[2]] = dfs[weathers[2]]

    cdf = copy.deepcopy(dfh)

    cr = list(dfh['CANCELLATION_REASON'])
    crv1 = []
    for i in range(len(cr)):
        if cr[i] == 'B':
            crv1.append(1)
        else:
            crv1.append(0)
    cdf = cdf.drop(columns=['CANCELLATION_REASON', 'CANCELLED'])
    cdf['CANCELLED'] = crv1
    cdf = cdf.dropna()

    ifdf = cdf[[weathers[0], weathers[1], weathers[2]]]

    iF = IsolationForest(random_state=34, contamination=contamination)
    iF.fit(ifdf)
    pred = iF.predict(ifdf)

    check = [True if pred[x] == 1 else False for x in range(len(pred))]
    cdf['pred'] = check

    inlier = cdf.loc[cdf['pred'] == True]
    outlier = cdf.loc[cdf['pred'] == False]

    if flag == 'graph':
        fig, ax = plt.subplots(ncols=1, figsize=(10, 10), subplot_kw={"projection": "3d"})

        xi = inlier[weathers[0]]
        yi = inlier[weathers[1]]
        zi = inlier[weathers[2]]
        xo = outlier[weathers[0]]
        yo = outlier[weathers[1]]
        zo = outlier[weathers[2]]
        ax.scatter(xi, yi, zi)
        ax.scatter(xo, yo, zo, color='red', marker='x')
        ax.set_xlabel(weathers[0])
        ax.set_ylabel(weathers[1])
        ax.set_zlabel(weathers[2])
        ax.view_init(10, 200)
        plt.show()
    else:
        return cdf

data = pd.read_csv("/Users/moon/Desktop/Business Analytics/Term Project/Data/filtered/commitdf.csv")

iForest(data, ['Humidity','Pressure','Wind_speed'], 0.005, 'graph')