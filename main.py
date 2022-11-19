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

path1 = ['JFK','LAX']
path2 = ['FLL', 'ATL']
path3 = ['ORD','DFW']

path4 = ['BOS','DCA']
path5 = ['DEN', 'LAX']
path6 = ['SEA','LAX']
path7 = ['SFO','LAX']
path8 = ['MSP','ORD']

d_path = path1 + path2 + path3
o_path = path4 + path5 + path6 + path7 + path8

flights_v2 = pd.DataFrame([])
for i in range(0,10,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == o_path[i]) & (flights['DESTINATION_AIRPORT'] == o_path[i+1])]
    flights_v2 = pd.concat([flights_v2, temp_df1])
flights_v3 = copy.deepcopy(flights_v2)
for i in range(0,6,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == d_path[i]) & (flights['DESTINATION_AIRPORT'] == d_path[i+1])]
    temp_df2 = flights.loc[(flights['ORIGIN_AIRPORT'] == d_path[i+1]) & (flights['DESTINATION_AIRPORT'] == d_path[i])]
    temp_df = pd.concat([temp_df1, temp_df2])
    flights_v3 = pd.concat([flights_v3,temp_df])



date = list(weather['datetime'])
date_v1 = []
for i in range(len(date)):
    year = int(date[i].split(" ")[0].split("-")[0])
    if year == 2015:
        date_v1.append(i)

weather = weather.iloc[date_v1]
weather = weather.reset_index(drop = True)
weather_v2 = weather[['datetime','Type','Atlanta','Boston','Denver','Dallas','Miami','New York', 'Los Angeles'
,'Minneapolis','Chicago','Seattle','San Francisco']]

date1 = list(weather_v2['datetime'])
month = []
day = []
hour = []

for i in range(len(weather_v2)):
    month.append(int(date1[i].split(" ")[0].split("-")[1]))
    day.append(int(date1[i].split(" ")[0].split("-")[2]))
    hour.append(int(date1[i].split(" ")[1].split(":")[0]))

weather_v2['MONTH'] = month
weather_v2['DAY'] = day
weather_v2['HOUR'] = hour

weather_v2 = weather_v2.drop(columns = ['datetime'])

flights_v3 = flights_v3.drop(columns = ['DAY_OF_WEEK','YEAR','AIR_SYSTEM_DELAY',
       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
       'WEATHER_DELAY','TAIL_NUMBER','FLIGHT_NUMBER','TAXI_OUT', 'WHEELS_OFF','WHEELS_ON','TAXI_IN','ELAPSED_TIME','AIR_TIME',
                                       'DEPARTURE_TIME','ARRIVAL_TIME','DISTANCE'])

templist = list(flights_v3['SCHEDULED_DEPARTURE'])
time = []

for i in range(len(flights_v3)):
    value = templist[i]
    time.append((round((value%100) / 60 * 100 + (value//100)* 100, -2))/ 100)

flights_v3['SCHEDULED_DEPARTURE'] = time
flights_v3 = flights_v3.drop(columns = ['SCHEDULED_TIME','SCHEDULED_ARRIVAL','ARRIVAL_DELAY'])
flights_v3 = flights_v3.rename(columns = {'SCHEDULED_DEPARTURE':'HOUR'})
flights_v3 = flights_v3.reset_index(drop = True)

newdf = flights_v3.merge(weather_v2, on = ['MONTH','DAY','HOUR'])

check = {'ATL':'Atlanta', 'BOS':'Boston','DEN':'Denver','DFW':'Dallas',
       'FLL':'Miami', 'JFK':'New York','LAX':'Los Angeles','MSP':'Minneapolis', 'ORD':'Chicago','SEA':'Seattle',
       'SFO':'San Francisco'}
departure = np.unique(newdf['ORIGIN_AIRPORT'])
comdf = pd.DataFrame([])
for i in range(len(departure)):
    tempdf = newdf.loc[newdf['ORIGIN_AIRPORT'] == departure[i]]
    col = ['Atlanta', 'Boston', 'Denver', 'Dallas',
       'Miami', 'New York', 'Los Angeles', 'Minneapolis', 'Chicago', 'Seattle',
       'San Francisco']
    col.remove(check[departure[i]])
    tempdf = tempdf.drop(columns = col)
    tempdf = tempdf.rename(columns = {check[departure[i]]:'Weather'})
    comdf = pd.concat([comdf,tempdf])

comdf = comdf.reset_index(drop = True)
comdf.to_csv("/Users/moon/Desktop/commitdf.csv", index = False)

#

def iForest(data, weathers, contamination, flag):
    data = data.loc[(data['Type'] == weathers[0]) | (data['Type'] == weathers[1]) | (data['Type'] == weathers[2])]
    data = data.reset_index(drop=True)
    df1 = data.loc[data['Type'] == weathers[0]].reset_index(drop=True)
    df2 = data.loc[data['Type'] == weathers[1]].reset_index(drop=True)
    df3 = data.loc[data['Type'] == weathers[2]].reset_index(drop=True)

    df1 = df1.drop(columns=['Type'])
    df1 = df1.rename(columns={'Weather': weathers[0]})
    df2 = df2.drop(columns=['Type'])
    df2 = df2.rename(columns={'Weather': weathers[1]})
    df3 = df3.drop(columns=['Type'])
    df3 = df3.rename(columns={'Weather': weathers[2]})

    df1[weathers[1]] = df2[weathers[1]]
    df1[weathers[2]] = df3[weathers[2]]

    cdf = copy.deepcopy(df1)

    cr = list(df1['CANCELLATION_REASON'])
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
