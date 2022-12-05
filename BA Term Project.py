# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import copy
import numpy as np

flights = pd.read_csv("/Users/user/Downloads/flights.csv")
weather = pd.read_csv("/Users/user/Downloads/weather.csv")

# # Preprocessing
#

# ## Progress 1

flights.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['DISTANCE'].mean() #거리 평균 확인하다가 숫자로 이루어진 공항 발견

flights['ORIGIN_AIRPORT'] = flights['ORIGIN_AIRPORT'].astype(str)

flights['DESTINATION_AIRPORT'] = flights['DESTINATION_AIRPORT'].astype(str)

# +
#flights.ORIGIN_AIRPORT.value_counts().keys().tolist()
# -

np.unique(flights['ORIGIN_AIRPORT'])

np.unique(flights['DESTINATION_AIRPORT'])

# +
index = []
check = list(flights['ORIGIN_AIRPORT'])
for i in range(len(flights)):
    if len(check[i]) != 3:
        index.append(i)
        
# 출발 공항 list를 check에 넣은 후 flights의 행 개수 만큼 for문을 반복해서 
#출발 공항의 이름이 3글자가 아니면 해당 인덱스를 index 배열에 추가
# -

index 
# 출발 공항의 이름이 3글자가 아닌 데이터의 index

len(flights) # flights 행 개수

len(index) # 출발 공항과 도착 공항이 숫자로 이루어진 데이터 행 개수

flights.iloc[4385712] #index 배열 속에 들어있는 index로 flights 행 접근 시 ORIGIN_AIRPORT와 DESTINATION_AIRPORT가 5자리 숫자임을 확인

flights = flights.drop(index) # 출발 공항과 도착 공항이 숫자로 이루어진 데이터 행 삭제

len(flights) # 삭제 후 원본 데이터 개수

np.unique(flights['ORIGIN_AIRPORT']) # 삭제 후 출발 공항이 숫자로 이루어진 데이터 행이 삭제됨

np.unique(flights['DESTINATION_AIRPORT']) # 삭제 후 도착 공항이 숫자로 이루어진 데이터 행이 삭제됨

numvalue = flights.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['ORIGIN_AIRPORT'].value_counts()

numvalue # 출발 공항에서 도착 공항으로 가는 데이터 셋이 얼마나 많은지 확인하기 위해 groupby 사용

key = flights.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['DISTANCE'].mean().keys()

index = []
for i in range(len(key)):
    if (numvalue[i] > 9000):
        index.append(i)
# 데이터 행 개수가 9000개 이상인 경로 인덱스 추출해서 index 배열에 넣기

index # 데이터 행이 9000개 이상인 경로의 인덱스

key[index[1]]

# +
# bidirectional flight path
path1 = ['JFK','LAX'] #NY to LA , LA to NY 
path2 = ['FLL', 'ATL'] #Miami to Atlanta, Atlanta to Miami 
path3 = ['ORD','DFW'] #Chicago to Dallas, Dallas to Chicago 
# unidirectional flight path
path4 = ['BOS','DCA'] #Boston to Washington 
path5 = ['DEN', 'LAX'] #Denver to LA
path6 = ['SEA','LAX'] #Seattle to LA 
path7 = ['SFO','LAX'] #San Francisco to LA
path8 = ['MSP','ORD'] #Minneapolis to Chicago

#적절한 개수와 위치 고려한 경로 선정
#ppt에 각 공항의 방위 넣어주기 
#path1,2,3은 왕복 비행 경로 path4,5,6,7,8은 단일 비행 경로

# +
b_path = path1 + path2 + path3
u_path = path4 + path5 + path6 + path7 + path8

#단일 운행 경로와 왕복 운행 경로를 끼리끼리 묶어서 저장
# -

len(b_path)

len(u_path)

flights_v2 = pd.DataFrame([])
for i in range(0,10,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == u_path[i]) & (flights['DESTINATION_AIRPORT'] == u_path[i+1])]
    flights_v2 = pd.concat([flights_v2, temp_df1])
# 단일 운행 경로 전처리 1-2, 3-4, ..인덱스로 접근해서 빈 dataframe에 concat으로 하나씩 추가하기

np.unique(flights_v2['ORIGIN_AIRPORT'])

np.unique(flights_v2['DESTINATION_AIRPORT'])

flights_v3 = copy.deepcopy(flights_v2)
# 깊은 복사

for i in range(0,6,2):
    temp_df1 = flights.loc[(flights['ORIGIN_AIRPORT'] == b_path[i]) & (flights['DESTINATION_AIRPORT'] == b_path[i+1])]
    temp_df2 = flights.loc[(flights['ORIGIN_AIRPORT'] == b_path[i+1]) & (flights['DESTINATION_AIRPORT'] == b_path[i])]
    temp_df = pd.concat([temp_df1, temp_df2])
    flights_v3 = pd.concat([flights_v3,temp_df])
# 왕복 운행 경로 전처리 (1-2, 2-1), (3-4, 4-3), (5-6, 6-5)

np.unique(flights_v3['ORIGIN_AIRPORT'])

np.unique(flights_v3['DESTINATION_AIRPORT'])

flights_v3

date = list(weather['datetime']);date # flights 데이터와 일치하는 2015년 기상 데이터만 남길 것임

# +
date_v1 = []
for i in range(len(date)):
    year = int(date[i].split(" ")[0].split("-")[0])
    if year == 2015:
        date_v1.append(i)

# date의 year를 추출해서 2015면 date_v1 배열에 추가
# -

weather = weather.iloc[date_v1]# weather의 date 2015에 맞추기

weather = weather.reset_index(drop = True) # index 초기화

weather.columns # 미국의 여러 지역 중 출발 공항 지역과 관련된 지역만 남기는 전처리 시작

np.unique(flights_v3['ORIGIN_AIRPORT'])

weather_v2 = weather[['datetime','Type','Atlanta','Boston','Denver','Dallas','Miami','New York', 'Los Angeles'
                      ,'Minneapolis','Chicago','Seattle','San Francisco']] 
# weather_v2에서 weather에서 필요한 컬럼만 남기기

weather_v2

# +
date1 = list(weather_v2['datetime'])
month = []
day = []
hour = []

for i in range(len(weather_v2)):
    month.append(int(date1[i].split(" ")[0].split("-")[1]))
    day.append(int(date1[i].split(" ")[0].split("-")[2]))
    hour.append(int(date1[i].split(" ")[1].split(":")[0]))

# weather_v2의 datetime에서 month, day, hour 추출 for flights 데이터 셋과 날짜 일치를 위해서
# -

weather_v2['MONTH'] = month
weather_v2['DAY'] = day
weather_v2['HOUR'] = hour

weather_v2 = weather_v2.drop(columns = ['datetime'])

weather_v2

flights_v3.columns

flights_v3 = flights_v3.drop(columns = ['DAY_OF_WEEK','YEAR','AIR_SYSTEM_DELAY','SECURITY_DELAY',
                                        'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY','WEATHER_DELAY','TAIL_NUMBER','FLIGHT_NUMBER',
                                        'TAXI_OUT', 'WHEELS_OFF','WHEELS_ON','TAXI_IN','ELAPSED_TIME','AIR_TIME',
                                       'DEPARTURE_TIME','ARRIVAL_TIME','DISTANCE','SCHEDULED_TIME','SCHEDULED_ARRIVAL','ARRIVAL_DELAY'])
#flights 데이터셋에서 분석에 사용될 지연과 관련되지 않은 데이터를 지우기

flights_v3 # 지운 후 데이터셋 출력한 모습 이 때 출발 시간이 24시간 기준으로 되어 있고 6:00의 경우 600으로 표시되어 있음

(((645%100)/60) * 100) + (645//100) * 100 # 시간 단위를 맞추고 시간만 추출하기 위한 전처리 시작

round((630%100) / 60 * 100 + (630//100) *100, -2) # ~30분까지는 -1시간 정각으로 31분부터 59분까지는 +1시간으로 만드는 수식

# +
templist = list(flights_v3['SCHEDULED_DEPARTURE'])
time = []

for i in range(len(flights_v3)):
    value = templist[i]
    time.append((round((value%100) / 60 * 100 + (value//100) *100, -2))/ 100)
    
# 에정 출발 시간을 리스트로 만든 후 수식을 적용해서 빈 time 배열에 넣기
# -

flights_v3['SCHEDULED_DEPARTURE'] = time #time 배열의 값들로 예정 출발 시간 컬럼 대체

flights_v3 = flights_v3.rename(columns = {'SCHEDULED_DEPARTURE':'HOUR'}) # 머지할 때 사용하기 위해 컬럼이름을 HOUR로 대체

# +
# flights_v3 = flights_v3.drop(columns = []) #출발 지연과 관련되지 않은 컬럼 제거
# -

weather_v2.iloc[6]

flights_v3 = flights_v3.reset_index(drop = True) # 인덱스 초기화

flights_v3

weather_v2[(weather_v2['MONTH'] == 1) & (weather_v2['DAY'] == 1) & (weather_v2['HOUR'] == 6)] 
# merge 전 weather 데이터 셋에서 특정 month, day. hour일 때의 데이터 뽑아보기
# 총 5개의 행이 출력되고 각각은 해당 month, day, hour일 때의 습도, 기압, 기온, 풍향, 풍속을 나타낸다.

weather_v2

newdf = flights_v3.merge(weather_v2, on = ['MONTH','DAY','HOUR']); newdf
# weather dataset과 flights dataset을 MONTH, DAY, HOUR 컬럼을 기준으로 merge

newdf # merge이후 뽑아본 newdf dataset

newdf.columns # 해당 출발 공항에 대한 날씨만 남기고 나머지 상관없는 지역에 대한 컬럼을 지우는 전처리 시작

check = {'ATL':'Atlanta', 'BOS':'Boston','DEN':'Denver','DFW':'Dallas',
       'FLL':'Miami', 'JFK':'New York','LAX':'Los Angeles','MSP':'Minneapolis', 'ORD':'Chicago','SEA':'Seattle',
       'SFO':'San Francisco'}

departure = np.unique(newdf['ORIGIN_AIRPORT']); departure

check[departure[0]]

# +
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
    
# for문을 내에서 temporary dataframe에 departure에 있는 항목과 출발 공항지역이 일치하는 행들을 저장
# 이후 col 변수에 각 출발 지역 날씨를 나타내는 컬럼들의 이름을 저장
# col 변수에서 현재 출발지역과 일치하는 공항만을 제거한 후 (출발지와 상관없는 지역의 날씨 컬럼이름만 남음)
# temporary dataframe에서 col 항목과 일치하는 컬럼들을 제거
# 남아 있는 현재 출발 지역의 날씨 컬럼 이름을 Weather로 변경
# for문이 돌 때마다 각 출발지역의 날씨 전처리가 완료된 행들이 concat 됨
# -

comdf = comdf.reset_index(drop = True) #인덱스 초기화

comdf

comdf.to_csv("/Users/user/Downloads/commitdf.csv", index = False)

pd.read_csv("/Users/user/Downloads/commitdf.csv") #최종 데이터 불러온 모습 약 48만개의 행과 12개의 열로 이루어져 있음

# ## Progress 2

import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import copy
import numpy as np

comdf = pd.read_csv("/Users/user/Downloads/commitdf.csv")

comdf

#Check rows that Weather is null
a = comdf[(comdf['Weather'].isnull())]; a.index

array = []
for i in range(5):
    array.append(a.index+i)

array

#Drop all weather data which has one weather type's weather value is null
#날씨 데이터가 null인 값을 가지고 있는 비행기록 5개의 열을 모두 삭제
for i in range(len(array)):
    for j in range(len(array[i])):
        comdf = comdf.drop(array[i][j])

comdf

# Humidity Boxplot
humidity = comdf[comdf['Type']=='Humidity']
plt.figure(figsize=(10,8))
boxplot = humidity.boxplot(column=['Weather'])
plt.title('Humidity')
plt.yticks(np.arange(0,101,step=5))
plt.show()


#Pressure boxplot
pressure = comdf[comdf['Type']=='Pressure']
plt.figure(figsize=(10,8))
boxplot = pressure.boxplot(column=['Weather'])
plt.title('Pressure')
plt.yticks(np.arange(800,1100, step=20))
plt.show()

#Temperature boxplot
temperature = comdf[comdf['Type']=='Temperature']
plt.figure(figsize=(10,8))
boxplot = temperature.boxplot(column=['Weather'])
plt.title('Temperature')
plt.yticks(np.arange(240,320, step=10))
plt.show()

# Wind_direction
wind_direction = comdf[comdf['Type']=='Wind_direction']
plt.figure(figsize=(15,15))
boxplot = wind_direction.boxplot(column=['Weather'])
plt.title('Wind_direction')
plt.yticks(np.arange(0,360,step=10))
plt.show()

# Wind_speed
wind_speed = comdf[comdf['Type']=='Wind_speed']
plt.figure(figsize=(15,15))
boxplot = wind_speed.boxplot(column=['Weather'])
plt.title('Wind_speed')
plt.yticks(np.arange(0,50,step=5))
plt.show()

comdf = comdf.reset_index(drop = True)

#Check null value for each column 
a = ['MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',       
     'HOUR', 'DIVERTED', 'CANCELLED', 'Type', 'Weather','CANCELLATION_REASON', 'DEPARTURE_DELAY']
for i in range(len(a)):
    b = comdf[comdf[a[i]].isnull()]
    print(a[i])
    print(b)

#Check instance that DEPARTURE_DELAY is null and CANCELLED is 1
comdf[(comdf['DEPARTURE_DELAY'].isnull())&(comdf['CANCELLED'])]

#Check instance that DEPARTURE_DELAY is null and CANCELLED is 0
# no data --> When the flight is cancelled, there is no departure delay
comdf[(comdf['DEPARTURE_DELAY'].isnull())&(comdf['CANCELLED']==0)]


comdf

#Check the index that DEPARTURE_DELAY is null
nul_index = comdf[comdf['DEPARTURE_DELAY'].isnull()].index; nul_index

# +
# nul_df = pd.DataFrame()
# for i in range(len(nul_index)):
#     temp_df = comdf.iloc[nul_index[i]]
#     nul_df = nul_df.append(temp_df)
# nul_df
# -

#Drop the index
comdf = comdf.drop(nul_index); comdf

comdf = comdf.reset_index(drop = True)

len(comdf[comdf['Type']=='Humidity'])

len(comdf[comdf['Type']=='Pressure'])

len(comdf[comdf['Type']=='Temperature'])

len(comdf[comdf['Type']=='Wind_direction'])

len(comdf[comdf['Type']=='Wind_speed'])

comdf

hum_data = comdf[comdf['Type']=='Humidity']
hum_data = hum_data.rename(columns={'Weather':'Humidity'})
hum_weather = hum_data['Humidity']
hum_weather = hum_weather.reset_index(drop=True); hum_weather

# +
#Code for changing the table structure
#Each weather type becomes a column and the value of each weather becomes an instance
weather_name = ['Humidity','Pressure','Temperature','Wind_direction', 'Wind_speed']
weather_df = pd.DataFrame()
for i in range(5):
    each_weather_data = comdf[comdf['Type']==weather_name[i]]
    each_weather_data = each_weather_data.rename(columns={'Weather':weather_name[i]})
    each_weather = each_weather_data[weather_name[i]]
    each_weather = each_weather.reset_index(drop=True)
    weather_df = pd.concat([weather_df,each_weather],axis = 1)
weather_df
    
    
# -

check_df = pd.DataFrame()
for i in range(5):
    check_df = check_df.append(comdf.iloc[i])
check_df

# 중복 데이터 삭제 코드
# 중복 데이터 삭제 이후 위의 날씨 데이터와 concat -> regression에 사용하기 위해
# 5의 배수 열만 추출
from tqdm import trange, notebook
empty_df = pd.DataFrame()
for i in notebook.tqdm(range(0, len(comdf), 5)):
    empty_df = empty_df.append(comdf.iloc[i])
empty_df

empty_df = empty_df.reset_index(drop = True) # index 초기화

# weather_df = pd.concat([weather_df,each_weather],axis = 1)
reg_df = pd.concat([empty_df, weather_df],axis=1)
reg_df

reg_df = reg_df.drop(['Type','Weather'],axis=1)

reg_df

reg_df.to_csv("/Users/user/Downloads/reg.csv", index = False)

# # Regression

# ### Weather Selection - PairPlot & Correlation Matrix 

import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import copy
import numpy as np

reg_df = pd.read_csv("reg.csv")

X = reg_df[['Humidity', 'Pressure', 'Temperature', 'Wind_direction', 'Wind_speed']]
y = reg_df['DEPARTURE_DELAY']

X

y

df = pd.concat([X,y],axis = 1); df

df['dd'] = df['DEPARTURE_DELAY'].apply(lambda x : 0 if x < 0 else x)

df['dd'].describe()

import seaborn as sns
sns.pairplot(X)
plt.show()

X.corr() #temperature와 pressure가 상관관계

# +
fig, ax = plt.subplots(figsize=(11, 11))
X_corr = X.corr()

# mask
mask = np.triu(np.ones_like(X_corr, dtype=np.bool))

# adjust mask and df
mask = mask[:, :]
corr = X_corr.iloc[:,:].copy()

# color map
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
            linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
            cbar_kws={"shrink": .8}, square=True)

# ticks
yticks = [i.upper() for i in corr.index]
xticks = [i.upper() for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
plt.xticks(plt.xticks()[0], labels=xticks)

# title
title = 'CORRELATION MATRIX\n'
plt.title(title, loc='left', fontsize=15)
plt.show()
# -

# ### Weather Selection - VIF

# +
#Check VIF factor for 5 weather variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
# -

X_2 = X.drop(['Pressure'],axis=1)

# +
#Check VIF factor for Humidity, Temperature, Wind_direction, Wind_speed
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_2.values, i) for i in range(X_2.shape[1])]
vif["features"] = X_2.columns
vif
# -

X_3 = X.drop(['Temperature'], axis=1)

# +
#Check VIF Factor for Humidity, Pressure, Wind_direction, Wind_speed
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_3.values, i) for i in range(X_3.shape[1])]
vif["features"] = X_3.columns
vif
# -

X_4 = X_2.drop(['Temperature'], axis=1)

# +
#Check VIF Factor for Humidity, Wind_direction, Wind_speed
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_4.values, i) for i in range(X_4.shape[1])]
vif["features"] = X_4.columns
vif
# -

# ### Lidge & Lasso Regression with CV and Grid Search

X = df[['Humidity','Wind_direction','Wind_speed']]
X

y = df['dd']
y

#train/test set 나누기
from sklearn.model_selection import train_test_split
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=0)

# +
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
import tqdm


pipe = Pipeline([('scaler', None), ('poly', None), ('linear_model', None)])

param_grid = [
    {'scaler':[StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(),None],
     'poly':[PolynomialFeatures()],
     'poly__degree':np.arange(1,3),
     'linear_model':[Ridge(random_state=0)],
     'linear_model__alpha': np.linspace(2,5,10),
     'linear_model__tol': np.logspace(-5,0,10)},
    
    {'scaler':[StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(),None],
     'poly':[PolynomialFeatures()],
     'poly__degree':np.arange(1,3),
     'linear_model':[Lasso(random_state=0)],
     'linear_model__alpha': np.logspace(-5,1,10),
     'linear_model__tol': np.logspace(-5,0,10)}]

grid = GridSearchCV(pipe, param_grid, cv=5, scoring= 'neg_mean_squared_error') #-MSE score
grid.fit(X_trn, y_trn)

# -

import pandas as pd
results = pd.DataFrame(grid.cv_results_)
results

print("Best hyperparams: {}".format(grid.best_params_))
print("Best cross-validation score: {}".format(grid.best_score_))
print("Test-set score: {}".format(grid.score(X_tst, y_tst)))

# +
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
import tqdm


pipe = Pipeline([('scaler', None), ('poly', None), ('linear_model', None)])

param_grid = [
    {'scaler':[StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(),None],
     'poly':[PolynomialFeatures()],
     'poly__degree':np.arange(1,3),
     'linear_model':[Ridge(random_state=0)],
     'linear_model__alpha': np.linspace(2,5,10),
     'linear_model__tol': np.logspace(-5,0,10)},
    
    {'scaler':[StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(),None],
     'poly':[PolynomialFeatures()],
     'poly__degree':np.arange(1,3),
     'linear_model':[Lasso(random_state=0)],
     'linear_model__alpha': np.logspace(-5,1,10),
     'linear_model__tol': np.logspace(-5,0,10)}]

grid = GridSearchCV(pipe, param_grid, cv=5, scoring= 'r2') #-R2 score
grid.fit(X_trn, y_trn)
# -

import pandas as pd
results = pd.DataFrame(grid.cv_results_)
results

print("Best hyperparams: {}".format(grid.best_params_))
print("Best cross-validation score: {}".format(grid.best_score_))
print("Test-set score: {}".format(grid.score(X_tst, y_tst)))

# ### Linear Regression for estimating weight of weather

df

X = df[['Humidity','Wind_direction','Wind_speed']]

y = df['dd']

X

y

#split into train/test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=0)

# +
#Standard Scaler 적용 O
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr.fit(X_train_scaled, y_train) 

y_train_hat = lr.predict(X_train_scaled)
y_test_hat = lr.predict(X_test_scaled)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))

# -

ylist = y_train.tolist()

y_train_hat

residual = [ylist[x] - y_train_hat[x] for x in range(len(ylist))]

plt.scatter(np.arange(0,len(residual),1),residual)
plt.ylim(-500,500)
plt.axhline(y= 0, color = 'r')
plt.axhline(y= 200 ,color = 'r', linestyle = '--')
plt.axhline(y= -200, color = 'r', linestyle = '--')

scaler = StandardScaler()
scaler.fit(df[['Humidity']])
X_train_scaled_h = scaler.transform(df[['Humidity']])
dfh = pd.DataFrame(X_train_scaled_h)

dfh.plot(kind='kde')
plt.axvline(x = 0, color = 'r', linestyle = '--')
plt.xlim(-3,3)
plt.legend(['Humidity'])

scaler = StandardScaler()
scaler.fit(df[['Wind_direction']])
X_train_scaled_wd = scaler.transform(df[['Wind_direction']])
dfwd = pd.DataFrame(X_train_scaled_wd)

dfwd.plot(kind='kde')
plt.axvline(x = 0, color = 'r', linestyle = '--')
plt.xlim(-3,3)
plt.legend(['Wind Direction'])

scaler = StandardScaler()
scaler.fit(df[['Wind_speed']])
X_train_scaled_ws = scaler.transform(df[['Wind_speed']])
dfws = pd.DataFrame(X_train_scaled_ws)

dfws.plot(kind='kde')
plt.axvline(x = 0, color = 'r', linestyle = '--')
plt.xlim(-3,3)
plt.legend(['Wind Speed'])

import scipy.stats as stats

stats.probplot(residual, dist = stats.norm, plot = plt)

# # Appendix - extra analysis

# The analyses which are not contained in presentation

# +
#General linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()

lr.fit(X_train, y_train) 

y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))
# -

#observe weight
lr.coef_ 

# +
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()

lr.fit(X_train, y_train) 

y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))

# +
#Standard Scaler 적용 O
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr.fit(X_train_scaled, y_train) 

y_train_hat = lr.predict(X_train_scaled)
y_test_hat = lr.predict(X_test_scaled)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))
# -

X_train

lr.coef_ 
# 차례로 습도,압력,온도,풍향,풍량 -> 즉 출발 지연 정도는 습도가 낮고 압력이 낮고, 온도가 낮고, 
# 풍향이 작고(북->동->남, 0~181일 때 음수의 scaled된 값을 가지므로 이 때 지연 정도가 커짐), 풍량이 많을수록 커짐

#y= DEPARTUE_DELAY
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, reg_df['DEPARTURE_DELAY'] , random_state=0)

# +
#scaler 적용 X
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()

lr.fit(X_train, y_train) 

y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))

# -

lr.coef_ 

# +
#Standard Scaler 적용 O
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr.fit(X_train_scaled, y_train) 

y_train_hat = lr.predict(X_train_scaled)
y_test_hat = lr.predict(X_test_scaled)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))
# -

lr.coef_ 
# 차례로 습도,압력,온도,풍향,풍량 -> 즉 출발 지연 정도는 습도가 낮고 압력이 낮고, 온도가 낮고, 
# 풍향이 작고(북->동->남, 0~181일 때 음수의 scaled된 값을 가지므로 이 때 지연 정도가 커짐), 풍량이 많을수록 커짐

# +
#Robust Scaler 적용
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


robustScaler = RobustScaler()
robustScaler.fit(X_train)

lr = LinearRegression()
X_train_scaled = robustScaler.transform(X_train)
X_test_scaled = robustScaler.transform(X_test)

lr.fit(X_train_scaled, y_train) 

y_train_hat = lr.predict(X_train_scaled)
y_test_hat = lr.predict(X_test_scaled)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))

# -

lr.coef_ 

# ### Linear Regression with Humidity, Wind_direction, Wind_speed

X = df[['Humidity','Wind_direction', 'Wind_speed']]
X

y=reg_df['DEPARTURE_DELAY']

y1 = df['dd']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1 , random_state=0)

# +
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


robustScaler = RobustScaler()
robustScaler.fit(X_train)

lr = LinearRegression()
X_train_scaled = robustScaler.transform(X_train)
X_test_scaled = robustScaler.transform(X_test)

lr.fit(X_train, y_train) 

y_train_hat = lr.predict(X_train_scaled)
y_test_hat = lr.predict(X_test_scaled)

print('performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('performance for TRAIN--------')
print('train R2 : ', r2_score(y_train, y_train_hat))
print('performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))
print('performance for TRAIN--------')
print('test R2 : ', r2_score(y_test, y_test_hat))

# -

# # Anomaly Detection with Isolation Forest

from sklearn.ensemble import IsolationForest

reg_df

reg_df['DEPARTURE_DELAY'] = reg_df['DEPARTURE_DELAY'].apply(lambda x : 0 if x < 0 else x)
reg_df['Label'] = reg_df['DEPARTURE_DELAY'].apply(lambda x : 1 if x > 70 else 0)

reg_df

reg_df['Label'].value_counts()

4813 / 93831


def iForest(data, weathers, contamination, flag):
    df = data[['MONTH','DAY','ORIGIN_AIRPORT','DESTINATION_AIRPORT',weathers[0], weathers[1], weathers[2],'DEPARTURE_DELAY', 'CANCELLED','CANCELLATION_REASON','Label']]
    
    cdf = copy.deepcopy(df)


    ifdf = df[[weathers[0],weathers[1],weathers[2]]]

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

        ax.scatter(xi, yi, zi, label = 'Normal')
        ax.scatter(xo, yo, zo, color='red', marker='x', label = 'Abnormal')
        ax.set_xlabel(weathers[0])
        ax.set_ylabel(weathers[1])
        ax.set_zlabel(weathers[2])
        ax.view_init(10, 120)
        plt.legend()
        plt.show()
    else:
        return cdf

iForest(reg_df, ['Humidity','Wind_direction','Wind_speed'], 0.05, 'graph')

ifdf = iForest(reg_df, ['Humidity','Wind_direction','Wind_speed'], 0.05, 'df')

ifdf.loc[ifdf['CANCELLATION_REASON'] == 'B'][['CANCELLED','pred']].reset_index(drop = True)

fr = ifdf.loc[ifdf['pred'] == False]
tr = ifdf.loc[ifdf['pred'] == True]

fr['Label'].value_counts()

tr['Label'].value_counts()

fr[['Humidity','Wind_direction','Wind_speed','DEPARTURE_DELAY']].describe().loc[['mean','std','25%','50%','75%']]

tr[['Humidity','Wind_direction','Wind_speed','DEPARTURE_DELAY']].describe().loc[['mean','std','25%','50%','75%']]


