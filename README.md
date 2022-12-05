# BusinessAnalyticsProject
Business Analytics course aims to provide students with technologies, applications, practices, and skills for continuous iterative exploration and investigation of past business performance along with external data generated from diverse sources such as web and social network service (SNS) to gain insights and drive business planning.

This term project is for students in the year 3 module at the ITM Programme.

## Data
2015 Flights Delay and Cancellations (https://www.kaggle.com/datasets/usdot/flight-delays)  
Historical Hourly Weather Data 2012-2017 (https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)

## Goal
The goal of project is to predict the relationship between some weather factors and the amount time of delay in terms of the flights take-off.

## Method
Train dataset using ridge or lasso regression model to figure out the relationship explained above.  
Isolation Forest is used to find anomalies related to the extreme weather condition and We compare them to the amount of time delay of a specific flights take-off. 

## Specification
Candidates of the weather type are humidity, pressure, temperature, wind direction, and wind speed.  
3 of 5 was selected based on the VIF (Variance Inflaton Factor) and the most influential weight.  
Considering the location of airports in the US for the variety of the weather, and the number of people who depature at a specific airport, total 11 paths are selected.  

## Conclusion
We apply Ridge & Lasso regression method for selected weather variables with gridSearch, however, there is too low score of train model and test result.  
We try to find the reason why by analyzing the data distribution and residual, we conclude that weather does not affect the amount of delay time. To check verification related to the extreme weather condition again, we apply the Isolation forest method, however, extreme weather condition also does not affect the amount of delay time.  
We failed to predict the amount of delay time though, unlike our empirical thought, we figured out that the weather is not a critical factor for the aircraft take-off.

## Contributors 
Sunho Kim,  
Eunji Lee
