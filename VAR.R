install.packages("vars")
library(vars)
install.packages("mFilter")
library(mFilter)
library(tseries)
install.packages("TSstudio")
library(TSstudio)
library(forecast)
library(tidyverse)

data <- read_csv('DummyData.csv') 
head(data)
str(data)


GDP <- ts(data$`GDP Growth`, start = c(2000,1,1), frequency = 12)
AvSP <- ts(data$`Monthly Average Stock Price`, start = c(2000,1,1), frequency = 12)
infrat <- ts(data$`Monthly Inflation Rate`, start = c(2000,1,1), frequency = 12)
exports <- ts(data$`Monthly Exports`, start = c(2000,1,1), frequency = 12)
imports <- ts(data$`Monthly Imports`, start = c(2000,1,1), frequency = 12)
M1 <- ts(data$`Money Supply M1`, start = c(2000,1,1), frequency = 12)
oilprod <- ts(data$`Crude Oil Production`, start = c(2000,1,1), frequency = 12)
unempdata <- ts(data$`Unemployment Rate`, start = c(2000,1,1), frequency = 12)
emprat <- ts(data$`Employment-to-Population Ratio`, start = c(2000,1,1), frequency = 12)


ts_plot(GDP)
ts_plot(AvSP)
ts_plot(infrat)
ts_plot(exports)
ts_plot(imports)
ts_plot(M1)
ts_plot(oilprod)
ts_plot(unempdata)
ts_plot(emprat)

#Use Phillips-Perron Test to determine whether to reject the null hypothesis of unit root
#Stationary if reject null hypothesis
pp.test(GDP)
pp.test(AvSP) #not stationary
pp.test(infrat)
pp.test(exports)
pp.test(imports)
pp.test(M1) #not stationary
pp.test(oilprod) 
pp.test(unempdata)
pp.test(emprat)
pp.test(df_AvSP)
pp.test(df_oilprod)



v1 <- cbind(GDP,  infrat, exports,imports,   oilprod, unempdata, emprat)
colnames(v1) <- cbind("GDP",  'infrat', 'exports','imports',   'oilprod', 'unempdata', 'emprat')

lagselect <- VARselect(v1, lag.max = 24, type = "const")
lagselect$selection

Model1 <- VAR(v1, p = 2, type = "const", season = NULL, exog = NULL) 
#summary(Model1)

for (i in 1:60)
{
  Serial1 <- serial.test(Model1, lags.pt = i, type = "PT.asymptotic")
  print(Serial1)  
}

for (i in 1:10)
{
  Arch1 <- arch.test(Model1, lags.multi = i, multivariate.only = TRUE)
  print(Arch1)  
}


par("mar")
par(mar=c(1,1,1,1))

Stability1 <- stability(Model1, type = "OLS-CUSUM")
plot(Stability1)

trainingdata <- window(v1, c(2000,1), c(2017,12))
testingdata <- window(v1, c(2018,1), c(2020,12))
v <- VAR(trainingdata, p=2)
p <- predict(v, n.ahead=36)
res <- residuals(v)
fits <- fitted(v)
for(i in 1:7)
{
  fc <- structure(list(mean=p$fcst[[i]][,"fcst"], x=trainingdata[,i],
                       fitted=c(NA,NA,fits[,i])),class="forecast")
  print(accuracy(fc,testingdata[,i]))
}

fc <- structure(list(mean=p$fcst[[1]][,"fcst"], x=trainingdata[,1],
                     fitted=c(NA,NA,fits[,1])),class="forecast")
fc
print(accuracy(fc,testingdata[,1]))

forecast <- predict(Model1, c(2000,1), c(2020,12), n.ahead = 12)
forecast

