install.packages('tempdisagg')
install.packages('tsbox')
library(tempdisagg)
library(tsbox)
getwd()
setwd('C:/Users/jaylu/Documents/R DATA/CP2')
data <- read.csv('2_unemploymentrateq.csv',header = T)
str(data)
data <- data[,-1]
data_ts <- ts(data, frequency = 4)
data_ts
mod <- td(data_ts~1, to = "monthly", conversion = "first",  method = "chow-lin-maxlog")
unemplpoymentrate <- predict(mod)
write.csv(unemplpoymentrate, 'unemploymentm.csv')


data2 <- read.csv('7_employedpersons.csv')
str(data2)
data2 <- data2[,-1]
data_ts2 <- ts(data2, frequency = 1)
data_ts2
mod2 <- td(data_ts2~1, to = "monthly", conversion = "last",  method = "chow-lin-maxlog")
predict(mod2)
employedpersons <- predict(mod2)
write.csv(employedpersons, 'employedpersonsm.csv')

data3 <- read.csv('gdpq.csv')
str(data3)
data3 <- data3[,-1]
data_ts3 <- ts(data3, frequency = 4)
data_ts3
mod3 <- td(data_ts3~1, to = "monthly", conversion = "last",  method = "chow-lin-maxlog")
predict(mod3)
gdpgrowth <- predict(mod3)
write.csv(gdpgrowth, 'gdpm.csv')
