# econometrics_deep_learning_comparison
Forecasting economic growth using econometrics and deep learning

## Data preprocessing

* A timeseries dataset consisting of various economic but purely numerical variables was used.
* Given the nature of the data, there were no missing data or outliers.
* Data obtained was not homogeneous in terms of data frequency (some daily, some monthly, some quarterly)
* Daily data were aggregated to monthly average values using Excel and quarterly data were converted using temporal disaggregation using R.
* A total of 21 years worth of data was used and split into 18 years for training and 3 years for testing.

## Model Implementation

* A vector autoregression (VAR) model was tested as the econometric method while a seq2seq LSTM Network was tested as the deep learning method. 
* The VAR model was developed and had its assumptions tested (stationarity, autocorrelation, and heteroskedasticity).
* Due to certain violations of the assumptions, variables that were found to be in violation had to be removed.
* The model was rerun and evaluated before forecasting the economic growth for the year 2021.

* The LSTM Network was developed and additional data preprocessing was carried out as necessitated by the model.
* Tuning of the model hyperparameters were done by testing various hidden and dense layers across various epoch values. 
* A form of validation for timeseries analysis known as walk-forward validation was implemented for the model.
* The model was evaluated using the same evaluation metrics as the VAR model and the economic growth values for the year 2021 were forecasted.
* The evaluation metrics and the forecasted values of each model were then compared to each other as well as the official economic growth values as published by the Department of Statistics, Malaysia (DOSM).
