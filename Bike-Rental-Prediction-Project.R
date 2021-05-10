#Lets Clear the environment first
rm(list = ls())

# Lets set the working directory
setwd("E:/DataScienceEdwisor/Rscripts")

## Lets Check Our Present working directory
getwd()

# Loading required libraries
x = c("ggplot2", "corrgram", "DMwR", "rpart.plot", "randomForest","e1071","rpart", "gbm", 'sampling', 'DataCombine', 
      'inTrees',"xgboost","readr","stringr", "caret", "car", "class","Metrics","caTools","usdm")

## Lets Load the Packages
lapply(x, require, character.only = TRUE)
rm(x)

## Lets Read The Dataset into the environment
bike_actual = read.csv('day.csv',header = T, na.strings = c(" "," ", "NA"))

# Lets Create another instance copy of our dataset
data = bike_actual

#*******************************************************************************************************#

#************************************** EXPLORATORY DATA ANALYSIS***************************************#

#******************************************************************************************************#
# Lets View dataset and its total observations
View(data) # we have 731 observations and 16 features / predictors

# Lets Check the Dimension of the dataset
dim(data)

# Lets Check the Structure of the dataset
str(data)

# Lets CHeck the summary of the data
summary(data) 

# Lets Check the column names of data
colnames(data)


# Now lets Check the Unique Values of each feature
apply(data, 2,function(x) length(table(x)))

# Seperating categorical and numerical columns
num_cols = c('temp','atemp','hum','windspeed','casual','registered')
cat_cols = c('season','yr','mnth','holiday','weekday','workingday','weathersit')

# Converting Datatypes and assigning them to their actual datatypes
for (i in cat_cols) {
  data[,i] = as.factor(data[,i])
}

# Lets see the datatypes of the cat_cols
#str(cat_cols)

# Lets Check missing values in dataset 
apply(data, 2, function(x) {sum(is.na(x))}) # there are no missing values in our dataset

#*******************************************************************************************************#

#***************************ANALYZING DATA THROUGH VISUALIZATION ***************************************#

#******************************** BAR PLOT ANALYSIS ****************************************************#

# Visulizing data for all categorical columns
# Season wise Count
ggplot(data=data, aes(season, fill=cnt)) + geom_bar(fill='DarkSlateBlue') +
  labs(x='Season', y='Count') + ggtitle("Count in four Seasons->(1:springer, 2:summer, 3:fall, 4:winter)")

# Holiday wise Count
ggplot(data=data, aes(holiday, fill=cnt)) + geom_bar(fill='DarkSlateBlue') +
  labs(x='Holiday', y='Count') + ggtitle("Count vs Holiday")

# Count of season by year wise
ggplot(data=data, aes(season, fill=cnt)) + geom_bar(fill='DarkSlateBlue') + facet_wrap(~yr)+
  labs(x='Season', y='Count') + ggtitle("Count of season by year wise (0: 2011, 1:2012)")

# Count season by workingday
ggplot(data=data, aes(season, fill=cnt)) + geom_bar(fill='DarkSlateBlue') + facet_wrap(~workingday)+
  labs(x='Season', y='Count') + ggtitle("Count of season by workingday ")

# Month Vs Workingday 
ggplot(data=data, aes(mnth, fill=workingday)) + geom_bar(position = 'dodge') +
  labs(x='Month', y='Workingday') + ggtitle("Workingday's in 12 Months")

# Month Vs Weekday 
ggplot(data, aes(x=mnth,fill=weekday)) + geom_bar(position="dodge")+
  theme_bw()+ggtitle("Weekday wise in 12 Months")

# Weekday vs Weathersit
ggplot(data, aes(x=weekday,fill=weathersit)) + geom_bar(position="dodge")+
  theme_bw()+ggtitle("Weekday wise in weathersit")

# Weathersit vs Workingday
ggplot(data, aes(x=weathersit,fill=workingday)) + geom_bar(position="dodge")+
  theme_bw()+ggtitle("Weathersit wise in workingday")+scale_x_discrete(labels=c("cloudy","mist","rainy"))

#******************************** HISTOGRAM PLOT ANALYSIS ****************************************************#

# Visualizing data for all numeric columns
# Histogram Plot for Four Season VS temp 
ggplot(data,aes(x=temp,fill=season))+geom_histogram()+scale_x_discrete(labels=c("2011","2012")) +
  theme_bw()+ggtitle("season vs Temp")

# Histogram Plot for Four Season VS Humidity
ggplot(data,aes(x=hum,fill=season))+geom_histogram()+scale_x_discrete(labels=c("2011","2012")) +
  theme_bw()+ggtitle("season vs Humidity")

# Histogram Plot for Four Season VS windspeed
ggplot(data,aes(x=windspeed,fill=season))+geom_histogram()+scale_x_discrete(labels=c("2011","2012")) + stat_bin(bins = 30) +
  theme_bw()+ggtitle("season vs Windspeed")

#******************************** SCATTER PLOT ANALYSIS ****************************************************#

# Scatter Plot for Temp VS Hum in different season
ggplot(data, aes(x = temp, y = hum, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("temp VS humidity in different seasons")

# Scatter Plot for aTemp VS Hum in different season
ggplot(data, aes(x = atemp, y = hum, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("atemp VS humidity in different seasons")

# Scatter Plot for Temp VS Windspeed in different season
ggplot(data, aes(x = temp, y = windspeed, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("temp VS windspeed in different seasons")

# Scatter Plot for aTemp VS Windspeed in different season
ggplot(data, aes(x = atemp, y = windspeed, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("atemp VS windspeed in different seasons")

# Scatter Plot for Temp VS Count in different season
ggplot(data, aes(x = temp, y = cnt, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("temp VS Count of bike riders in different seasons")

# Scatter Plot for aTemp VS Count in different season
ggplot(data, aes(x = atemp, y = cnt, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("atemp VS Count of bike riders in different seasons")

# Scatter Plot for Windspeed VS Count in different season
ggplot(data, aes(x = windspeed, y = cnt, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("Windspeed VS Count of bike riders in different seasons")

# Scatter Plot for Humidity VS Count in different season
ggplot(data, aes(x = hum, y = cnt, col= season)) +
  geom_point(shape = 17, size = 2)+ggtitle("Humidity VS Count of bike riders in different seasons")

# Scatter Plot for temp VS Count in different years
ggplot(data, aes(x = temp, y = cnt, col= yr)) +
  geom_point(shape = 17, size = 2)+ggtitle("temp VS Count of bike riders in different years")

# Scatter Plot for Windspeed VS Count in different years
ggplot(data, aes(x = windspeed, y = cnt, col= yr)) +
  geom_point(shape = 17, size = 2)+ggtitle("windspeed VS Count of bike riders in different years")

# Scatter Plot for Humidity VS Count in different years
ggplot(data, aes(x = hum, y = cnt, col= yr)) +
  geom_point(shape = 17, size = 2)+ggtitle("Humidity VS Count of bike riders in different years")

#**************************************** DENSITY PLOT ANALYSIS ********************************************#
# density plot for casual and registered in four different seasons 
casualUsers <- ggplot(data, aes(casual, fill = season)) + geom_density(alpha = 0.7) +
               theme(legend.position = "null")+ggtitle('Casual Bikers in Seasons')

registeredUsers <- ggplot(data, aes(registered, fill = season)) + geom_density(alpha = 0.7) + 
                   theme(legend.position = "null")+ggtitle('Registered Bikers in Seasons')

gridExtra:: grid.arrange(casualUsers, registeredUsers, ncol = 2, nrow = 1)

# density plot for casual and registered in different years 
casualUsers <- ggplot(data, aes(casual, fill = yr)) + geom_density(alpha = 0.7) +
  theme(legend.position = "null")+ggtitle('Casual Bikers in Different years')

registeredUsers <- ggplot(data, aes(registered, fill = yr)) + geom_density(alpha = 0.7) + 
  theme(legend.position = "null")+ggtitle('Registered Bikers in Different years')

gridExtra:: grid.arrange(casualUsers, registeredUsers, ncol = 2, nrow = 1)

# density plot for casual and registered in different weathers 
casualUsers <- ggplot(data, aes(casual, fill = weathersit)) + geom_density(alpha = 0.7) +
  theme(legend.position = "null")+ggtitle('Casual Bikers in Different weathers')

registeredUsers <- ggplot(data, aes(registered, fill = weathersit)) + geom_density(alpha = 0.7) + 
  theme(legend.position = "null")+ggtitle('Registered Bikers in Different weathers')

gridExtra:: grid.arrange(casualUsers, registeredUsers, ncol = 2, nrow = 1)


#*******************************************************************************************************#

#********************************************* OUTLIER ANALYSIS ****************************************#

#******************************************************************************************************#
# Loop For Detecting Outliers in numeric columns
for (i in 1:length(num_cols)) 
  {
    assign(paste0("gn",i), 
       ggplot(aes_string(y = (num_cols[i]), x = "cnt"), data = subset(data))+ 
       stat_boxplot(geom = "errorbar", width = 0.5) +
       geom_boxplot(outlier.colour="red", fill = "lightgreen" ,outlier.shape=18,
                       outlier.size=1, notch=FALSE) +
       labs(y=num_cols[i],x="cnt")+
       ggtitle(paste("Box plot of Responded for",num_cols[i])))
}

## Now Lets Plot, plots togeather
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,gn6,ncol=3,nrow=2)

# Loop to detect outlier and replace with na
for(i in c('temp', 'atemp', 'hum', 'windspeed')){
  print(i)
  oa = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  print(length(oa))
  data[,i][data[,i] %in% oa] = NA
}

# Lets impute values in humidity and windspeed
sum(is.na(data)) # 15 missing values in the dataset

# Imputing using mean method
data$hum[is.na(data$hum)] = mean(data$hum,na.rm = T)

data$windspeed[is.na(data$windspeed)] = mean(data$windspeed, na.rm = T)

# Boxplot for Count vs Season
ggplot(data, aes(x = season, y = cnt, fill = season)) +
  geom_boxplot(outlier.color = adjustcolor("black", alpha.f = 0), na.rm = TRUE) +
  ylab("count") +
  ggtitle("Total count of Bike Users in different seasons") +
  scale_fill_manual(values = c("skyblue", "lightgreen", "red", "grey"), 
                    name="Season:",
                    breaks=c(1, 2, 3, 4),
                    labels=c("Spring", "Summer", "Fall","Winter"))

# Boxplot for bike rentals in weekday
ggplot(data, aes(x = workingday, y = cnt, fill =season)) +
  geom_boxplot(outlier.color = adjustcolor("black", alpha.f = 0), na.rm = TRUE) +
  theme_light(base_size = 11) +
  xlab("workingday") +
  ylab("count of Bike Rentals") +
  ggtitle("count of Bike Rentals in workingdays for different seasons") +
  scale_fill_manual(values=c("skyblue", "lightgreen", "red", "grey"), 
                    name="Season:",
                    breaks=c(1, 2, 3, 4),
                    labels=c( "Spring", "Summer", "Fall","Winter")) +
  theme(plot.title = element_text(size = 11, face="bold"))

# Boxplot for bike rentals in weather
ggplot(data, aes(x = weathersit, y = cnt, fill = weathersit)) +
  geom_boxplot(outlier.color = adjustcolor("black", alpha.f = 0), na.rm = TRUE) +
  theme_light(base_size = 11) +
  xlab("Weather") +
  ylab("count of Bike Rentals") +
  ggtitle("COunt of Bike Rentals in different weather situations") +
  scale_fill_manual(values = c("skyblue", "lightgreen", "red", "grey"), 
                    name = "Type of Weather:",
                    breaks = c(1, 2, 3, 4),
                    labels = c("Clear or Cloudy ", 
                               " Mist ", 
                               " Light Rain  or Light Snow ", 
                               "Heavy rain or ice pallets ")) +
  theme(plot.title = element_text(size = 11, face="bold"))

# Boxplot between season and count
ggplot(data,aes(x=season,y=cnt,fill=season))+
  geom_boxplot(outlier.color ="red",outlier.size = 3)+ggtitle("Boxplot for season ~ count")

# Boxplot between season and temp 
ggplot(data,aes(x=season,y=temp,fill=season))+
  geom_boxplot(outlier.color ="red",outlier.size = 3)+ggtitle("Boxplot for season ~ temp")

# Boxplot between season and windspeed
ggplot(data,aes(x=season,y=windspeed,fill=season))+
  geom_boxplot(outlier.color ="red",outlier.size = 3)+ggtitle("Boxplot for season ~ windspeed")

# Boxplot between year and count
ggplot(data,aes(x=yr,y=cnt,fill=season))+
  geom_boxplot(outlier.color ="red",outlier.size = 3)+ggtitle("Boxplot for year ~ count")

# Boxplot between year and temp
ggplot(data,aes(x=yr,y=temp,fill=season))+
  geom_boxplot(outlier.color ="red",outlier.size = 3)+ggtitle("Boxplot for year ~ count")

#*******************************************************************************************************#

#********************************************* FEATURE SELECTION ***************************************#

#*******************************************************************************************************# 
# Lets do correlation analysis using corrgram library

# Heatmap with coefficients
bike_subset = data[,10:16]
bike_subset$casual = as.numeric(bike_subset$casual)
bike_subset$registered = as.numeric(bike_subset$registered)
bike_subset$cnt = as.numeric(bike_subset$cnt)
data_corr <- cor(bike_subset)
corrplot(data_corr, method = 'color', addCoef.col="black")



#*******************************************************************************************************#

#********************************************* CHI-2 TEST OF INDEPENDENCE ******************************#

#*******************************************************************************************************#
# Lets Do Chi2-Square Test of Independence by using for loop for each categorical level
cat_level = data[,cat_cols]

for (i in cat_cols) {
  for (j in cat_cols) {
    #print(i)
    #print(j)
    print(chisq.test(table(cat_level[,i], cat_level[,j]))$p.value)
  }
}

#*******************************************************************************************************#

#********************************************* ANOVA TEST **********************************************#

#*******************************************************************************************************#
# Anova Test Analysis of variance to check how much I.V explain the target variables
# Anova test with Count numerical var with all categorical variables
aov_cnt = aov(cnt~season+yr+mnth+holiday+weekday+workingday+weathersit,data = data)
summary(aov_cnt)

# Anova test with Casual numerical var with all categorical variables
aov_casual = aov(casual~season+yr+mnth+holiday+weekday+workingday+weathersit,data = data)
summary(aov_casual)

# Anova test with Registered numerical var with all categorical variables
aov_reg = aov(registered~season+yr+mnth+holiday+weekday+workingday+weathersit,data = data)
summary(aov_reg)

# anova test for individual categorical features VS target Variable
anova_season =(lm(cnt ~ season, data = data))
summary(anova_season)
anova_yr =(lm(cnt ~ yr, data = data))
summary(anova_yr)
anova_mnth =(lm(cnt ~ mnth, data = data))
summary(anova_mnth)
anova_holiday =(lm(cnt ~ holiday, data = data))
summary(anova_holiday)
anova_weekday =(lm(cnt ~ weekday, data = data))
summary(anova_weekday)
anova_workingday =(lm(cnt ~ workingday, data = data))
summary(anova_workingday)
anova_weathersit =(lm(cnt ~ weathersit, data = data))
summary(anova_weathersit)

#*******************************************************************************************************#

#************************* MULTICOLLINEARITY CHECK USING (VIF) VARIANCE INFLATION FACTOR****************#

#*******************************************************************************************************#
# VIF Check for multicollinearity between features
vif(data[,c(10,11,12,13)])
vifcor(data[,c(10,11,12,13)])

#*******************************************************************************************************#

#***************************************** FEATURE IMPORTANCES *****************************************#

#*******************************************************************************************************#

# Cheking feature importances by excluding instant,dteday,casual and registered features
imp_feats = randomForest(cnt ~ ., data = data[,-c(1,2,14,14)],
                         ntree=300, keep.forest = FALSE, importance = TRUE)
imp_feats_df = data.frame(importance(imp_feats, type=1))
#View(imp_feats_df)

#*******************************************************************************************************#

#****************************************** DIMENSION REDUCTION ****************************************#

#*******************************************************************************************************#
data = subset(data, select = -c(instant,dteday,holiday,atemp,casual,registered))
#View(data)


#*******************************************************************************************************#

#****************************************** FEATURE SCALING ********************************************#

#*******************************************************************************************************#
# Normality Check Using qqnorm
# Lets see the target variable distribution VS all Independent variables 
hist(data$cnt)
# qqnorm normality check for temperature feature
qqnorm(data$temp)
hist(data$temp)
# qqnorm normality check for humidity feature
qqnorm(data$hum)
hist(data$hum)
# qqnorm normality check for windspeed feature
qqnorm(data$windspeed)
hist(data$windspeed)
# We are not going to scale the data as the data is already scaled and it is uniformly distributed


# Lets Clean-up the environment before we proceed further for model development

rmExcept(c("bike_actual","data"))

#*******************************************************************************************************#

#****************************************** SAMPLINNG OF DATA ******************************************#

#*******************************************************************************************************#
# Splitting into train and test sets
set.seed(1234)
split_data = sample.split(data$cnt, SplitRatio = 0.80)
train = subset(data, split_data == TRUE)   
test = subset(data, split_data == FALSE)  
  
#*******************************************************************************************************#

#********************************************* MODEL DEVELOPMENT ***************************************#

#*******************************************************************************************************#  
# Lets Define MAPE function 
MAPE = function(responded, predicted) {
      mean(abs((responded - predicted)/responded))*100
}

# Lets create dummy columns 
dumy = dummyVars(~., data)
dummy_df = data.frame(predict(dumy, data))

# Dividing data for linear regression model
set.seed(1234)
split_data = sample.split(dummy_df$cnt, SplitRatio = 0.80)
dummy_df_train = subset(dummy_df, split_data == TRUE)   
dummy_df_test = subset(dummy_df, split_data == FALSE)  


################################
## ~ Linear Regression Model~ ##
################################

LR_Model = lm(cnt~., data = dummy_df_train)

# Lets See the summary of Regression Model
summary(LR_Model)

# Saving linear regression model summary to hard disk
write(capture.output(summary(LR_Model)), "LR_Summary.txt")

# Change the panel layout to 2 x 2
par(mfrow = c(2, 2))

# Lets Plot Linear regression model
plot(LR_Model)  

# Lets Predict Linear regression on train data
LR_Predictions = predict(LR_Model, dummy_df_train[,-34])
plot(dummy_df_train$cnt, LR_Predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'LR Model on train data')

# Lets Evaluate Linear regression model on train dataset
postResample(LR_Predictions, dummy_df_train$cnt)
#    RMSE         Rsquared      MAE 
# 758.4540191   0.8437638   556.8700427
# Mean Absolute Percentage Errror 
MAPE(dummy_df_train$cnt, LR_Predictions) # MAPE_LR_Model_train_data = 42.93592

# Linear Regression Predictions on test data set
LR_Predictions_test = predict(LR_Model, dummy_df_test[,-34])
plot(dummy_df_test$cnt, LR_Predictions_test,xlab = 'Actual values',ylab = 'Predicted values',main = 'LR Model on test data')

# Lets Evaluate linear regression model on test dataset
postResample(LR_Predictions_test, dummy_df_test$cnt)
#   RMSE         Rsquared      MAE 
# 805.2292082   0.8480556  603.4942786
# Mean Absolute Percentage Errror 
MAPE(dummy_df_test$cnt, LR_Predictions_test) # MAPE_LR_Model_test_data = 17.45633

#Stepwise Model Selection
# Now performs stepwise model selection by AIC with both directions(Forward, Backward)
library(MASS)
LR_Model_AIC<-stepAIC(LR_Model, direction="both")
summary(LR_Model_AIC)

# Step wise selection forward and backward plot
plot(LR_Log_AIC)

# Saving Step-Wise linear regression model summary to hard disk
write(capture.output(summary(LR_Model_AIC)), "LR_StepWise_Summary.txt")

#Prediction on Validation / test set
# Apply prediction on validation set
lR_Predict_test <- predict(LR_Model_AIC, newdata = dummy_df_test[,-34])
postResample(lR_Predict_test, dummy_df_test$cnt)
#  RMSE          Rsquared        MAE 
# 801.883727    0.847627     601.222217

# Mean Absolute Percentage Errror of Step wise model selection direction both
MAPE(dummy_df_test$cnt, lR_Predict_test) # MAPE_LR_Model__AIC_test_data = 17.6618

# Lets Check the summary of the predicted count values
cat("\n")
print("summary of predicted count values")
summary(lR_Predict_test)
#  Min.   1st Qu.  Median    Mean   3rd Qu.    Max. 
# -1278    3400     4545     4578    6119      7551

# Lets Check the summary of the actual count values
cat("\n")
print("summary of actual count values")
summary(dummy_df_test$cnt)
# Min.  1st Qu.  Median    Mean   3rd Qu.    Max. 
# 441    3524     4672     4796    6426      8714 

#From above summary we saw negative values of predicted count.
# We don't want negative values as forecast for bike count. Replace all negative numbers with 1 
Output2Pos = lR_Predict_test
Output2Pos[lR_Predict_test<=0] = 1

# Check again the summary of predicted count values
print("summary of predicted count values after replaced the negative values with 1")
summary(Output2Pos)
# Min. 1st Qu.  Median    Mean   3rd Qu.    Max. 
# 1    3400     4545      4587    6119      7551

# As we replaced the negative values, the rmse value got reduced
print("root-mean-square error value after replaced the negative values")
print(rmse(dummy_df_test$cnt,Output2Pos))
# 789.3738

#If we want to penalize under-prediction of demand, rmsle might be a better metric
LR_rmsle=rmsle(dummy_df_test$cnt,Output2Pos)
print("root-mean-square-log error value after replaced the negative values")
print(LR_rmsle)
# RMSLE after replacing the negative values with 1 = 0.5201974

#Log Transformation
# Since we got negative predicted values, let's do log transformation and run regression model again
LR_Log = lm(log(cnt)~., data = dummy_df_train)

# Now performs stepwise model selection on log model
LR_Log_AIC = stepAIC(LR_Log, direction="both")

#Prediction on Validation / test set
lR_Predict_Log_test = predict(LR_Log_AIC, newdata = dummy_df_test)

# As the predicted values are in log format, use exponential(exp) to convert from log to non-log values
LR_Predict_log_nonlog = exp(lR_Predict_Log_test)

# Let's check the summary of predicted count values, it shows there are no negative values
print("summary of predicted count values after log transformation")
summary(LR_Predict_log_nonlog)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 487    3030    4106    4548    6147    9334

# Check rmsle value again, it got reduced from 0.5201974 to 0.2388554
LR_nonlog_rmsle<-rmsle(dummy_df_test$cnt,LR_Predict_log_nonlog)
print("root-mean-square-log error value after log transformation")
print(LR_nonlog_rmsle)
# RMSLE After log transformation & again converted to exp : 0.2388554

#Residual vs Fitted plot
# Let's check the Residual vs Fitted plot
# It shows some points forms a straight lines
# If you select bottom straight line points using "identify", you will find that the bike rent count is 1
# and next straight line points will have bike rent count of 2. 
plot(LR_Log)


#######################################
## ~ Decision Tree Regression Model~ ##
#######################################
DT_Model = rpart(cnt~., data = train, method = "anova")

# lets save decision tree results back to hardisk 
write(capture.output(summary(DT_Model)), "DTRules.txt")

# Plotting Decision tree GRAPH
plot_dt = rpart.plot(DT_Model, type = 5, digits = 2, fallen.leaves = TRUE)

# Decision Tree Prediction on train data
DT_Predictions = predict(DT_Model, train[,-10])
plot(train$cnt, DT_Predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'DT Model on train data')

# Lets Evaluate Decision tree regression model on train dataset
postResample(DT_Predictions, train$cnt)
#   RMSE           Rsquared     MAE 
# 851.7111567   0.8029812   632.6669863
MAPE(train$cnt, DT_Predictions) # MAPE_DT_Model_train_data = 56.38422

# Decision Tree Prediction on test data
DT_Predictions_test = predict(DT_Model, test[,-10])
plot(test$cnt, DT_Predictions_test,xlab = 'Actual values',ylab = 'Predicted values',main = 'DT Model on test data')

# Lets Evaluate Decision tree regression model on test dataset
postResample(DT_Predictions_test, test$cnt)
#   RMSE        Rsquared         MAE 
# 952.3817605   0.7683867    720.0017702 
MAPE(test$cnt, DT_Predictions_test) # MAPE_DT_Model_test_data = 26.99524

########################################
## ~ Support Vector Regression Model~ ##
########################################
SVM_Model = svm(cnt~., train)

# SVM Prediction on train data
SVM_Predictions = predict(SVM_Model, train[,-10])
plot(train$cnt, SVM_Predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'SVM Model on train data')

# Lets Evaluate SVM model on train dataset
postResample(SVM_Predictions, train$cnt)
#    RMSE        Rsquared         MAE 
# 626.0047375   0.8961997    426.7444956 
MAPE(train$cnt, SVM_Predictions) # MAPE_SVM_Model_train_data = 37.97459

# SVM Prediction on test data
SVM_Predictions_test = predict(SVM_Model, test[,-10])
plot(test$cnt, SVM_Predictions_test,xlab = 'Actual values',ylab = 'Predicted values',main = 'SVM Model on test data')

# Lets Evaluate SVM model on test dataset
postResample(SVM_Predictions_test, test$cnt)
#   RMSE          Rsquared      MAE 
# 690.0739705   0.8800138   502.0440156 
MAPE(test$cnt, SVM_Predictions_test) # MAPE_SVM_Model_test_data = 15.4358

########################################
######## ~ Random Forest Model~ ########
########################################
RF_Model = randomForest(cnt~., train, importance = TRUE, ntree = 500)
RF_Model
# Mean of squared residuals: 503733
# % Var explained: 86.32

# Extracting rules from random forest model
tree_list = RF2List(RF_Model)
# Extract Rules
exct = extractRules(tree_list, train[,-10])
# 4788 rules (length<=6) were extracted from the first 100 trees.
# Visualizing some of the rules
exct[1:2,]
# making rules more readable
read_rules = presentRules(exct,colnames(train))
read_rules[1:2,]
# Rule Metrics
rule_metric = getRuleMetric(exct,train[,-10],train$cnt)
rule_metric[1:2,]

# Variable Importances for random forest model plot
varImpPlot(RF_Model)


# Random forest Prediction on train data
RF_Predictions = predict(RF_Model, train[,-10])
plot(train$cnt, RF_Predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'RF Model on train data')


# Lets Evaluate RF model on train dataset
postResample(RF_Predictions, train$cnt)
#    RMSE        Rsquared         MAE 
# 333.8655091   0.9746288    235.2725488 
MAPE(train$cnt, RF_Predictions) # MAPE_RF_Model_train_data = 22.57292


# Random forest Prediction on test data
RF_Predictions_test = predict(RF_Model, test[,-10])
plot(test$cnt, RF_Predictions_test,xlab = 'Actual values',ylab = 'Predicted values',main = 'RF Model on test data')

# Lets Evaluate SVM model on test dataset
postResample(RF_Predictions_test, test$cnt)
#  RMSE           Rsquared         MAE 
# 662.707111      0.900422    498.064969
MAPE(test$cnt, RF_Predictions_test) # MAPE_RF_Model_test_data = 18.46774

###################################################################
######## ~ Hyper Parameter tuning for Random Forest Model~ ########
###################################################################
library(caret)
control = trainControl(method="repeatedcv", number=10, repeats=3)
RF_Model_HP = caret::train(cnt~., data = train, method = "rf",trControl = control)
RF_Model_HP$bestTune #   mtry 2   14
rf_predict = predict(RF_Model_HP, test[,-10])

# MOdel RMSE error plot
plot(RF_Model_HP)

# Tuned Random forest Prediction on train data
rf_predictions = predict(RF_Model_HP, train[,-10])
plot(train$cnt, rf_predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'Tuned RF Model on train data')


# Lets Evaluate Tuned RF model on train dataset
postResample(rf_predictions, train$cnt)
#    RMSE        Rsquared         MAE 
# 334.0120332   0.9742905    232.8790834 
MAPE(train$cnt, rf_predictions) # MAPE_Tuned_RF_Model_train_data = 24.27306

# Random forest tuned model Prediction on test data
plot(test$cnt, rf_predict,xlab = 'Actual values',ylab = 'Predicted values',main = 'RF Tuned Model on test data')

# Lets Evaluate Rf Tuned model on test dataset
postResample(rf_predict, test$cnt)
#  RMSE           Rsquared         MAE 
# 629.9692316     0.9033033   466.7887276

# Mean Absolute Percentage error on Final Model Random Forest Hyper parameter model
MAPE(test$cnt, rf_predict) # MAPE_RF_Tuned_Model_test_data = 17.33545

###################################################################
######## ~ Hyper Parameter tuning for SVM Model~ ##################
###################################################################
library(LiblineaR)
control = trainControl(method="repeatedcv", number=10, repeats=3)
SVM_Model_HP = caret::train(cnt~., data = train, method = "svmLinear3",trControl = control)
SVM_Model_HP$bestTune 

plot(SVM_Model_HP)
# cost Loss
# 6    1   L2

# Tuned SVM Prediction on train data
svm_predictions = predict(SVM_Model_HP, train[,-10])
plot(train$cnt, SVM_Predictions,xlab = 'Actual values',ylab = 'Predicted values',main = 'Tuned SVM Model on train data')

# Lets Evaluate TUned SVM model on train dataset
postResample(svm_predictions, train$cnt)
#    RMSE        Rsquared         MAE 
# 760.4524668   0.8430114     557.6934236 
MAPE(train$cnt, svm_predictions) # MAPE_Tuned_SVM_Model_train_data = 44.26525

# SVM Predictions on Test Data
svm_predict = predict(SVM_Model_HP, test[,-10])
# SVM tuned model Prediction on test data
plot(test$cnt, svm_predict,xlab = 'Actual values',ylab = 'Predicted values',main = 'SVM Tuned Model on test data')

# Lets Evaluate Rf Tuned model on test dataset
postResample(svm_predict, test$cnt)
#  RMSE           Rsquared         MAE 
# 797.3890684   0.8503455     600.1752801

# Mean Absolute Percentage error on Final Model SVM Hyper parameter model
MAPE(test$cnt, svm_predict) # MAPE_SVM_Tuned_Model_test_data = 17.66163


# All models predictions on the train data 
Predictions_All_Models_train_data = data.frame(LR_Predictions, DT_Predictions, SVM_Predictions, rf_predictions, svm_predictions)

# All models predictions on the unseen data that is actual values VS the Predicted Values
Predictions_All_Models_test_data = data.frame(LR_Predictions_test, lR_Predict_test, LR_Predict_log_nonlog, DT_Predictions_test,
                                    SVM_Predictions_test, RF_Predictions_test,rf_predict,svm_predict)

# Lets save the results back to hard disk
## Setting up the Working Directory
setwd("E:/DataScienceEdwisor/PROJECT-2/R")
write.csv(train, "train_data.csv", row.names = F)
write.csv(test, "test_data.csv", row.names = F)
write.csv(Predictions_All_Models_train_data, "Predictions_train_data.csv", row.names = F)
write.csv(Predictions_All_Models_test_data, "Predictions_test_data.csv", row.names = F)



