rm(list = ls())

library(caret)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(ggcorrplot)
library(dplyr)
library(ggcorrplot)
library (glmnet)
library(MASS)
library(readr)
library(kknn) 
library(e1071)
library(tree)

#######################################################################################

crab_data <- read.csv("/Users/alexparson/Documents/UT MSBA 2025/STA 380/Group Project/CrabAgePrediction.csv")
print(head(crab_data))

#######################################################################################
# EXPLORATORY DATA ANALYSIS - 
#____________________________

# Data summary and check for missing values
# -----------------------------------------
summary(crab_data)
dim(crab_data)

colSums(is.na(crab_data)) # there are no missing values in any columns

# code to plot scatter plots for each variable in dataset with target variable Age
#----------------------------------------------------------------------------------

# Get the names of predictor variables (excluding 'Age')
predictors <- setdiff(names(crab_data), "Age")


dir.create("regression_plots", showWarnings = FALSE)


for (predictor in predictors) {
  p <- ggplot(crab_data, aes_string(x = predictor, y = "Age")) +
    geom_point() +
    geom_smooth(method = "lm", color = "blue") +
    labs(x = predictor, y = "Age", title = paste("Regression Plot of Age vs", predictor))
  
  print(p)
  
  ggsave(filename = paste("regression_plots/plot_", predictor, ".png", sep = ""), plot = p)
}

# Plot the correlation matrix using corrplot
# --------------------------------------------

numeric_data <- crab_data[sapply(crab_data, is.numeric)]

correlation_matrix <- cor(numeric_data, use = "complete.obs")

print(correlation_matrix)

corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", # Add correlation coefficient values
         number.cex = 0.7, # Size of correlation coefficients
         col = colorRampPalette(c("red", "white", "blue"))(200)) # Color palette

ggcorrplot(correlation_matrix, method = "circle", 
           type = "upper", 
           lab = TRUE, # Add correlation coefficient values
           lab_size = 3, # Size of correlation coefficients
           colors = c("red", "white", "blue"), # Color palette
           title = "Correlation Matrix", 
           ggtheme = theme_minimal())


ggplot(crab_data, aes(x = Sex, y = Age)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Sex", x = "Sex", y = "Age")

#######################################################################################
# General parameter settings
#######################################################################################
set.seed(1234)

#Creating Test-train split
train_index <- sample(1:nrow(crab_data), 0.8 * nrow(crab_data))
train_data <- crab_data[train_index, ]
test_data <- crab_data[-train_index, ]

# cross-validation settings
kcv = 10

cv_folds = createFolds(train_data$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

#######################################################################################
# BAGGING WITH RANDOM FORESTS (MANASA)
#######################################################################################
#NOTE: Bagging is simply a special case of a random forest with m = p. 
#Therefore, the function can randomForest() random be used to perform both random forests and bagging.

# MODEL FIT 
#___________
# Implementing the bagging algorithm using Random forest approach where m=p 


num_predictors <- ncol(crab_data) - 1
bag.crab_data <- randomForest(Age ~ ., data = train_data, trControl = fit_control, mtry = num_predictors, ntree = 200, importance = TRUE)

print(bag.crab_data)

oob_error <- bag.crab_data$mse

if(all(is.finite(oob_error))) {
  plot(oob_error, type = "l", xlab = "Number of Trees", ylab = "OOB Error Rate", main = "OOB Error Plot", col = "blue")
  
  
  abline(h = min(oob_error), col = "red", lty = 2)
} else {
  print("OOB error rates contain non-finite values.")
}

yhat.bag <- predict(bag.crab_data, newdata = test_data)

RMSE_bag <- sqrt(mean((yhat.bag - test_data$Age)^2))
print(RMSE_bag)

plot(yhat.bag, test_data$Age, xlab = "Predicted Age", ylab = "Actual Age", main = "Predicted vs Actual Age")
abline(0, 1, col = "orange")


importance_vals <- importance(bag.crab_data)
print(importance_vals)

varImpPlot(bag.crab_data, main = "Variable Importance Plot")

####################################################################
# LASSO REGRESSION
####################################################################

# Without Shell weight ratio
#----------------------------

# Check for non-numeric columns and convert them if needed
crab_data_clean <- crab_data
crab_data_clean[] <- lapply(crab_data_clean, function(x) if(is.factor(x)) as.numeric(as.character(x)) else x)


x <- as.matrix(crab_data_clean[, setdiff(names(crab_data_clean), "Age")])
y <- crab_data_clean$Age

train <- sample(1:nrow(crab_data_clean), 0.8 * nrow(crab_data_clean))
test <- -train

x.train <- x[train, ]
y.train <- y[train]
x.test <- x[test, ]
y.test <- y[test]

# Define a grid of lambda values for Lasso
grid <- 10^seq(10, -2, length = 100)

lasso.mod <- glmnet(x.train, y.train, alpha = 1, lambda = grid)
plot(lasso.mod, xvar = "lambda", label = TRUE)

# Perform cross-validation to find the best lambda
set.seed(1234)
cv.out <- cv.glmnet(x.train, y.train, alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min

# Make predictions using the best lambda
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x.test)

# Calculate Mean Squared Error
RMSE_lasso <- sqrt(mean((lasso.pred - y.test)^2))
#print(paste("Mean Squared Error: ", mse))
#rmse <- sqrt(mse)
print(paste("Root Mean Squared Error: ", RMSE_lasso))

# Extract and display coefficients for the best lambda
out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s = bestlam)

# Print the first few coefficients
print("Lasso Coefficients for the best lambda:")
print(lasso.coef)


############################################################
# End of Manasa's code
############################################################

#Multiple Linear Regression

#running multiple linear regression
lmFit <- train(Age ~., data = train_data, method = 'lm')

#Printing out the summary of test data regression
print(summary(lmFit))


#Using our test data to make prediction
predictions <- predict(lmFit, newdata=test_data)


#Seeing how well our MLR model fits the test data with (RMSE) 
RMSE_lm <- sqrt(mean((test_data$Age - predictions)^2))

print(paste("RMSE:", RMSE_lm))

#Variable importance
importance <- varImp(lmFit)
plot(importance)

#Plotting actual to predicted
plot1 <- ggplot(data = test_data, aes(x = Age, y = predictions)) +
  geom_point(color = 'black') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = 'dashed') +
  ggtitle('Actual vs Predicted Age') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()
plot1

##############################################################################
## KNN 
##############################################################################

#Training the model
kcv = 10

cv_folds = createFolds(train_data$Age,
                       k = kcv)
ctrl <- trainControl(method = "cv",
                     number=10)
knnFit <- train(Age~., data = train_data, method = "knn", trControl = ctrl, tuneLength = 20)

best_k <- knnFit$bestTune$k
best_rmse_knn <- min(knnFit$results$RMSE)
results <- knnFit$results
k_values <- seq(5, max(results$k),by=4)
results$SE <- results$RMSESD / sqrt(fit_control$number)

print(paste("Best k:", best_k))
print(paste("Best RMSE:", best_rmse_knn))

predictions <- predict(knnFit, newdata = test_data)
RMSE_knn <- RMSE(predictions, test_data$Age)

#Graph 1
plot(knnFit)

#Graph 2
ggplot(results, aes(x = k, y = RMSE)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "red") +
  labs(title = "Number of Neighbors vs RMSE",
       x = "K",
       y = "RMSE") +
  theme_minimal() 

#Graph 3
ggplot(results, aes(x = k, y = RMSE)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = RMSE - SE, ymax = RMSE + SE), width = 0.2) +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "red") +
  labs(title = "RMSE and Standard Error at Each k",
       x = "Number of Neighbors (k)",
       y = "Root Mean Squared Error (RMSE)") +
  theme_minimal() 

#######################################################################
# End of Tom's code
#######################################################################

# 6. Run a single tree regression on training data, using all variables. Provide a summary and plot output
treeModel <- rpart(Age ~ ., 
                   data = train_data,
                   method = "anova",
                   xval = 10, # number of cross-validations
                   control = rpart.control(cp=0.001, minsplit = 5))
summary(treeModel)
rpart.plot(treeModel)
plotcp(treeModel)

#Finding the optimal cp using the one sd rule
best_cp_ix = which.min(treeModel$cptable[,4]) # "Best"
treeModel$cptable[best_cp_ix,4]

tol = treeModel$cptable[best_cp_ix,4] + treeModel$cptable[best_cp_ix,5]
treeModel$cptable[treeModel$cptable[,4]<tol,][1,]
best_cp_onesd = treeModel$cptable[treeModel$cptable[,4]<tol,][1,1]

pruned_tree = prune(treeModel, cp=best_cp_onesd)

rpart.plot(pruned_tree)
summary(pruned_tree)
pruned_tree$cptable



# 10. Write a random forest regression on training data, where mtry = 3. Calculate RMSE
# Creating a custom grid
rf_grid = data.frame(mtry = c(2,3,4,5,6,7))
rf_model <- train( Age ~ ., data = train_data, 
                   method = "rf", 
                   trControl = fit_control,
                   tuneGrid = rf_grid,
                   ntree = 50 # best RMSE at 50 : 2.23 with 4 var, RMSE at 500: 2.17 with 4 var, RMSE at 1000: 2.17 with 4 var
)

# Getting a plot of CV error estimates
ggplot(rf_model)

# Adding +/- one se
best = rf_model$results[which.min(rf_model$results$RMSE),]
onesd = best$RMSE + best$RMSESD/sqrt(kcv)

ggplot(rf_model) + 
  geom_segment(aes(x=mtry, 
                   xend=mtry, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv)), 
               data=rf_model$results) + 
  geom_hline(yintercept = onesd, linetype='dotted')

rf_model$finalModel

# 11. Provide a variable importance plot
varImpPlot(rf_model$finalModel)

# 7. Calculate RMSE of each
# Predict MSE of Single Tree
predictions_tree <- predict(treeModel, newdata = test_data)
RMSE_tree <- sqrt(mean((predictions - test_data$Age)^2))
print(RMSE_tree)

# Predicting MSE of Pruned Tree
predictions <- predict(pruned_tree, newdata = test_data)
RMSE_pruned <- sqrt(mean((predictions - test_data$Age)^2))
print(RMSE_pruned)

# Calculate RMSE with OOS data
predictions_rf <- predict(rf_model, newdata=test_data)
RMSE_rf <- sqrt(mean( (test_data$Age - predictions_rf)^2 ))
print(RMSE_rf)

# Plot of Predicted vs Actual
plot(predictions_rf, test_data$Age, xlab = "Predicted Age", ylab = "Actual Age", main = "Predicted vs Actual Age")
abline(0, 1, col = "orange")


############################################################
# Boosting Model
############################################################

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid_2 <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                           n.trees = c(1000, 1500, 2000, 2500, 3000, 3500), 
                           shrinkage = c(.05, .1),
                           n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_3 <- train(Age ~ ., data = train_data, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid_2,
                  verbose = FALSE)

print(gbmfit_3$bestTune)

best_ix = which.min(gbmfit_3$results$RMSE)
best = gbmfit_3$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit_3$results$RMSE<onese_max_RMSE

print(gbmfit_3$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)

# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit_3$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")

## Validation
gbm_yhat = predict(gbmfit_3, newdata=crab_test)
# So is validation RMSE
RMSE_boost <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE_boost)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_3)
ggplot(gbm_imp)


##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train2$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

##Fit Model
gbmfit <- train( Age ~ ., data = crab_train2, 
                 method = "gbm", 
                 trControl = fit_control,
                 verbose = FALSE)

best_ix = which.min(gbmfit$results$RMSE)
best = gbmfit$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit$results$RMSE<onese_max_RMSE

print(gbmfit$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)



# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")

print(gbmfit$bestTune)

ggplot(gbmfit)


## Validation
gbm_yhat = predict(gbmfit, newdata=crab_test2)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test2$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit)
ggplot(gbm_imp)

################################################
#Boosting Best Model, default parameters 
################################################


##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train2$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid_2 <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                           n.trees = c(1000, 1500, 2000, 2500, 3000, 3500), 
                           shrinkage = c(.05, .1),
                           n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_3 <- train(Age ~ ., data = crab_train2, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid_2,
                  verbose = FALSE)

print(gbmfit_3$bestTune)

best_ix = which.min(gbmfit_3$results$RMSE)
best = gbmfit_3$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit_3$results$RMSE<onese_max_RMSE

print(gbmfit_3$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)

# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit_3$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")


## Validation
gbm_yhat = predict(gbmfit_3, newdata=crab_test2)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test2$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_3)
ggplot(gbm_imp)

#####################################################################
# Ridge Regression Optimal Model
#####################################################################

X <- model.matrix(Age ~ . - 1, data = crab_data) # Create a design matrix without intercept
y <- crab_data$Age

# Split the data into training and testing sets
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardize the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)

# Define the lambda sequence for cross-validation
lambda_seq <- seq(0.001, 0.3, by = 0.001)

# Perform ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq, nfolds = 10)

# Best lambda value
#best_lambda <- ridge_cv$lambda.min

#THIS IS THE ACTUAL BEST ONE NEED TO REPLECATE THE CODE WITH THE FOLLOWING
# It does the less complex one within one standard error

#best_lambda <- ridge_cv$lambda.1se
best_lambda <- ridge_cv$lambda.1se

# Train the final model with the best lambda
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


# Coefficients for the model
coef(ridge_model)

# Predict and evaluate
y_train_pred <- predict(ridge_model, X_train)
y_test_pred <- predict(ridge_model, X_test)

# Calculate R^2
train_r2 <- cor(y_train, y_train_pred)^2
test_r2 <- cor(y_test, y_test_pred)^2

# Calculate RMSE
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
RMSE_ridge <- sqrt(mean((y_test - y_test_pred)^2))

# Print R^2 and RMSE scores
print(paste("Training R^2:", train_r2))
print(paste("Testing R^2:", test_r2))
print(paste("Training RMSE:", train_rmse))
print(paste("Testing RMSE:", RMSE_ridge))



# Visualize the model's prediction performance
train_data <- data.frame(Actual = y_train, Predicted = y_train_pred)
test_data <- data.frame(Actual = y_test, Predicted = y_test_pred)


# Plot for training set
ggplot(train_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Training Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot for testing set
ggplot(test_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'green', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Testing Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot to show the chosen lambda value
plot(ridge_cv)
abline(v = log(best_lambda), col = "red", lwd = 2)

plot(ridge_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), col = "red", lwd = 2)
legend("topright", legend = c("Best Lambda"), col = c("red"), lty = c(1), lwd = c(2))

#######################################################################
# Collection of OOS results
#######################################################################

rmse_table_standard <- data.frame(
  Model = c("Bagging", "Lasso", "Linear Regression", "KNN", "Single Tree", "Pruned Tree", "Random Forest", "Ridge"), #Boosting
  RMSE = c(RMSE_bag, RMSE_lasso, RMSE_lm, RMSE_knn, RMSE_tree, RMSE_pruned, RMSE_rf, RMSE_ridge)#,RMSE_boost)
)

print(rmse_table_standard)
#######################################################################
# Testing with weight/shell variable, see if it improves our model
#######################################################################

# Implementing the bagging algorithm using Random forest approach with Shell_Weight_Ratio

crab_data_2 <- mutate(crab_data, Shell_Weight_Ratio = crab_data$`Shell.Weight` / Weight)

# creating new sample set
train_index <- sample(1:nrow(crab_data_2), 0.8 * nrow(crab_data_2))
train_data_2 <- crab_data_2[train_index, ]
test_data_2 <- crab_data_2[-train_index, ]

# Number of folds
kcv = 10

cv_folds = createFolds(train_data_2$Age,
                       k = kcv)

fit_control_2 <- trainControl(
  method = "cv",
  indexOut = cv_folds)

num_predictors <- ncol(crab_data_2) - 1

if(anyNA(crab_data)) {
  crab_data <- na.omit(crab_data_2)
}

num_predictors <- ncol(crab_data_2) - 1
bag.crab_data_2 <- randomForest(Age ~ ., data = train_data_2, trControl = fit_control_2, mtry = num_predictors, ntree = 200, importance = TRUE)

print(bag.crab_data_2)

oob_error <- bag.crab_data_2$mse

if(all(is.finite(oob_error))) {
  plot(oob_error, type = "l", xlab = "Number of Trees", ylab = "OOB Error Rate", main = "OOB Error Plot", col = "blue")
  
  
  abline(h = min(oob_error), col = "red", lty = 2)
} else {
  print("OOB error rates contain non-finite values.")
}


yhat.bag_2 <- predict(bag.crab_data_2, newdata = test_data_2)

RMSE_bag_2 <- sqrt(mean((yhat.bag_2 - test_data_2$Age)^2))
print(RMSE_bag_2)

plot(yhat.bag_2, test_data_2$Age, xlab = "Predicted Age", ylab = "Actual Age", main = "Predicted vs Actual Age")
abline(0, 1, col = "orange")


importance_vals <- importance(bag.crab_data_2)
print(importance_vals)

varImpPlot(bag.crab_data_2, main = "Variable Importance Plot")

####################################################################
# LASSO REGRESSION
####################################################################


# Check for non-numeric columns and convert them if needed
crab_data_clean_2 <- crab_data_2
crab_data_clean_2[] <- lapply(crab_data_clean_2, function(x) if(is.factor(x)) as.numeric(as.character(x)) else x)


x_2 <- as.matrix(crab_data_clean_2[, setdiff(names(crab_data_clean_2), "Age")])
y_2 <- crab_data_clean_2$Age

train <- sample(1:nrow(crab_data_clean_2), 0.8 * nrow(crab_data_clean_2))
test <- -train

x.train_2 <- x_2[train, ]
y.train_2 <- y_2[train]
x.test_2 <- x_2[test, ]
y.test_2 <- y_2[test]

# Define a grid of lambda values for Lasso
grid <- 10^seq(10, -2, length = 100)

lasso.mod_2 <- glmnet(x.train_2, y.train_2, alpha = 1, lambda = grid)
plot(lasso.mod_2, xvar = "lambda", label = TRUE)

# Perform cross-validation to find the best lambda
set.seed(1234)
cv.out_2 <- cv.glmnet(x.train_2, y.train_2, alpha = 1)
plot(cv.out_2)
bestlam_2 <- cv.out_2$lambda.min

# Make predictions using the best lambda
lasso.pred_2 <- predict(lasso.mod_2, s = bestlam_2, newx = x.test_2)

# Calculate Mean Squared Error
RMSE_lasso_2 <- sqrt(mean((lasso.pred_2 - y.test_2)^2))
#print(paste("Mean Squared Error: ", mse))
#rmse <- sqrt(mse)
print(paste("Root Mean Squared Error: ", RMSE_lasso_2))

# Extract and display coefficients for the best lambda
out_2 <- glmnet(x_2, y_2, alpha = 1, lambda = grid)
lasso.coef_2 <- predict(out_2, type = "coefficients", s = bestlam)

# Print the first few coefficients
print("Lasso Coefficients for the best lambda:")
print(lasso.coef_2)


############################################################
# End of Manasa's code
############################################################

#Multiple Linear Regression

#running multiple linear regression
lmFit_2 <- train(Age ~., data = train_data_2, method = 'lm')

#Printing out the summary of test data regression
print(summary(lmFit_2))


#Using our test data to make prediction
predictions_2 <- predict(lmFit_2, newdata=test_data_2)


#Seeing how well our MLR model fits the test data with (RMSE) 
RMSE_lm_2 <- sqrt(mean((test_data_2$Age - predictions_2)^2))

print(paste("RMSE:", RMSE_lm_2))

#Variable importance
importance <- varImp(lmFit_2)
plot(importance)

#Plotting actual to predicted
plot1 <- ggplot(data = test_data_2, aes(x = Age, y = predictions_2)) +
  geom_point(color = 'black') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = 'dashed') +
  ggtitle('Actual vs Predicted Age') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()
plot1

##############################################################################
## KNN 
##############################################################################

#Training the model

ctrl <- trainControl(method = "cv",
                     number=10)
knnFit_2 <- train(Age ~., data = train_data_2, method = "knn", trControl = ctrl, tuneLength = 20)

best_k_2 <- knnFit_2$bestTune$k
best_rmse_knn_2 <- min(knnFit_2$results$RMSE)
results_2 <- knnFit_2$results
k_values_2 <- seq(5, max(results_2$k),by=4)
results_2$SE <- results_2$RMSESD / sqrt(fit_control$number)

print(paste("Best k:", best_k_2))
print(paste("Best RMSE:", best_rmse_knn_2))

predictions_knn_2 <- predict(knnFit_2, newdata = test_data_2)
RMSE_knn_2 <- sqrt(mean((predictions_knn_2 - test_data_2$Age)^2)) 

#Graph 1
plot(knnFit_2)

#Graph 2
ggplot(results_2, aes(x = k, y = RMSE)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = best_k_2, linetype = "dashed", color = "red") +
  labs(title = "Number of Neighbors vs RMSE",
       x = "K",
       y = "Out-of-sample RMSE") +
  theme_minimal() 

#Graph 3
ggplot(results_2, aes(x = k, y = RMSE)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = RMSE - SE, ymax = RMSE + SE), width = 0.2) +
  geom_vline(xintercept = best_k_2, linetype = "dashed", color = "red") +
  labs(title = "RMSE and Standard Error at Each k",
       x = "Number of Neighbors (k)",
       y = "Root Mean Squared Error (RMSE)") +
  theme_minimal() 

#######################################################################
# End of Tom's code
#######################################################################

# 6. Run a single tree regression on training data, using all variables. Provide a summary and plot output
treeModel_2 <- rpart(Age ~ ., 
                   data = train_data_2,
                   method = "anova",
                   xval = 10, # number of cross-validations
                   control = rpart.control(cp=0.001, minsplit = 5))
summary(treeModel_2)
rpart.plot(treeModel_2)
plotcp(treeModel_2)

#Finding the optimal cp using the one sd rule
best_cp_ix_2 = which.min(treeModel_2$cptable[,4]) # "Best"
treeModel_2$cptable[best_cp_ix,4]

tol_2 = treeModel_2$cptable[best_cp_ix,4] + treeModel_2$cptable[best_cp_ix,5]
treeModel_2$cptable[treeModel_2$cptable[,4]<tol_2,][1,]
best_cp_onesd_2 = treeModel_2$cptable[treeModel_2$cptable[,4]<tol_2,][1,1]

pruned_tree_2 = prune(treeModel_2, cp=best_cp_onesd_2)

rpart.plot(pruned_tree_2)
summary(pruned_tree_2)
pruned_tree_2$cptable

# 7. Calculate RMSE of each
# Predict MSE of Single Tree
predictions_2_tree <- predict(treeModel_2, newdata = test_data_2)
RMSE_tree_2 <- sqrt(mean((test_data_2$Age - predictions_2_tree)^2))

# Predicting MSE of Pruned Tree
predictions_pruned_2 <- predict(pruned_tree_2, newdata = test_data_2)
RMSE_pruned_2 <- sqrt(mean((predictions_pruned_2 - test_data_2$Age)^2))
print(RMSE_pruned_2)

# comparing full tree and pruned tree. Need to create a better plot
plot(predict(pruned_tree), predict(treeModel))
abline(0,1)

# 10. Write a random forest regression on training data, where mtry = 3. Calculate RMSE
# Creating a custom grid
rf_grid_2 = data.frame(mtry = c(2,3,4,5,6,7))
rf_model_2 <- train( Age ~ ., data = train_data_2, 
                   method = "rf", 
                   trControl = fit_control_2,
                   tuneGrid = rf_grid_2,
                   ntree = 50 # best RMSE at 50 : 2.23 with 4 var, RMSE at 500: 2.17 with 4 var, RMSE at 1000: 2.17 with 4 var
)

# Getting a plot of CV error estimates
ggplot(rf_model_2)

# Adding +/- one se
best_2 = rf_model_2$results[which.min(rf_model_2$results$RMSE),]
onesd_2 = best_2$RMSE + best_2$RMSESD/sqrt(kcv)

ggplot(rf_model_2) + 
  geom_segment(aes(x=mtry, 
                   xend=mtry, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv)), 
               data=rf_model_2$results) + 
  geom_hline(yintercept = onesd_2, linetype='dotted')

rf_model_2$finalModel

# 11. Provide a variable importance plot
varImpPlot(rf_model_2$finalModel)

# Calculate RMSE with OOS data
predictions_rf_2  = predict(rf_model_2, newdata=test_data_2)
RMSE_rf_2 <- sqrt(mean( (test_data_2$Age - predictions_rf_2)^2 ))
print(RMSE_rf_2)

# Plot of Predicted vs Actual
plot(predictions_rf_2, test_data_2$Age, xlab = "Predicted Age", ylab = "Actual Age", main = "Predicted vs Actual Age (with Shell Weight Ratio)")
abline(0, 1, col = "orange")

############################################################
# Boosting Model
############################################################

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid_2 <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                           n.trees = c(1000, 1500, 2000, 2500, 3000, 3500), 
                           shrinkage = c(.05, .1),
                           n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_3 <- train(Age ~ ., data = train_data_2, 
                  method = "gbm", 
                  trControl = fit_control_2,
                  tuneGrid = gbm_grid_2,
                  verbose = FALSE)

print(gbmfit_3$bestTune)

best_ix = which.min(gbmfit_3$results$RMSE)
best = gbmfit_3$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit_3$results$RMSE<onese_max_RMSE

print(gbmfit_3$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)

# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit_3$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")

## Validation
gbm_yhat_2 = predict(gbmfit_3, newdata=test_data_2)
# So is validation RMSE
RMSE_boost_2 <- sqrt(mean( (test_data_2$Age - gbm_yhat_2)^2 ))
print(RMSE_boost_2)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_3)
ggplot(gbm_imp)

##########################################
# Default Model, which also performed well
##########################################

##Fit Model
gbmfit <- train( Age ~ ., data = train_data_2, 
                 method = "gbm", 
                 trControl = fit_control_2,
                 verbose = FALSE)

best_ix = which.min(gbmfit$results$RMSE)
best = gbmfit$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit$results$RMSE<onese_max_RMSE

print(gbmfit$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)



# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")

print(gbmfit$bestTune)

ggplot(gbmfit)


## Validation
gbm_yhat_2 = predict(gbmfit, newdata=train_data_2)
# So is validation RMSE
RMSE_boost_default_2 <- sqrt(mean( (train_data_2$Age - gbm_yhat_2)^2 ))
print(RMSE_boost_default_2)
# Comparing variable importance
gbm_imp <- varImp(gbmfit)
ggplot(gbm_imp)

#####################################################################
# Ridge Regression Optimal Model
#####################################################################

X <- model.matrix(Age ~ . - 1, data = crab_data_2) # Create a design matrix without intercept
y <- crab_data_2$Age

# Split the data into training and testing sets
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardize the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)

# Define the lambda sequence for cross-validation
lambda_seq <- seq(0.001, 0.3, by = 0.001)

# Perform ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq, nfolds = 10)

# Best lambda value
#best_lambda <- ridge_cv$lambda.min

#THIS IS THE ACTUAL BEST ONE NEED TO REPLECATE THE CODE WITH THE FOLLOWING
# It does the less complex one within one standard error

#best_lambda <- ridge_cv$lambda.1se
best_lambda <- ridge_cv$lambda.1se

# Train the final model with the best lambda
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


# Coefficients for the model
coef(ridge_model)

# Predict and evaluate
y_train_pred <- predict(ridge_model, X_train)
y_test_pred <- predict(ridge_model, X_test)

# Calculate R^2
train_r2 <- cor(y_train, y_train_pred)^2
test_r2 <- cor(y_test, y_test_pred)^2

# Calculate RMSE
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
RMSE_ridge_2 <- sqrt(mean((y_test - y_test_pred)^2))

# Print R^2 and RMSE scores
print(paste("Training R^2:", train_r2))
print(paste("Testing R^2:", test_r2))
print(paste("Training RMSE:", train_rmse))
print(paste("Testing RMSE:", RMSE_ridge))



# Visualize the model's prediction performance
train_data <- data.frame(Actual = y_train, Predicted = y_train_pred)
test_data <- data.frame(Actual = y_test, Predicted = y_test_pred)


# Plot for training set
ggplot(train_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Training Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot for testing set
ggplot(test_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'green', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Testing Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot to show the chosen lambda value
plot(ridge_cv)
abline(v = log(best_lambda), col = "red", lwd = 2)

plot(ridge_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), col = "red", lwd = 2)
legend("topright", legend = c("Best Lambda"), col = c("red"), lty = c(1), lwd = c(2))

###########################################################
# Consolidating all results into one table
###########################################################
rmse_table_SWR <- data.frame(
  Model = c("Bagging", "Lasso", "Linear Regression", "KNN", "Single Tree", "Pruned Tree", "Random Forest", "Ridge"), #Boosting
  RMSE = c(RMSE_bag_2, RMSE_lasso_2, RMSE_lm_2, RMSE_knn_2, RMSE_tree_2, RMSE_pruned_2, RMSE_rf_2, RMSE_ridge_2)#,RMSE_boost,RMSE_boost_default)
)

print(rmse_table_standard)
print(rmse_table_SWR)














#############################################################
# Appendix Code, Finding Optimal Boosting/Ridge Models
# FYI, This will take a very long time to run
############################################################


#Set Up



CrabAgePrediction <- read_csv("C:/Users/mbmma/OneDrive/Desktop/Intro to Machine Learning/Group Project/CrabAgePrediction.csv")

set.seed(1234)

summary(CrabAgePrediction)


#CrabAgePrediction_2 <- mutate(CrabAgePrediction, Shell_Weight_Ratio = CrabAgePrediction$`Shell Weight`/Weight)





#Training Test Split


# Hold out 20% of the data
train_ix = createDataPartition(CrabAgePrediction$Age,
                               p = 0.8)

crab_train = CrabAgePrediction[train_ix$Resample1,]
crab_test  = CrabAgePrediction[-train_ix$Resample1,]
```


#Boosting (GBM)

##Default Search - Lowest RMSE


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
gbmfit <- train( Age ~ ., data = crab_train, 
                 method = "gbm", 
                 trControl = fit_control,
                 verbose = FALSE)


print(gbmfit$bestTune)

ggplot(gbmfit)


## Validation
gbm_yhat = predict(gbmfit, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit)
ggplot(gbm_imp)



##Default Search - Lowest RMSE - 1SD


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

##Fit Model
gbmfit <- train( Age ~ ., data = crab_train, 
                 method = "gbm", 
                 trControl = fit_control,
                 verbose = FALSE)


print(gbmfit$bestTune)

best_ix = which.min(gbmfit$results$RMSE)
best = gbmfit$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit$results$RMSE<onese_max_RMSE

print(gbmfit$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)



# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")


## Validation
gbm_yhat = predict(gbmfit, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit)
ggplot(gbm_imp)



#########################################################################################################

##Grid Search - Lowest RMSE

```{r}
set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = seq(50, 500, by = 50), 
                         shrinkage = c( .05, .06, .07, .08, .09, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)


##Grid Search - Lowest RMSE (2)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = seq(50, 500, by = 50), 
                         shrinkage = c(.08, .09, .1, .11, .12),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)


##Grid Search - Lowest RMSE (3)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = seq(200, 700, by = 50), 
                         shrinkage = c(.08, .09, .1, .11, .12),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```



##Grid Search - Lowest RMSE (4)

set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:5), 
                         n.trees = seq(200, 700, by = 50), 
                         shrinkage = c(.08, .09, .1, .11, .12),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (5)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = seq(400, 1000, by = 50), 
                         shrinkage = c(.1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (6)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:5), 
                         n.trees = seq(400, 1000, by = 50), 
                         shrinkage = c(.1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (7)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:5), 
                         n.trees = seq(800, 1200, by = 50), 
                         shrinkage = c(.1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```



##Grid Search - Lowest RMSE (8)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:5), 
                         n.trees = seq(100, 200, by = 10), 
                         shrinkage = c(.1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)





######START HERE WITH PROF MURRAY##################
##Grid Search - Lowest RMSE (9)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = c(50,100,150), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (10)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = c(50,100,200, 500, 1000), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)

##Grid Search - Lowest RMSE (11)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = c(50,100,200, 500, 1000, 1500), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (12)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:8), 
                         n.trees = c(50,100, 500, 1000, 1500), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)




##Grid Search - Lowest RMSE (13)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1:15), 
                         n.trees = c(50,100, 500, 1000, 1500), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (14)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1,5, 10, 15, 20), 
                         n.trees = c(100, 500, 1000, 1500), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```



##Grid Search - Lowest RMSE (15)


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1,5, 10, 15, 20, 25, 30), 
                         n.trees = c(100, 500, 1000, 1500), 
                         shrinkage = c(.01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (16)

set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1,10, 15, 20, 25, 30, 35), 
                         n.trees = c(500, 1000, 1500), 
                         shrinkage = c( .1, .2, .3),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```




##Grid Search - Lowest RMSE (17)

set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 15, 25, 35), 
                         n.trees = c(1000, 1500, 2000), 
                         shrinkage = c( .01, .1, .2),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)


##Grid Search - Lowest RMSE (18)

set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35), 
                         n.trees = c(1000, 1500, 2000, 2500), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (19) [Do the same as above but add in another larger size of tree and a couple more depth places]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35), 
                         n.trees = c(1000, 1500, 2000, 2500), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (19b) [Do the same as above but add in another larger size of tree and a couple more depth places]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                         n.trees = c(1000, 1500, 2000, 2500), 
                         shrinkage = c(.05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (19c) [Do the same as above but add in another larger size of tree and a couple more depth places]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                         n.trees = c(1000, 1500, 2000, 2500, 3000, 3500), 
                         shrinkage = c(.05),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)



##Grid Search - Lowest RMSE (20) [Make one with only a few depth (1,3,5) but have a lot of number of trees we will test]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 3,5), 
                         n.trees = c(500, 1000, 1500, 2000, 2500, 3000, 3500, 4500), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```



##Grid Search - Lowest RMSE (20b) [Make one with only a few depth (1,3,5) but have a lot of number of trees we will test]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 3,5), 
                         n.trees = c(500, 1000,2500, 3500, 4500, 5000, 5500, 6000), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (20c) [Make one with only a few depth (1,3,5) but have a lot of number of trees we will test]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 3,5), 
                         n.trees = c(500, 1000,2500, 3500, 4500, 5000, 5500, 6000, 7000, 8000), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (20d) [Make one with only a few depth (1,3,5) but have a lot of number of trees we will test]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 3,5), 
                         n.trees = c(500, 1000,2500, 3500, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)
```


##Grid Search - Lowest RMSE (20e) [Make one with only a few depth (1,3,5) but have a lot of number of trees we will test]


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds)

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid <-  expand.grid(interaction.depth = c(1, 3,5), 
                         n.trees = c(500, 1000,2500, 3500, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000), 
                         shrinkage = c( .01, .05, .1),
                         n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_2 <- train(Age ~ ., data = crab_train, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid,
                  verbose = FALSE)

print(gbmfit_2$bestTune)

ggplot(gbmfit_2)


## Validation
gbm_yhat = predict(gbmfit_2, newdata=crab_test)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_2)
ggplot(gbm_imp)

################################################################
# Ridge Regression
################################################################


# Load the data
data <- read_csv("C:/Users/mbmma/OneDrive/Desktop/Intro to Machine Learning/Group Project/CrabAgePrediction.csv")


# Define features and target
X <- model.matrix(Age ~ . - 1, data = data) # Create a design matrix without intercept
y <- data$Age

# Split the data into training and testing sets
set.seed(1234)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardize the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)

# Define the lambda sequence for cross-validation
lambda_seq <- 10^seq(2, -2, by = -0.1)

# Perform ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq, nfolds = 10)

# Best lambda value
#best_lambda <- ridge_cv$lambda.min


#THIS IS THE ACTUAL BEST ONE NEED TO REPLECATE THE CODE WITH THE FOLLOWING
# It does the less complex one within one standard error

#best_lambda <- ridge_cv$lambda.1se
best_lambda <- ridge_cv$lambda.1se


# Train the final model with the best lambda
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


# Coefficients for the model
coef(ridge_model)

# Predict and evaluate
y_train_pred <- predict(ridge_model, X_train)
y_test_pred <- predict(ridge_model, X_test)

# Calculate R^2
train_r2 <- cor(y_train, y_train_pred)^2
test_r2 <- cor(y_test, y_test_pred)^2

# Calculate RMSE
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
test_rmse <- sqrt(mean((y_test - y_test_pred)^2))

# Print R^2 and RMSE scores
print(paste("Training R^2:", train_r2))
print(paste("Testing R^2:", test_r2))
print(paste("Training RMSE:", train_rmse))
print(paste("Testing RMSE:", test_rmse))



# Visualize the model's prediction performance
train_data <- data.frame(Actual = y_train, Predicted = y_train_pred)
test_data <- data.frame(Actual = y_test, Predicted = y_test_pred)


# Plot for training set
ggplot(train_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Training Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot for testing set
ggplot(test_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'green', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Testing Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()


#This might make a plot to help select best lambda value for the model [IF IT IS ADD IT TO THE REST OF THE MODEL]
plot(ridge_cv)

# Plot to show the chosen lambda value
plot(ridge_cv)
abline(v = log(best_lambda), col = "red", lwd = 2)

plot(ridge_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), col = "red", lwd = 2)
legend("topright", legend = c("Best Lambda"), col = c("red"), lty = c(1), lwd = c(2))

#RUN DIAGNOSTICS ON THE MODEL TO SEE IF IT FITS WELL LIKE ON PAGES
#Test a few different lambda sequences and the default
#plot the choose lambda by RMSE plot like in the lectures and the example code


#Ridge Regression [Lambda defualt]



# Load the data
data <- read_csv("C:/Users/mbmma/OneDrive/Desktop/Intro to Machine Learning/Group Project/CrabAgePrediction.csv")


# Define features and target
X <- model.matrix(Age ~ . - 1, data = data) # Create a design matrix without intercept
y <- data$Age

# Split the data into training and testing sets
set.seed(1234)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardize the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)


# Perform ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10)

# Best lambda value
#best_lambda <- ridge_cv$lambda.min

#THIS IS THE ACTUAL BEST ONE NEED TO REPLECATE THE CODE WITH THE FOLLOWING
# It does the less complex one within one standard error

#best_lambda <- ridge_cv$lambda.1se
best_lambda <- ridge_cv$lambda.1se


# Train the final model with the best lambda
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


# Coefficients for the model
coef(ridge_model)


# Predict and evaluate
y_train_pred <- predict(ridge_model, X_train)
y_test_pred <- predict(ridge_model, X_test)

# Calculate R^2
train_r2 <- cor(y_train, y_train_pred)^2
test_r2 <- cor(y_test, y_test_pred)^2

# Calculate RMSE
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
test_rmse <- sqrt(mean((y_test - y_test_pred)^2))

# Print R^2 and RMSE scores
print(paste("Training R^2:", train_r2))
print(paste("Testing R^2:", test_r2))
print(paste("Training RMSE:", train_rmse))
print(paste("Testing RMSE:", test_rmse))



# Visualize the model's prediction performance
train_data <- data.frame(Actual = y_train, Predicted = y_train_pred)
test_data <- data.frame(Actual = y_test, Predicted = y_test_pred)


# Plot for training set
ggplot(train_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Training Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot for testing set
ggplot(test_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'green', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Testing Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()



# Plot to show the chosen lambda value
plot(ridge_cv)
abline(v = log(best_lambda), col = "red", lwd = 2)

plot(ridge_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), col = "red", lwd = 2)
legend("topright", legend = c("Best Lambda"), col = c("red"), lty = c(1), lwd = c(2))

#RUN DIAGNOSTICS ON THE MODEL TO SEE IF IT FITS WELL LIKE ON PAGES
#Test a few different lambda sequences and the default
#plot the choose lambda by RMSE plot like in the lectures and the example code




##Ridge Regression [Lambda tune 2]


# Load the data
data <- read_csv("C:/Users/mbmma/OneDrive/Desktop/Intro to Machine Learning/Group Project/CrabAgePrediction.csv")


# Define features and target
X <- model.matrix(Age ~ . - 1, data = data) # Create a design matrix without intercept
y <- data$Age

# Split the data into training and testing sets
set.seed(1234)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardize the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)

# Define the lambda sequence for cross-validation
lambda_seq <- seq(0.001, 0.3, by = 0.001)

# Perform ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq, nfolds = 10)

# Best lambda value
#best_lambda <- ridge_cv$lambda.min

#THIS IS THE ACTUAL BEST ONE NEED TO REPLECATE THE CODE WITH THE FOLLOWING
# It does the less complex one within one standard error

#best_lambda <- ridge_cv$lambda.1se
best_lambda <- ridge_cv$lambda.1se

# Train the final model with the best lambda
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)


# Coefficients for the model
coef(ridge_model)


# Predict and evaluate
y_train_pred <- predict(ridge_model, X_train)
y_test_pred <- predict(ridge_model, X_test)

# Calculate R^2
train_r2 <- cor(y_train, y_train_pred)^2
test_r2 <- cor(y_test, y_test_pred)^2

# Calculate RMSE
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
test_rmse <- sqrt(mean((y_test - y_test_pred)^2))

# Print R^2 and RMSE scores
print(paste("Training R^2:", train_r2))
print(paste("Testing R^2:", test_r2))
print(paste("Training RMSE:", train_rmse))
print(paste("Testing RMSE:", test_rmse))



# Visualize the model's prediction performance
train_data <- data.frame(Actual = y_train, Predicted = y_train_pred)
test_data <- data.frame(Actual = y_test, Predicted = y_test_pred)


# Plot for training set
ggplot(train_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Training Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()

# Plot for testing set
ggplot(test_data, aes(x = Actual, y = s0)) +
  geom_point(color = 'green', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle('Testing Set: Actual vs Predicted') +
  xlab('Actual Age') +
  ylab('Predicted Age') +
  theme_minimal()


# Plot to show the chosen lambda value
plot(ridge_cv)
abline(v = log(best_lambda), col = "red", lwd = 2)

plot(ridge_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), col = "red", lwd = 2)
legend("topright", legend = c("Best Lambda"), col = c("red"), lty = c(1), lwd = c(2))


#RUN DIAGNOSTICS ON THE MODEL TO SEE IF IT FITS WELL LIKE ON PAGES
#Test a few different lambda sequences and the default
#plot the choose lambda by RMSE plot like in the lectures and the example code

#EDA

#install.packages("corrplot")
library(corrplot)
data <- read.csv("CrabAgePrediction.csv")

# Display the first few rows of the dataset
head(data)

# Summary statistics
summary(data)

# Check for missing values
missing_values <- colSums(is.na(data))
missing_values

# Visualize distributions of numerical variables
num_cols <- sapply(data, is.numeric)
data_num <- data[, num_cols]

# Plot histograms for each numerical variable
par(mfrow=c(3,3))  # Adjust the layout to fit all plots
for (col in colnames(data_num)) {
  hist(data_num[[col]], main=col, xlab=col, col="skyblue", border="white")
}

# Visualize distributions of categorical variables if any
cat_cols <- sapply(data, is.factor)
if (sum(cat_cols) > 0) {
  data_cat <- data[, cat_cols]
  for (col in colnames(data_cat)) {
    barplot(table(data_cat[[col]]), main=col, xlab=col, col="lightgreen", border="white")
  }
}

# Correlation matrix for numerical variables
cor_matrix <- cor(data_num)
corrplot::corrplot(cor_matrix, method="circle")

# Pair plot to visualize relationships between variables
pairs(data_num, pch = 19, col = rgb(0,0,1,0.5))




# Now do again but with a new enginnered variable
## Set up and training and test split
```{r}
CrabAgePrediction <- read_csv("C:/Users/mbmma/OneDrive/Desktop/Intro to Machine Learning/Group Project/CrabAgePrediction.csv")

set.seed(1234)
CrabAgePrediction_2 <- mutate(CrabAgePrediction, Shell_Weight_Ratio = CrabAgePrediction$`Shell Weight`/Weight)


# Hold out 20% of the data
train_ix2 = createDataPartition(CrabAgePrediction_2$Age,
                                p = 0.8)

crab_train2 = CrabAgePrediction_2[train_ix2$Resample1,]
crab_test2  = CrabAgePrediction_2[-train_ix2$Resample1,]

#Boosting (GBM)

##Default Search - Lowest RMSE


set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train2$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

##Fit Model
gbmfit <- train( Age ~ ., data = crab_train2, 
                 method = "gbm", 
                 trControl = fit_control,
                 verbose = FALSE)

best_ix = which.min(gbmfit$results$RMSE)
best = gbmfit$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit$results$RMSE<onese_max_RMSE

print(gbmfit$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)



# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")

print(gbmfit$bestTune)

ggplot(gbmfit)


## Validation
gbm_yhat = predict(gbmfit, newdata=crab_test2)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test2$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit)
ggplot(gbm_imp)
```
#Boosting Best Model 

```{r}
set.seed(1234)
##CV
# Number of folds
kcv = 10

cv_folds = createFolds(crab_train2$Age,
                       k = kcv)

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  selectionFunction="oneSE")

##Fit Model
#When increasing shrinkage it does not fit as well if you go over .1
gbm_grid_2 <-  expand.grid(interaction.depth = c(1, 5, 10, 15, 20, 25, 30, 35, 40), 
                           n.trees = c(1000, 1500, 2000, 2500, 3000, 3500), 
                           shrinkage = c(.05, .1),
                           n.minobsinnode = 10)

#The reason we only tune the first three is that is what the book says are the three parameters on page 347
#interaction depth for boosting tends fitting 1 since it works well often to 8 which is the number of variables

gbmfit_3 <- train(Age ~ ., data = crab_train2, 
                  method = "gbm", 
                  trControl = fit_control,
                  tuneGrid = gbm_grid_2,
                  verbose = FALSE)

print(gbmfit_3$bestTune)

best_ix = which.min(gbmfit_3$results$RMSE)
best = gbmfit_3$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv)

# These are the parameter values within one SD:
onese_ixs = gbmfit_3$results$RMSE<onese_max_RMSE

print(gbmfit_3$results[onese_ixs,])

# tidyverse subsetting:
# gbmfit_2$results %>% filter(RMSE<onese_max_RMSE)



# Or we can build our own to choose facets/colors/etc, and add
# +/- 1 SE

gbm_plot_df = gbmfit_3$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)

ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom")


## Validation
gbm_yhat = predict(gbmfit_3, newdata=crab_test2)
# So is validation RMSE
RMSE <- sqrt(mean( (crab_test2$Age - gbm_yhat)^2 ))
print(RMSE)
# Comparing variable importance
gbm_imp <- varImp(gbmfit_3)
ggplot(gbm_imp)
```



#EDA
```{r}
#install.packages("corrplot")
library(corrplot)
data <- CrabAgePrediction_2

# Display the first few rows of the dataset
head(data)

# Summary statistics
summary(data)

# Check for missing values
missing_values <- colSums(is.na(data))
missing_values

# Visualize distributions of numerical variables
num_cols <- sapply(data, is.numeric)
data_num <- data[, num_cols]

# Plot histograms for each numerical variable
par(mfrow=c(3,3))  # Adjust the layout to fit all plots
for (col in colnames(data_num)) {
  hist(data_num[[col]], main=col, xlab=col, col="skyblue", border="white")
}

# Visualize distributions of categorical variables if any
cat_cols <- sapply(data, is.factor)
if (sum(cat_cols) > 0) {
  data_cat <- data[, cat_cols]
  for (col in colnames(data_cat)) {
    barplot(table(data_cat[[col]]), main=col, xlab=col, col="lightgreen", border="white")
  }
}

# Correlation matrix for numerical variables
cor_matrix <- cor(data_num)
corrplot::corrplot(cor_matrix, method="circle")

# Pair plot to visualize relationships between variables
pairs(data_num, pch = 19, col = rgb(0,0,1,0.5))


