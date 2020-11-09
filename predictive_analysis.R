# Predictive Analysis                               
# Based on SOEP data 2015                          
# Author: Roman Schulze                           
                                                                              
# Set working directory
setwd("/Users/romanschulze/Desktop/Roman/Uni/Master/4.Semester/MachineLearning/SOEP/SOEPv33")

# Load libraries 
library(caret)
library(forecast)
library(gbm)
library(ggplot2)
library(Metrics)
library(partykit)
library(randomForest)
library(rpart)
library(ROCR)
library(stargazer)
library(tikzDevice)

# Read Data
data <- read.csv("myData.csv")

#Remove first column, which is ID
data <- data[, -1]                     

# Summary data
summary(data)
str(data)

# Plot density of Wage
ggplot(data, aes(x = Salary)) + geom_density() + 
                     scale_x_continuous(name = "Salary", breaks = seq(0, 5, 0.5),
                     limits=c(0, 5)) + theme_bw() +
                     ggtitle("Density plot log hourly Wage 2015") +
                     theme(plot.title = element_text(hjust = 0.5))


# Split data in two partitions (70% Training, 30% Test)
set.seed(10)
inTrain <- createDataPartition(data$Salary, p = 0.7, list = FALSE)
train <- data[inTrain, ]
test <- data[-inTrain, ]
x <- train[, 2:13]
y <- train[, 1]

# -------------------------------------

# 1. Regression Tree

# -------------------------------------

# For reproducibility
set.seed(1)

# Grow a full regression tree with alpha = 0 ,
# xval corresponds to number of cv performed 
fit <- rpart(Salary ~ ., data = train, method = "anova", cp = 0, xval = 10)

# Plot tree
plot(fit)
#text(fit ,pretty = 0)


# Display cross-validation results 
printcp(fit)

# Visualize cross- validation results
plotcp(fit)


# Prune the tree
fit$cptable[which.min(fit$cptable[, "xerror"]), "CP"]
pfit <- prune(fit, cp = fit$cptable[which.min(fit$cptable[, "xerror"]), "CP"])    


# Plot the pruned regression tree
plot(pfit)
text(pfit, pretty = 0, cex = 1.1)

# MSE
pred_tree <- predict(pfit, test)
pred_tree_tr <- predict(pfit, train)

mse(pred_tree, test$Salary)
mse(pred_tree_tr, train$Salary)

# Plot predicted vs actual values
par(mfrow = c(1, 1))
plot(pred_tree_tr,train$Salary, xlab = "Predicted Values", ylab = "Actual Values",
     main = "Training Data")
plot(pred_tree,test$Salary, xlab = "Predicted Values", ylab = "Actual Values", 
     main = "Test Data")
grid()
abline(0, 1, col = "darkorange", lwd = 2)


# Prune tree by default alpha = 0.01
pfit1 <- rpart(Salary ~ ., data = train, method = "anova", cp = 0.01)


# Plot the pruned regression tree
plot(pfit1)
text(pfit1,pretty = 0, cex = 0.7)

# plot with boxplot, for 2nd presentation
plot(as.party(pfit1))


# -------------------------------------

# 2. Bagging

# -------------------------------------


# For reproducibility
set.seed(1)
bag <- randomForest(Salary ~ ., data = train, mtry = 12, ntree = 300, 
                    importance = TRUE)
bag
# -argument mtry=12 indicates that all 12 predictors should be considered 
# for each split of the tree

# Plot the Out-of-Bag error rate vs the number of trees
par(mfrow = c(1, 1))
plot(bag, main = "Bagging: Error Rate vs Number of Trees", cex.main = 1.5, 
     cex.lab = 1.4, cex.axis = 1.4)


# Compare the performance of the bagged decision tree on the traning
# and test data
pred_t_Bag <- predict(bag, newdata = train)
pred_v_Bag <- predict(bag, newdata = test)

# RMSE and other Measures
accuracy(pred_t_Bag, train$Salary)
accuracy(pred_v_Bag, test$Salary)


# PLot predicted vs actual Values
par(mfrow = c(2, 1))
plot(pred_t_Bag,train$Salary, xlab = "Predicted Values", ylab = "Actual Values",
     main = "Training Data")
plot(pred_v_Bag,test$Salary, xlab = "Predicted Values", ylab = "Actual Values", 
     main = "Test Data")


# Tuning Bagging using Grid Search

# Set Seed
set.seed(1)   

# Defining Hyperparameters
hyper_grid_bag <- expand.grid(
  nodesize = seq(12, 30, 2),
  ntree = seq(200, 600, 100)
  )

# Checking hyper_grid_bag
head(hyper_grid_bag)

# total number of combinations
nrow(hyper_grid_bag)

# OutOfBag Error
oob_err_bag <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid_bag)) {
  
  # get minsplit, maxdepth values at row i
  nodesize <- hyper_grid_bag$nodesize[i]
  ntree <- hyper_grid_bag$ntree[i]
  
  # train a model and store in the list
  model <- randomForest(formula = Salary ~ ., data = train, mtry = 12,
                        nodesize = nodesize, ntree = ntree)
  
  oob_err_bag[i] <- model$mse[max(ntree)]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err_bag)
print(hyper_grid_bag[opt_i, ])
bestParams_bag <- hyper_grid_bag[opt_i, ]

# Use bestParams for Bagging 
bag_grid <- randomForest(Salary ~ ., data = train, mtry = 12, 
                         nodesize = bestParams_bag[, 1],
                         ntree = bestParams_bag[, 2], 
                         importance = TRUE)

# Compare the performance of the bagged decision tree on the traning 
# and test data
pred_t_bag_grid <- predict(bag_grid, newdata = train)
pred_v_bag_grid <- predict(bag_grid, newdata = test)


# MSE
mse(pred_t_bag_grid, train$Salary) 
mse(pred_v_bag_grid, test$Salary) 


# Plot the Out-of-Bag error rate vs the number of trees
par(mfrow = c(1, 1))
plot(bag_grid, main = "Bagging: OOB-Error Rate vs Number of Trees",
     cex.main = 0.9)
     #cex.main=1.5, cex.lab=1.4, cex.axis=1.4
dev.off()

# Plot importance of variables
test_b <- sort(importance(bag)[, 1]) / max(importance(bag)[, 1])
test_b <- data.frame(x1 = labels(test_b), y1 = test_b)
test_b <- transform(test_b, x1 = reorder(x1, y1))

tikz(file = "plot_test.tex", width = 6.5, height = 4)

ggplot(data=test_b, aes(x = x1, y = y1)) + 
  ggtitle("Feature Importance") +
  theme(plot.title = element_text(size = 17, face = "bold")) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  ylab("Mean Decrease RSS") + xlab("") +
  geom_bar(stat = "identity",fill = "black",alpha = .8,width = .75) + 
  coord_flip() + theme(axis.text = element_text(size = 14),
  axis.title=element_text(size = 14, face = "bold"))


# PLot predicted vs actual Values
par(mfrow = c(2, 1))
plot(pred_t_bag_grid,train$Salary, xlab = "Predicted Values",
     ylab = "Actual Values",
     main = "Predicted vs Actual: Bagging, Training Data",  col = "black",
     pch = 10, cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)
abline(0, 1, col = "red")

plot(pred_v_bag_grid,test$Salary, xlab = "Predicted Values",
     ylab = "Actual Values",  main = "Bagging: Random Forest, Test Data", 
     col = "black", pch = 10, cex.main = 0.9) # cex.lab = 1.4, cex.axis = 1.4)
abline(0, 1, col = "red")



# -------------------------------------------

# 3. Random Forest

# -------------------------------------------

# For reproducibility
set.seed(1)

# Random forest using default settings
rf <- randomForest(Salary ~ ., data = train, mtry = 12/3, ntree = 300, 
                   importance = TRUE)

# Compare the performance of the traning and test data
pred_t_ranFor <- predict(rf, newdata = train)
pred_v_ranFor <- predict(rf, newdata = test)

# RMSE and other Measures
accuracy(pred_t_ranFor, train$Salary)
accuracy(pred_v_ranFor, test$Salary)

# PLot predicted vs actual Values
par(mfrow=c(2, 1))
plot(pred_t_ranFor,train$Salary, xlab = "Predicted Values", 
     ylab = "Actual Values",
     main = "Predicted vs Actual: Random Forest, Training Data")
abline(0, 1, col = "orange")
plot(pred_v_ranFor, test$Salary, xlab = "Predicted Values",
     ylab = "Actual Values", 
     main = "Predicted vs Actual: Random Forest, Test Data")
abline(0, 1, col = "orange")


# Tuning the Forest using Grid Search

# Set Seed
set.seed(1)   

# Defining Hyperparameters
hyper_grid_rf <- expand.grid(
nodesize = seq(10, 20, 2),
mtry = seq(2, 6, 1), 
ntree = seq(200, 600, 100)
)

# Checking hyper_grid_rf
head(hyper_grid_rf)

# total number of combinations
nrow(hyper_grid_rf)

# OutOfBag Error
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid_rf)) {
  
  # get minsplit, maxdepth values at row i
  nodesize <- hyper_grid_rf$nodesize[i]
  mtry <- hyper_grid_rf$mtry[i]
  ntree <- hyper_grid_rf$ntree[i]
  
  # train a model and store in the list
  model <- randomForest(formula = Salary ~ ., data  = train, 
            nodesize = nodesize, mtry = mtry, ntree = ntree )
  
         oob_err[i] <- model$mse[max(ntree)]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid_rf[opt_i, ])
bestParams_rf <- hyper_grid_rf[opt_i, ]

# Use bestParams for Forest 
rf_grid <- randomForest(Salary ~ ., data = train, nodesize = bestParams_rf[, 1],
                        mtry = bestParams_rf[, 2], ntree = bestParams_rf[, 3],
                        importance = TRUE)


# Plot out-og-bag error against number of trees
par(mfrow = c(1, 1))
plot(rf_grid, main = "Random Forest: OOB-Error Rate vs Number of Trees", 
     cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)



### Plot Bagging and Forest in One

# Save mse in dataframe
Bagging <- bag_grid$mse
RandomForest <- rf_grid$mse
tree <- seq(1, 500, 1)
d <- data.frame(tree,Bagging,RandomForest)


# Plot results
g <- ggplot(d, aes(tree)) + 
  geom_line(aes(y = Bagging, colour = "Bagging")) + 
  geom_line(aes(y = RandomForest, colour = "RandomForest")) + 
  ggtitle("Out-of-bag Error")  + 
   ylab("MSE") + xlab("Number of Trees") +
  theme(plot.title = element_text(size = 10, face = "bold"),
        legend.title = element_text(size = 8), 
        legend.text = element_text(size = 7)) +
        theme(plot.title = element_text(hjust = 0.5)) + 
        theme(axis.text = element_text(size = 9),
        axis.title = element_text(size = 9, face = "bold")) 
      # theme(legend.position = c(0.85, 0.85)) 
g$labels$colour <- "Method"


# Plot Feature Importance
test_rf <- sort(importance(rf_grid)[, 1]) / max(importance(rf_grid)[, 1])
test_rf <- data.frame(x1 = labels(test_rf),y1 = test_rf)
test_rf <- transform(test_rf, x1 = reorder(x1, y1))

ggplot(data=test_rf, aes(x = x1, y = y1)) + ggtitle("Feature Importance") +
       theme(plot.title = element_text(size = 10, face = "bold")) +
       theme(plot.title = element_text(hjust = 0.5)) + 
       ylab("Mean Decrease RSS") + xlab(" ") + 
       geom_bar(stat = "identity", fill = "black", alpha = .8, width = .75) + 
       coord_flip()  + 
       theme(axis.text = element_text(size = 9),
       axis.title = element_text(size = 9, face = "bold"))


# Compare the performance of the bagged decision tree on the traning and test
# data
pred_t_ranFor_grid <- predict(rf_grid, newdata = train)
pred_v_ranFor_grid <- predict(rf_grid, newdata = test)

# MSE
rf_mse_tr <- mse(pred_t_ranFor_grid, train$Salary) 
rf_mse_test <- mse(pred_v_ranFor_grid, test$Salary) 


# Plot predicted vs actual Values
par(mfrow = c(2, 1))
plot(pred_t_ranFor_grid,train$Salary, main = "Predicted vs Actual:Random Forest,
     Training Data", xlab = "Predicted Values", ylab = "Actual Values",
     col = "black", pch = 20, cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)
grid()
abline (0,1, col = "red",  lwd = 2)
plot(pred_v_ranFor_grid, test$Salary,main = "Predicted vs Actual:Random Forest,
     Test Data", xlab = "Predicted Values", ylab = "Actual Values", 
     col = "black", pch = 20, cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)
grid()
abline(0, 1, col = "red", lwd = 2)

# -------------------------------------------

# 4. Boosting    

# -------------------------------------------

# For reproducibility
set.seed(1)

# Gradient Boosting
boost <- gbm(Salary ~ ., data = train, distribution = "gaussian", 
             n.trees = 10000, shrinkage = 0.001, interaction.depth = 7,
             bag.fraction = 0.5)

# Summary
print(boost)

# Summary gives a table of Variable Importance and a plot of Variable Importance
summary(boost)

# Predict on training and test
predict_boost <- predict(boost, train, n.tree = 10000)
predict1_boost <- predict(boost, test, n.tree = 10000)

# Just RMSE for train and test
mse(predict_boost,train$Salary) 
mse(predict1_boost, test$Salary) 


# Tuning Hyperparameters using Grid search
  

# Defining Hyperparameters
hyper_grid_boost <- expand.grid(
  shrinkage = c(0.001, 0.01),
  interaction.depth = seq(1, 7, 2),
  n.trees = seq (1000, 10000, 1000))

# Checking hyper_grid_rf
head(hyper_grid_boost)

# total number of combinations
nrow(hyper_grid_boost)

# OutOfBag Error
oob_err_b <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid_boost)) {
  
  # get minsplit, maxdepth values at row i
  n.trees <- hyper_grid_boost$n.trees[i]
  interaction.depth <- hyper_grid_boost$interaction.depth[i]
  shrinkage <- hyper_grid_boost$shrinkage[i]
  print(i)
  
# train a model and store in the list
model_b <- gbm(formula = Salary ~ ., data = train, distribution = "gaussian", 
                 bag.fraction = 0.5,
                 n.trees = n.trees, interaction.depth = interaction.depth , 
                 shrinkage = shrinkage)
  
  oob_err_b[i] <- model_b$train.error[max(n.trees)]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i_b <- which.min(oob_err_b)
print(hyper_grid_boost[opt_i_b, ])
bestParams_b <- hyper_grid_boost[opt_i_b, ]
# Use bestParams for gbm 
gbm_grid <- gbm(formula = Salary ~ ., data = train, distribution = "gaussian", 
                shrinkage = bestParams_b[, 1] , 
                interaction.depth = bestParams_b[, 2],  
                n.trees = bestParams_b[, 3])

print(gbm_grid)

# Predict on training and test
predict_boost_grid <- predict(gbm_grid, train, n.tree = bestParams_b[, 3])
predict1_boost_grid <- predict(gbm_grid, test, n.tree = bestParams_b[, 3])


# MSE for training and test dataset
gmb_mse_tr <- mse(predict_boost_grid, train$Salary) 
gmb_mse_ <- mse(predict1_boost_grid, test$Salary) 


# Plot Prediction
par(mfrow = c(2, 1))
plot(predict_boost_grid, train$Salary, main = "Predicted vs Actual: GBM,
     Training Data", xlab = "Predicted Values", ylab = "Actual Values", 
     col = "black", pch = 20, cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)
grid()
abline (0,1, col = "red", lwd = 2)
plot(predict1_boost_grid, test$Salary,main = "Predicted vs Actual: GBM,
     Test Data", xlab = "Predicted Values", ylab = "Actual Values",
     col = "black",pch = 20, cex.main = 1.5, cex.lab = 1.4, cex.axis = 1.4)
grid()
abline(0, 1, col = "orange", lwd = 2)


# Plot Feature Importance
par(mfrow = c(1, 1))
tibble::as_tibble(summary(gbm_grid), validate = TRUE, 
                  abbreviate(use.classes = T))

# Error Tree plot
par(mfrow = c(2, 1))
gbm.perf(gbm_grid, method = "OOB", plot.it = TRUE, oobag = TRUE, overlay = TRUE)
bestTree

Prediction <- gbm.perf(gbm_grid)


# Plot Partial Dependencies
par(mfrow = c(1, 2))
#plot(gbm_grid, i = "Occupation", col = "navy", lwd = 2)
#grid()
plot(gbm_grid, i = "Tenure", col = "navy", lwd = 2)
grid()
plot(gbm_grid, i = "Male", col = "navy", lwd = 2)
grid()
title("Partial Dependence Plot", outer = TRUE, line = -2, cex = 4)


# -------------------------------------------

# 5. Linear Regression as benchmark

# -------------------------------------------

# Performing Linear Regression on test set
lm <- lm(Salary ~ ., data = test)

# Printing Results 
summary(lm)

# MSE 
lm_mse <- mse(lm$fitted.values, test$Salary)



-------------------------------------------

# 6. Outputs for presentation

-------------------------------------------


# Descriptive Statistic
stargazer(data, type = "latex", title = "Descriptive statistics",
          digits = 1, out = "table1.tex")

# Bagging Forest txt file
cat("Bagging Output", file = "bag.txt")
# add 2 newlines
cat("\n\n", file = "bag.txt", append = TRUE)
# export anova test output
cat("Bagging\n", file = "bag.txt", append = TRUE)
capture.output(bag_grid, file = "bag.txt", append = TRUE)

# Random Forest txt file
cat("Tests Output", file = "tests.txt")
# add 2 newlines
cat("\n\n", file = "tests.txt", append = TRUE)
# export anova test output
cat("Random Forest\n", file = "tests.txt", append = TRUE)
capture.output(rf_grid, file = "tests.txt", append = TRUE)

# Linear Regression Latex Output
stargazer(lm, type = "latex", title = "Linear Regression", digits = 1,
          out = "table2.tex", omit.stat = c("LL", "ser", "f"), ci = TRUE, 
          ci.level = 0.95, single.row = TRUE)

# -------------------------------------------

# 7. Evaluation of models using MSE

# -------------------------------------------


# MSE of all models
actual <- test$Salary
lm_mse <- mse(lm$fitted.values, actual)
tree_mse <- mse(pred_tree, actual) 
bag_mse <- mse(pred_v_bag_grid, actual) 
rf_mse <- mse(pred_v_ranFor_grid, actual) 
gbm_mse <- mse(predict1_boost_grid, actual) 

# Print results
sprintf("Regression Tree Test RMSE: %.3f", tree_mse)
sprintf("Bagged Trees Test RMSE: %.3f", bag_mse)
sprintf("Random Forest Test RMSE: %.3f", rf_mse)
sprintf("GBM Test RMSE: %.3f", gbm_mse)
sprintf("Regression Test RMSE: %.3f", lm_mse)


# Comparing MSE of all models using Bar chart
Method <- c("Regression Tree", "Bagging", "Random Forest", "GBM", "OLS")
MSE <- c(tree_mse, bag_mse, rf_mse, gbm_mse, lm_mse)
Results <- data.frame(Method, MSE)

#Round results
Results[] <- lapply(Results, function(x) if(is.numeric(x)) round(x, 3) else x)


# Reorder MSE of methods in a decreasing manner
Results$Method <- factor(Results$Method, 
                         levels = Results$Method[order(Results$MSE)])
Results$Method   # notice the changed order of factor levels

# Plot Results
par(mfrow = c(1, 1))
p <- ggplot(data = Results, aes(x = Method, y = MSE , fill = Method, MSE)) +
  geom_bar(stat = "identity",width = 0.4) + 
  ggtitle("Mean Squared Test set error") + 
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        legend.title = element_text(size = 9), 
        legend.text = element_text(size = 9),
        axis.text = element_text(size = 9)) +
  axis.title = element_text(size = 9,face = "bold")) + 
  scale_fill_brewer(palette = "Greys") + 
  geom_text(aes(label = MSE), position = position_dodge(width = 0.9), 
            vjust = -0.25, size = 3) 

p <- p + theme(legend.position = "none")   
  







