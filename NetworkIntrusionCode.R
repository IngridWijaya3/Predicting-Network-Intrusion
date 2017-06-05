# Clear the workspace and start fresh
cat("\014")
closeAllConnections()
rm(list=ls())

setwd("/Users/ingridwijaya/Desktop/Courses/Regression/")

####################################################################################
# Data prepration

# Dataset doesn't contain information about header and attribute names
# Load the training data set
kddTrain <- read.csv("kddcup.data_10_percent_corrected.csv",header=F, stringsAsFactors = FALSE)

# Get the list of attributes from the the following text file and assign to datasets
attributeNames <- read.table("kddcup.names", skip = 1, sep = ":")
names(kddTrain) <- attributeNames$V1
names(kddTrain)[dim(kddTrain)[2]] <- "label"
remove('attributeNames')
####################################################################################
# CLEAN UP the datasets

# Verify is the feature has an NA value
if(sum(is.na(kddTrain$label))){
  message("NA item found in training dataset","\r",appendLF=FALSE)
}

# caret library is needed for nearzerovar function and etc
library(caret)

# Remove the features that have close to zero variance
kddTrain <- kddTrain[, -nearZeroVar(kddTrain)]

# Convert label to factor type to ensure the modeling functions treat label correctly
kddTrain$label <- as.factor(kddTrain$label)

# Remove redundant data from the dataset
kddTrainUnique <- unique(kddTrain)

####################################################################################
# Multiple Linear Regression

# From this point on, work with reduced dataset
attackOrNotTrain <- (kddTrainUnique[,ncol(kddTrainUnique)]=="normal.")

mlrMdl <- glm(attackOrNotTrain ~ .-label,family=binomial(link='logit'),data=kddTrainUnique)
nullMdl <- glm(attackOrNotTrain ~ 1,family=binomial(link='logit'),data=kddTrainUnique)

summary(mlrMdl)


# Forward Attribute Selection
step(nullMdl, scope=list(lower=nullMdl, upper=mlrMdl), direction="forward")

# Backward Attribute Selection
step(mlrMdl, data=trainDataset, direction="backward")

# Stepwise selection of attributes
step(nullMdl, scope=list(lower=nullMdl, upper=mlrMdl), direction="both")



install.packages("caret")
#install.packages("MASS")
require(caret)
flds <- createFolds(kddTrain$label, k = 10, list = TRUE, returnTrain = FALSE)
# Train your classifiers on train1 and test them on test1
test1 <- kddTrain[c(flds[[1]]), ]
train1 <- kddTrain[c(flds[[3]], flds[[4]],
                     flds[[5]], flds[[6]],
                     flds[[7]], flds[[8]],
                     flds[[9]], flds[[10]],
                     flds[[2]]), ]
attackOrNotTrain <- (train1[,ncol(train1)]=="normal.")
attackOrNotTest <- (test1[,ncol(test1)]=="normal.")


reducedModel <- glm(formula = attackOrNotTrain ~ 
             protocol_type + service + flag + src_bytes + logged_in + 
             count + srv_count + serror_rate + srv_serror_rate + rerror_rate + 
             srv_rerror_rate + diff_srv_rate + dst_host_diff_srv_rate + 
             dst_host_same_src_port_rate + dst_host_serror_rate + 
             dst_host_srv_serror_rate + dst_host_rerror_rate, family = binomial(link = "logit"), 
           data = train1)



# Adequacy checking
anova(nullMdl, reducedModel, test="Chisq")

# Analysis of Deviance
DevianceRes=sum(residuals(reducedModel, type = "deviance")^2)
DevianceRes/df.residual(reducedModel)
1 - pchisq(deviance(reducedModel), df.residual(reducedModel))


predictedValues <- predict(reducedModel,
                           data.frame(train1[ , 1:(dim(train1)[2]-1)]),
                           interval = 'prediction',
                           level = 0.95)
#confusion matrix
trainingconfusionmatrix<-100*table(attackOrNotTrain, predictedValues > 0.5)/nrow(train1)
trainingconfusionmatrix
#ROCR Curve
library(ROCR)
ROCRpred <- prediction(predictedValues, attackOrNotTrain)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))


newDataTest <- data.frame(test1[ , 1:(dim(test1)[2]-1)])

predTestData <- predict(reducedModel, newDataTest)

#confusion matrix
100*table(attackOrNotTest, predTestData > 0.5)/nrow(test1)

ROCRpred <- prediction(predTestData, attackOrNotTest)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

TPTN=79.7791827+19.3940457
tptnfnfp= 0.2968821+0.5298895+TPTN
trainingAccuracy= TPTN/tptnfnfp
trainingAccuracy
testAccuracy = (79.8222888+19.4146460)/(79.8222888+19.4146460+0.4776748+0.2732462)

#Studentized 
studentizedresidual <- rstudent(reducedModel)
outliers<-studentizedresidual[abs(studentizedresidual) > 2]
length(outliers)
plot(studentizedresidual,main="Studentized Residual - Cutoff Point 2 ")
abline(h = 2, col="red")  # add cutoff line

#Leverage
leverage <- hat(model.matrix(reducedModel))
thresh2 <- 2*length(reducedModel$coefficients)/length(leverage)
thresh2
#print leverage values above threshold
leverageabovethreshold<-leverage[leverage > thresh2]
length(leverageabovethreshold)
plot(leverage,main="Leverage")
abline(h = thresh2, col="red")  # add cutoff line

# DFFITS 
dffits1 <- dffits(reducedModel)
dffits1
thresh3 <- 2*sqrt(length(reducedModel$coefficients)/length(dffits1))
#print influential points 
influentialpoints<-dffits1[dffits1 > thresh3]
n <- nrow(as.data.frame(kddTrain))
k <- length(reducedModel$coefficients)-1
cv <- 2*sqrt(k/n)

plot(dffits1, 
     ylab = "Standardized dfFits",  
     main = paste("Standardized DfFits, \n critical value = 2*sqrt(k/n) = +/-", round(cv,3)))
#Critical Value horizontal lines
abline(h = cv, lty = 2,col="red")
abline(h = -cv, lty = 2,col="red")

#Values larger than 2*sqrt((k+1)/n) in absolute value are considered highly influential. 
hightlyinfluential<-dffits1[dffits1 > cv]
length(hightlyinfluential)

#The measure ranges from 0 to just under 1, with values closer to zero indicating that the model has no predictive power.
library(pscl)
pR2(reducedModel)  # look for 'McFadden'

#Cook Distance 
cookdistance<-cooks.distance(reducedModel)
plot(cookdistance, pch="*", cex=2, main="Influence by Cooks distance")
abline(h = 4*mean(cookdistance, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cookdistance)+1, y=cookdistance, labels=ifelse(cookdistance>4*mean(cookdistance, na.rm=T),names(cookdistance),""), col="red")
cutoff <- 4/((nrow(kddTrain)-length(reducedModel$coefficients)-2)) 
plot(reducedModel, which = 4,cook.levels=cutoff)

library(car)
#Bubble Plot 
influencePlot(reducedModel)






