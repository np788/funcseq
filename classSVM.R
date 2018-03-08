################################################################################
##################################### INIT #####################################
################################################################################

library(e1071)
library(randomForest)

################################################################################
################################## LOAD DATA ###################################
################################################################################

theData <- read.table("input/data.txt", header=TRUE, as.is=TRUE)

################################################################################
################################# PREPARE DATA #################################
################################################################################

########################### REMOVE CONSTANT FEATURES ###########################

noUnique <- apply(theData, 2, function(x) length(unique(x)))

badFeatures <- noUnique==1 & seq_len(ncol(theData)) > 3

theData <- theData[,!badFeatures]

######################### SPLIT INTO TEST AND TRAINING #########################

set.seed(141414) # fix a random seet for repeatability

percentTrain <- 0.7
trainInds <- sample(nrow(theData), round(nrow(theData)*0.7))

trainData <- theData[trainInds,]
testData  <- theData[-trainInds,]

################################################################################
################################### CLASSIFY ###################################
################################################################################

################################### FORMULA ####################################

formula <- paste(colnames(theData)[-(1:3)], collapse=" + ")
formula <- paste(colnames(theData)[3], "~", formula)
formula <- as.formula(formula)

##################################### SVM ######################################

# Next we will try SVM with radial kernel using various values of the cost
# parameter. 10-fold cross-validation will be done on each chosen parameter
# value and the final value will be selected based on the cross validation
# results

costs  <- c(0.1, 0.3, 0.5, 0.8, 1, 2, 5, 10)
errors <- numeric(length(costs))
for(i in 1:length(costs)) {
  errors[i] <- svm(formula, data=trainData, cost=costs[i], cross=10)$tot.MSE
  print(i)
}

# choose cost with lowest error and train on training Data
C <- costs[which.min(errors)]
SVM <- svm(formula, data=trainData, cost=C)

################################ RANDOM FOREST #################################

# Using random forest classifier.

RF <- randomForest(formula, data=trainData, ntree=10000, importance=TRUE)

################################################################################
################################### RESULTS ####################################
################################################################################

predictedSVM  <- predict(SVM, testData)
predictedRF   <- predict(RF, testData)
predictedMEAN <- rep(mean(trainData$CRISPRi_Score), nrow(testData))

# Unreducable error (error done on features with all 0es)
zeroInds  <- rowSums(testData[,4:ncol(testData)])==0
zeroMean  <- mean(testData$CRISPRi_Score[zeroInds])
zeroError <- mean((testData$CRISPRi_Score[zeroInds] - zeroMean)^2)

rmses <- c(mean=mean((predictedMEAN-testData$CRISPRi_Score)^2),
           SVM=mean((predictedSVM-testData$CRISPRi_Score)^2),
           RF=mean((predictedRF-testData$CRISPRi_Score)^2),
           unreducable=zeroError
           )

pdf("results.pdf", width=12, height=12)

par(mfrow=c(3,3))

plot(costs, errors, type="b", col="cornflowerblue", xlab="Cost", ylab="MSE",
     las=1, main="SVM Cost parameter vs MSE", lwd=3
     )

plot(testData$CRISPRi_Score, predictedSVM, pch=19, cex=0.5, las=1,
     xlab="real test values", ylab="SVM prediction",
     main="SVM prediction vs Real", col=ifelse(zeroInds, "red", "black")
     )

plot(testData$CRISPRi_Score, abs(testData$CRISPRi_Score-predictedSVM), las=1,
     pch=19, cex=0.5, xlab="real test values", ylab="absolute error",
     main="SVM Real vs Error", col=ifelse(zeroInds, "red", "black")
     )

par(mar=c(5, 8, 4, 2))
barplot(RF$importance[,1], horiz=TRUE, las=1, main="RF variable importance",
        xlab="increase in MSE when ommited"
        )

plot(testData$CRISPRi_Score, predictedRF, pch=19, cex=0.5, las=1,
     xlab="real test values", ylab="RF prediction",
     main="RF prediction vs Real", col=ifelse(zeroInds, "red", "black")
     )

plot(testData$CRISPRi_Score, abs(testData$CRISPRi_Score-predictedRF), las=1,
     pch=19, cex=0.5, xlab="real test values", ylab="absolute error",
     main="RF Real vs Error", col=ifelse(zeroInds, "red", "black")
     )

plot(predictedSVM, predictedRF, xlab="predicted SVM", ylab="predicted RF",
     main="SVM vs RF predictions", pch=19, cex=0.5, las=1
     )

plot(abs(predictedSVM-testData$CRISPRi_Score),
     abs(predictedRF-testData$CRISPRi_Score), xlab="SVM errors",
     ylab="RF errors", main="SVM vs RF errors", pch=19, cex=0.5, las=1
     )

barplot(sort(rmses, decreasing=TRUE), las=1, col="cornflowerblue", ylab="MSE",
        main="error comparison"
        )
legend("bottomleft", legend=paste(names(rmses), round(rmses, 4)), bg="white")

dev.off()


