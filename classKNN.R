################################################################################
##################################### INIT #####################################
################################################################################

library(AUC)
library(FNN)
library(e1071)
library(corrplot)
library(randomForest)

################################################################################
################################## LOAD DATA ###################################
################################################################################

theData <- read.delim("input/classified.txt", as.is=TRUE)

################################################################################
################################# PREPARE DATA #################################
################################################################################

########################### REMOVE CONSTANT FEATURES ###########################

noUnique <- apply(theData, 2, function(x) length(unique(x)))

badFeatures <- noUnique==1 & seq_len(ncol(theData)) > 3

theData <- theData[,!badFeatures]

######################### MAKE CLASS VARIABLE A FACTOR #########################

levels <- c("Nonfunctional", "Low Functionality", "Moderate Functionality",
            "High Functionality", "Very High Functionality"
            )
theData$Class <- factor(theData$CRISPRi_Score_Classes, levels=levels)
theData$CRISPRi_Score_Classes <- NULL

######################### SPLIT INTO TEST AND TRAINING #########################

set.seed(141414) # fix a random seet for repeatability

percentTrain <- 0.7
trainInds <- sample(nrow(theData), round(nrow(theData)*0.7))

trainData <- theData[trainInds,]
testData  <- theData[-trainInds,]

################################################################################
################################### CLASSIFY ###################################
################################################################################

##################################### KNN ######################################

# 10-fold cross validation will be used to select the number of nearest
# neighbors.

noNN  <- c(1, 3, 5, 7, 11, 15, 21, 31, 51)
results <- vector(length(noNN), mode="list")
for(i in 1:length(noNN)) {
  folds <- sample(ceiling((1:nrow(trainData))/nrow(trainData) * 10))
  for(f in unique(folds)) {
    inds <- folds==f
    results[[i]][inds] <- as.character(FNN::knn(trainData[inds,4:7],
                                                trainData[inds,4:7],
                                                trainData$Class[inds], noNN[i]
                                                ))
  }
  print(i)
}

# select best number of neighbors
accs  <- sapply(Map(`==`, list(trainData$Class), results), mean)
bestK <- noNN[which.max(accs)]

# predict on test data
predKNN <- knn(trainData[,4:7], testData[,4:7], trainData$Class, k=bestK)
predKNN <- factor(predKNN, levels=levels(testData$Class))

################################ RANDOM FOREST #################################

formula <- paste(colnames(trainData)[4:7], collapse=" + ")
formula <- paste(colnames(trainData)[8], "~", formula)
formula <- as.formula(formula)

RF <- randomForest(formula, data=trainData, ntree=10000, importance=TRUE)

predRF <- predict(RF, testData)

##################################### SVM ######################################

SVM <- svm(formula, data=trainData, cost=10)

predSVM <- predict(SVM, testData)

################################################################################
################################### RESULTS ####################################
################################################################################

pdf("results.pdf", width=16, height=12)

par(mfrow=c(3,4))

# confusion matrix
tab <- table(predKNN, testData$Class)
precTab   <- tab/rowSums(tab)
recallTab <- t(t(tab) / rowSums(t(tab)))

barplot(cbind(diag(precTab), diag(recallTab)), beside=TRUE, legend=TRUE,
        names=c("precision", "recall"), las=1, main="kNN accuracies",
        ylim=c(0, 1.5), args.legend=list(x="topleft")
        )
abline(h=mean(predKNN==testData$Class), lwd=2, col="red")
legend("topright", legend="total accuracy", lty=1, col="red")
text(1, mean(predKNN==testData$Class)+0.1, mean(predKNN==testData$Class), col="red", pos=4)

corrplot(tab, is.corr=FALSE, tl.cex=0.5, method="number", col="black", cl.pos="n")
mtext("predicted", 2, line=2.5, cex=0.7)
mtext("confusion matrix", 3, line=2.5, cex=0.7)

corrplot(precTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("precision", 3, line=2.5, cex=0.7)

corrplot(recallTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("recall", 3, line=2.5, cex=0.7)

# confusion matrix
tab <- table(predSVM, testData$Class)
precTab   <- tab/rowSums(tab)
recallTab <- t(t(tab) / rowSums(t(tab)))

barplot(cbind(diag(precTab), diag(recallTab)), beside=TRUE, legend=TRUE,
        names=c("precision", "recall"), las=1, main="SVM accuracies",
        ylim=c(0, 1.5), args.legend=list(x="topleft")
        )
abline(h=mean(predSVM==testData$Class), lwd=2, col="red")
legend("topright", legend="total accuracy", lty=1, col="red")
text(1, mean(predSVM==testData$Class)+0.1, mean(predSVM==testData$Class), col="red", pos=4)

corrplot(tab, is.corr=FALSE, tl.cex=0.5, method="number", col="black", cl.pos="n")
mtext("predicted", 2, line=2.5, cex=0.7)
mtext("confusion matrix", 3, line=2.5, cex=0.7)

corrplot(precTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("precision", 3, line=2.5, cex=0.7)

corrplot(recallTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("recall", 3, line=2.5, cex=0.7)

# confusion matrix
tab <- table(predRF, testData$Class)
precTab   <- tab/rowSums(tab)
recallTab <- t(t(tab) / rowSums(t(tab)))

barplot(cbind(diag(precTab), diag(recallTab)), beside=TRUE, legend=TRUE,
        names=c("precision", "recall"), las=1, main="RF accuracies",
        ylim=c(0, 1.5), args.legend=list(x="topleft")
        )
abline(h=mean(predRF==testData$Class), lwd=2, col="red")
legend("topright", legend="total accuracy", lty=1, col="red")
text(1, mean(predRF==testData$Class)+0.1, mean(predRF==testData$Class), col="red", pos=4)

corrplot(tab, is.corr=FALSE, tl.cex=0.5, method="number", col="black", cl.pos="n")
mtext("predicted", 2, line=2.5, cex=0.7)
mtext("confusion matrix", 3, line=2.5, cex=0.7)

corrplot(precTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("precision", 3, line=2.5, cex=0.7)

corrplot(recallTab, tl.cex=0.5, method="shade", addCoef.col="black")
mtext("recall", 3, line=2.5, cex=0.7)

dev.off()

