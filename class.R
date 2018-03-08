################################################################################
##################################### INIT #####################################
################################################################################

library(AUC)
library(FNN)
library(e1071)
library(corrplot)
library(randomForest)

set.seed(12345)

################################################################################
################################## LOAD DATA ###################################
################################################################################

theData <- read.delim("input/binary.txt", as.is=TRUE)

################################################################################
################################# PREPARE DATA #################################
################################################################################

######################### MAKE CLASS VARIABLE A FACTOR #########################

theData$Class <- ifelse(theData$CRISPRi_Score_Classes=="Nonfunctional", 0, 1)
theData$Class <- as.factor(theData$Class)

theData$CRISPRi_Score_Classes <- NULL

############################### EQUALIZE CLASSES ###############################

n <- sum(theData$Class==1)

inds1 <- which(theData$Class==1)
inds2 <- sample(which(theData$Class!=1), n)

theData <- theData[c(inds1, inds2),]

######################### SPLIT INTO TEST AND TRAINING #########################

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
}

# select best number of neighbors
accsKNN  <- sapply(Map(`==`, list(trainData$Class), results), mean)
bestK <- noNN[which.max(accsKNN)]

# predict on test data
predKNN <- knn(trainData[,4:7], testData[,4:7], trainData$Class, k=bestK)
predKNN <- factor(predKNN, levels=levels(testData$Class))

probKNN <- attributes(knn(trainData[,4:7], testData[,4:7], trainData$Class, k=bestK, prob=TRUE))$prob
probKNN <- ifelse(predKNN==1, probKNN, 1-probKNN)

################################ RANDOM FOREST #################################

formula <- paste(colnames(trainData)[4:7], collapse=" + ")
formula <- paste(colnames(trainData)[8], "~", formula)
formula <- as.formula(formula)

RF <- randomForest(formula, data=trainData, ntree=10000, importance=TRUE)

predRF <- predict(RF, testData)
probRF <- predict(RF, testData, type="prob")[,"1"]

##################################### SVM ######################################

costs  <- c(0.1, 0.3, 0.5, 0.8, 1, 2, 5, 10, 20, 50)
accsSVM <- numeric(length(costs))
for(i in 1:length(costs)) {
  accsSVM[i] <- svm(formula, data=trainData, cost=costs[i], cross=10)$tot.accuracy
}

# choose cost with lowest error and train on training Data
C <- costs[which.max(accsSVM)]
SVM <- svm(formula, data=trainData, cost=C)

SVM <- svm(formula, data=trainData, cost=10, probability=TRUE)

predSVM <- predict(SVM, testData)

probSVM <- predict(SVM, testData, probability=TRUE)
probSVM <- attributes(probSVM)$prob[,"1"]

################################################################################
################################### RESULTS ####################################
################################################################################

pdf("balanced_results.pdf", width=12, height=8)

par(mfrow=c(2,3))

plot(roc(probKNN, testData$Class), col="green", main="ROC curves", las=1)
plot(roc(probSVM, testData$Class), col="blue", add=TRUE)
plot(roc(probRF, testData$Class), col="red", add=TRUE)
legend <- c(paste("kNN:", round(auc(roc(probKNN, testData$Class)), 2)),
            paste("SVM:", round(auc(roc(probSVM, testData$Class)), 2)),
            paste("RF:", round(auc(roc(probRF, testData$Class)), 2))
            )
legend("topleft", legend=legend, fill=c("green", "blue", "red"), title="AUC")

plot(noNN, accsKNN*100, type="b", lwd=2, col="cornflowerblue", las=1, ylab="accuracy",
     xlab="# nearest neighbors", main="Accuracy by # of nearest neighbors"
     )
abline(v=bestK, col="red", lty=2)

plot(costs, accsSVM, type="b", lwd=2, col="cornflowerblue", las=1, ylab="accuracy",
     xlab="Cost", main="Accuracy by COST parameter (SVM)"
     )
abline(v=C, col="red", lty=2)


acc <- c(kNN=mean(predKNN==testData$Class),
         SVM=mean(predSVM==testData$Class),
         RF=mean(predRF==testData$Class)
         )
barplot(acc, main="accuracy", las=1, ylab="accuracy")
legend("bottomleft", legend=paste(names(acc), ":", round(acc,2)), bg="white",
       title="accuracy"
       )

freqs <- cbind(table(predKNN), table(predSVM), table(predRF))
colnames(freqs) <- c("kNN", "SVM", "RF")
rownames(freqs) <- c("nonfunctional", "highly functional")
barplot(freqs, beside=TRUE, legend=TRUE, args.legend=list(x="topright"), las=1,
        xlim=c(0,14), main="frequency of predictions"
        )

plot(probRF, probSVM, pch=19, cex=0.5, las=1, xlab="Random Forest", ylab="SVM",
     col=ifelse(testData$Class==0, "black", "red"),
     main="prediction probability correlation\n between SVM and RF"
     )
legend("bottomleft", legend=c("High F", "Non F"), fill=c("red", "black"))


dev.off()

