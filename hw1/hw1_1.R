library(ISLR)
dim(Weekly)
summary(Weekly)
#pairs(Weekly)
cor(Weekly[,-9])
attach(Weekly) # we can use variables by names below
plot(Volume)

glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family=binomial)
summary(glm.fits)

glm.probs=predict(glm.fits,type="response")
glm.probs[1:10]
contrasts(Direction)
glm.pred=rep("Down",1089)
glm.pred[glm.probs>.5]="Up" # classify as "up" if glm.probs>0.5??????
table(glm.pred,Direction)
(54+557)/1089 # prob of correct classification for training data
mean(glm.pred==Direction)
roc_true=rep(0,99)
roc_false=rep(0,99)
for (i in 1:99){
  glm.pred=rep("Down",1089)
  glm.pred[glm.probs>(i/100)]="Up"
  roc_true[i]=sum(glm.pred[glm.pred==Direction]=="Down")/(sum(glm.pred[glm.pred==Direction]=="Down")+sum(glm.pred[glm.pred!=Direction]=="Up"))
  roc_false[i]=sum(glm.pred[glm.pred!=Direction]=="Down")/(sum(glm.pred[glm.pred!=Direction]=="Down")+sum(glm.pred[glm.pred==Direction]=="Up"))
}
plot(roc_true,roc_false,xlab = "False positive rate",ylab = "True positive rate",main = "logistic regression full X ROC Curve")



train=(Year<2009) # training date
Weekly.20092010=Weekly[!train,] # test date ????????
dim(Weekly.20092010)
Direction.20092010=Direction[!train]

glm.fits=glm(Direction~Lag2,data=Weekly,family=binomial,subset=train)
glm.probs=predict(glm.fits,Weekly.20092010,type="response")
glm.pred=rep("Down",104)
glm.pred[glm.probs>(50/100)]="Up"
table(glm.pred,Direction.20092010)
mean(glm.pred==Direction.20092010)
mean(glm.pred!=Direction.20092010)
roc_true=rep(0,99)
roc_false=rep(0,99)
for (i in 1:99){
glm.pred=rep("Down",104)
glm.pred[glm.probs>(i/100)]="Up"
table(glm.pred,Direction.20092010)

roc_true[i]=sum(glm.pred[glm.pred==Direction.20092010]=="Down")/(sum(glm.pred[glm.pred==Direction.20092010]=="Down")+sum(glm.pred[glm.pred!=Direction.20092010]=="Up"))
roc_false[i]=sum(glm.pred[glm.pred!=Direction.20092010]=="Down")/(sum(glm.pred[glm.pred!=Direction.20092010]=="Down")+sum(glm.pred[glm.pred==Direction.20092010]=="Up"))
}
plot(roc_true,roc_false,xlab = "False positive rate",ylab = "True positive rate",main = "logistic regression Lag2 ROC Curve")
#roc_true
#roc_false



library(MASS)
lda.fit=lda(Direction~Lag2,data=Weekly,subset=train)
lda.fit
plot(lda.fit)
lda.pred=predict(lda.fit, Weekly.20092010)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,Direction.20092010)
mean(lda.class==Direction.20092010)
sum(lda.pred$posterior[,1]>=.5)
sum(lda.pred$posterior[,1]<.5)
lda.pred$posterior[1:20,1]
lda.class[1:20]
sum(lda.pred$posterior[,1]>.9)

library(ROCR)
# choose the posterior probability column carefully, it may be 
# lda.pred$posterior[,1] or lda.pred$posterior[,2], depending on your factor levels 
pred <- prediction(lda.pred$posterior[,1], Direction.20092010) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)







library(class)
?knn
train.X=cbind(Lag1,Lag2,Lag3,Lag4,Lag5,Volume)[train,]
test.X=cbind(Lag1,Lag2,Lag3,Lag4,Lag5,Volume)[!train,]
train.Direction=Direction[train]
set.seed(100)

# knn with the closest neighbor in the feature space X
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.20092010)
mean(knn.pred==Direction.20092010)
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.20092010)
mean(knn.pred==Direction.20092010)

kk=c(170,165,160,155,150,145,140,135,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
error_rate_l_test=rep(0,length(kkk))
error_rate_l_train=rep(0,length(kkk))
for (i in 1:length(kk)){
  knn.pred=knn(train.X,test.X,train.Direction,k=kk[i])
  error_rate_l_test[i]=(1-mean(knn.pred==Direction.20092010))
  knn.pred=knn(train.X,train.X,train.Direction,k=kk[i])
  error_rate_l_train[i]=(1-mean(knn.pred==Direction[train]))
}
plot(1/kk,error_rate_l_train,ylim = c(0,0.55),xlab="1/K",ylab = "Error Rate",type="l",col="red")
par(new=TRUE)
plot(1/kk,error_rate_l_test,ylim = c(0,0.55),xlab="",ylab = "",type="l",col="green")
legend(
  "bottomleft", 
  lty=c(1,2,1,2), 
  col=c("red", "green"), 
  legend = c("Training Errors", "Test Errors")
)

set.seed(123)

library("caret")
library(knitr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ROCR)
#- Define controls
x = trainControl(method = "repeatedcv",
                 number = 10,
                 repeats = 3,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

#- train model
knn = train( train.X,train.Direction, method = "knn",
            preProcess = c("center","scale"),
            trControl = x,
            metric = "ROC",
            tuneLength = 10)

# print model results
knn
plot(knn)
Predicted = predict(knn, test.X, "prob")[,2]

#- Area Under Curve
plot(performance(prediction(Predicted, Direction.20092010),
                 "tpr", "fpr"), main = "knn")
