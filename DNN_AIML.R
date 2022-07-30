library(reticulate)
library(foreach)
library(doParallel)
library(doRNG)

DataTransformation <- function(X)
{
  SS=dim(X)
  numimg=SS[1]
  DataX=matrix(0,nrow = numimg, ncol=SS[2]*SS[3])
  for (i in 1:numimg)
  {
    Timg=X[i,,]
    DataX[i,]=c(Timg/255)
  }
  return(DataX)
}

np <- import("numpy")
npz1 <- np$load("trandata_mnist.npz")
npz1$files
x_train <- npz1$f[["x_train"]]
x_train_tran <- npz1$f[["x_train_tran"]]
y_train <- npz1$f[["y_train"]]

x_test <- npz1$f[["x_test"]]
x_test_tran <- npz1$f[["x_test_tran"]]
y_test <- npz1$f[["y_test"]]

x_train<-DataTransformation(x_train)
x_test<-DataTransformation(x_test)

npz2 <- np$load("trandata_mnist_rotation.npz")
x_train_tran2 <- npz2$f[["x_train_tran"]]
x_test_tran2 <- npz2$f[["x_test_tran"]]

x_train2 <- npz2$f[["x_train"]]
y_train2 <- npz2$f[["y_train"]]
x_test2 <- npz2$f[["x_test"]]
y_test2 <- npz2$f[["y_test"]]

## sample size 50

TrainSize=50
TestSize=100
set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class')) %dorng% {
  kk=3
  TrainIndex=sample(1:60000,TrainSize)
  TestIndex=sample(1:10000,TestSize)
  
  TrainX=x_train[TrainIndex,]
  TrainXX=x_train_tran[TrainIndex,]
  TrainXXX=x_train_tran2[TrainIndex,]
  TrainLabel=y_train2[TrainIndex]
  TestX=x_test[TestIndex,]
  TestXX=x_test_tran[TestIndex,]
  TestXXX=x_test_tran2[TestIndex,]
  TestLabel=y_test2[TestIndex]
  
  Ree<-knn(train = TrainX, test = TestX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

## sample size 100
TrainSize=100
TestSize=100
set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class')) %dorng% {
  kk=4
  TrainIndex=sample(1:60000,TrainSize)
  TestIndex=sample(1:10000,TestSize)
  
  TrainX=x_train[TrainIndex,]
  TrainXX=x_train_tran[TrainIndex,]
  TrainXXX=x_train_tran2[TrainIndex,]
  TrainLabel=y_train2[TrainIndex]
  TestX=x_test[TestIndex,]
  TestXX=x_test_tran[TestIndex,]
  TestXXX=x_test_tran2[TestIndex,]
  TestLabel=y_test2[TestIndex]
  
  Ree<-knn(train = TrainX, test = TestX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

## sample size 200
TrainSize=200
TestSize=100
set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class')) %dorng% {
  kk=5
  TrainIndex=sample(1:60000,TrainSize)
  TestIndex=sample(1:10000,TestSize)
  
  TrainX=x_train[TrainIndex,]
  TrainXX=x_train_tran[TrainIndex,]
  TrainXXX=x_train_tran2[TrainIndex,]
  TrainLabel=y_train2[TrainIndex]
  TestX=x_test[TestIndex,]
  TestXX=x_test_tran[TestIndex,]
  TestXXX=x_test_tran2[TestIndex,]
  TestLabel=y_test2[TestIndex]
  
  Ree<-knn(train = TrainX, test = TestX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

## sample size 400
TrainSize=400
TestSize=100
set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class')) %dorng% {
  kk=5
  TrainIndex=sample(1:60000,TrainSize)
  TestIndex=sample(1:10000,TestSize)
  
  TrainX=x_train[TrainIndex,]
  TrainXX=x_train_tran[TrainIndex,]
  TrainXXX=x_train_tran2[TrainIndex,]
  TrainLabel=y_train2[TrainIndex]
  TestX=x_test[TestIndex,]
  TestXX=x_test_tran[TestIndex,]
  TestXXX=x_test_tran2[TestIndex,]
  TestLabel=y_test2[TestIndex]
  
  Ree<-knn(train = TrainX, test = TestX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = kk)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)



