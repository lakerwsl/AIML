library(keras)
library(EBImage)
library(kernlab)
library(foreach)
library(doParallel)
library(doRNG)

DataAugmentation <- function(X,nview)
{
  SS=dim(X)
  numimg=SS[1]
  DataX=matrix(0,nrow = numimg*nview, ncol=SS[2]*SS[3])
  for (i in 1:numimg)
  {
    Timg=X[i,,]/255
    start=(i-1)*nview
    DataX[start+1,]=c(Timg)
    
    for (t in 2:nview)
    {
      nsize=sample(29:32, 1)
      crop=sample(0:(nsize-28), 1)
      Rimg=resize(Timg,nsize,nsize)[(crop+1):(crop+28),(crop+1):(crop+28)]
      # if (runif(1)>0.5) {
      #   Rimg=flop(Rimg)
      # }
      #Rimg=Rimg*runif(1,0.8,1)
      #w = makeBrush(size = 5, shape = 'gaussian', sigma = runif(1,0.1,1))
      #Rimg=filter2(Rimg, w)
      DataX[start+t,] = c(Rimg)
    }
  }
  return(DataX)
}

DataAugmentation2 <- function(X,nview)
{
  SS=dim(X)
  numimg=SS[1]
  DataX=matrix(0,nrow = numimg*nview, ncol=SS[2]*SS[3])
  for (i in 1:numimg)
  {
    Timg=X[i,,]/255
    start=(i-1)*nview
    DataX[start+1,]=c(Timg)
    
    for (t in 2:nview)
    {
      Rimg=rotate(Timg,runif(1,-10,10),output.dim=c(28,28))
      nsize=sample(29:32, 1)
      crop=sample(0:(nsize-28), 1)
      Rimg=resize(Rimg,nsize,nsize)[(crop+1):(crop+28),(crop+1):(crop+28)]
      # if (runif(1)>0.5) {
      #   Rimg=flop(Rimg)
      # }
      #Rimg=Rimg*runif(1,0.8,1)
      #w = makeBrush(size = 5, shape = 'gaussian', sigma = runif(1,0.1,1))
      #Rimg=filter2(Rimg, w)
      DataX[start+t,] = c(Rimg)
    }
  }
  return(DataX)
}

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

AILaplacian <-function(data, m, n, sigma, K) {
  rbf <- rbfdot(sigma = sigma)
  kernel <- kernelMatrix(rbf, data)
  conkernel <- matrix(0,m,m)
  for (i in 1:m) {
    for (j in 1:m) {
      indexi = (n * (i-1) + 1) : (n * i)
      indexj = (n * (j-1) + 1) : (n * j)
      conkernel[i, j] = mean(kernel[indexi, indexj])
    }
  }
  Dsum=sqrt(colSums(conkernel))
  D=diag(1/Dsum)
  conkernel = D %*% conkernel %*% D
  eigs <- eigen(conkernel, symmetric=TRUE)
  K_eigenvectors <- eigs$vectors[ , 2:(K+1)]
  KK_eigenvectors = matrix(0, n*m, K)
  for (k in 1:K) {
    KK_eigenvectors[,k]=rep(K_eigenvectors[,k]/Dsum, rep(n,m))
  }
  return(KK_eigenvectors)
}

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

Timg=x_train[10,,]
Timg=Timg/255
Timg=flop(rotate(Timg,90))
display(Timg)
writeImage(Timg, "MNIST_original.jpeg", quality = 85)

set.seed(5)
nsize=sample(29:32, 1)
crop=sample(0:(nsize-28), 1)
Rimg=resize(Timg,nsize,nsize)[(crop+1):(crop+28),(crop+1):(crop+28)]
writeImage(Rimg, "MNIST_crop.jpeg", quality = 85)

Rimg=rotate(Timg,runif(1,-10,10),output.dim=c(28,28))
nsize=sample(29:32, 1)
crop=sample(0:(nsize-28), 1)
Rimg=resize(Rimg,nsize,nsize)[(crop+1):(crop+28),(crop+1):(crop+28)]
writeImage(Rimg, "MNIST_crop_rotate.jpeg", quality = 85)


## sample size 50

SelfTrainSize=1000
TrainSize=50
TestSize=100
m=TrainSize+TestSize
nview=7

set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class','EBImage','kernlab')) %dorng% {
  SelfTrainIndex=sample(1:60000,SelfTrainSize+TrainSize)
  TrainIndex=SelfTrainIndex[1:TrainSize]
  SelfTrainIndex=SelfTrainIndex[-(1:TrainSize)]
  TestIndex=sample(1:10000,TestSize)
  
  SelfTrainX=DataAugmentation(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation(x_train[TrainIndex,,],nview)
  TrainXX=DataTransformation(x_train[TrainIndex,,])
  TestX=DataAugmentation(x_test[TestIndex,,],nview)
  TestXX=DataTransformation(x_test[TestIndex,,])
  TrainLabel=y_train[TrainIndex]
  TestLabel=y_test[TestIndex]
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = 3)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.04, 15)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 3)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  SelfTrainX=DataAugmentation2(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation2(x_train[TrainIndex,,],nview)
  TestX=DataAugmentation2(x_test[TestIndex,,],nview)
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.04, 15)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 3)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

## sample size 100

SelfTrainSize=1000
TrainSize=100
TestSize=100
m=TrainSize+TestSize
nview=7

set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class','EBImage','kernlab')) %dorng% {
  SelfTrainIndex=sample(1:60000,SelfTrainSize+TrainSize)
  TrainIndex=SelfTrainIndex[1:TrainSize]
  SelfTrainIndex=SelfTrainIndex[-(1:TrainSize)]
  TestIndex=sample(1:10000,TestSize)
  
  SelfTrainX=DataAugmentation(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation(x_train[TrainIndex,,],nview)
  TrainXX=DataTransformation(x_train[TrainIndex,,])
  TestX=DataAugmentation(x_test[TestIndex,,],nview)
  TestXX=DataTransformation(x_test[TestIndex,,])
  TrainLabel=y_train[TrainIndex]
  TestLabel=y_test[TestIndex]
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = 4)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 15)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 4)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  SelfTrainX=DataAugmentation2(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation2(x_train[TrainIndex,,],nview)
  TestX=DataAugmentation2(x_test[TestIndex,,],nview)
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 15)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 4)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)


## sample size 200

SelfTrainSize=1000
TrainSize=200
TestSize=100
m=TrainSize+TestSize
nview=7

set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class','EBImage','kernlab')) %dorng% {
  SelfTrainIndex=sample(1:60000,SelfTrainSize+TrainSize)
  TrainIndex=SelfTrainIndex[1:TrainSize]
  SelfTrainIndex=SelfTrainIndex[-(1:TrainSize)]
  TestIndex=sample(1:10000,TestSize)
  
  SelfTrainX=DataAugmentation(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation(x_train[TrainIndex,,],nview)
  TrainXX=DataTransformation(x_train[TrainIndex,,])
  TestX=DataAugmentation(x_test[TestIndex,,],nview)
  TestXX=DataTransformation(x_test[TestIndex,,])
  TrainLabel=y_train[TrainIndex]
  TestLabel=y_test[TestIndex]
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 20)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  SelfTrainX=DataAugmentation2(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation2(x_train[TrainIndex,,],nview)
  TestX=DataAugmentation2(x_test[TestIndex,,],nview)
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 20)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

## sample size 400

SelfTrainSize=1000
TrainSize=400
TestSize=100
m=TrainSize+TestSize
nview=7

set.seed(1)
times=100
cl <- makeCluster(5) #not to overload your computer
registerDoParallel(cl)

Re<-foreach(t=1:times, .combine=rbind, .packages=c('class','EBImage','kernlab')) %dorng% {
  SelfTrainIndex=sample(1:60000,SelfTrainSize+TrainSize)
  TrainIndex=SelfTrainIndex[1:TrainSize]
  SelfTrainIndex=SelfTrainIndex[-(1:TrainSize)]
  TestIndex=sample(1:10000,TestSize)
  
  SelfTrainX=DataAugmentation(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation(x_train[TrainIndex,,],nview)
  TrainXX=DataTransformation(x_train[TrainIndex,,])
  TestX=DataAugmentation(x_test[TestIndex,,],nview)
  TestXX=DataTransformation(x_test[TestIndex,,])
  TrainLabel=y_train[TrainIndex]
  TestLabel=y_test[TestIndex]
  
  Ree<-knn(train = TrainXX, test = TestXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re1=sum(Ree==TestLabel)/TestSize
  
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 20)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re2=sum(Ree==TestLabel)/TestSize
  
  SelfTrainX=DataAugmentation2(x_train[SelfTrainIndex,,],nview)
  TrainX=DataAugmentation2(x_train[TrainIndex,,],nview)
  TestX=DataAugmentation2(x_test[TestIndex,,],nview)
  
  rr <- AILaplacian(rbind(TrainX,TestX,SelfTrainX), m+SelfTrainSize, nview, 0.03, 20)
  rr <- rr[seq(1,nrow(rr),nview),]
  
  TrainXXX=rr[1:TrainSize,]
  TestXXX=rr[(TrainSize+1):m,]
  Ree<-knn(train = TrainXXX, test = TestXXX, cl = factor(TrainLabel), k = 5)
  Ree<-as.numeric(as.character(Ree))
  Re3=sum(Ree==TestLabel)/TestSize
  
  c(Re1,Re2,Re3)
}
stopCluster(cl)
apply(1-Re,2,mean)

