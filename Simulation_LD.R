library("scatterplot3d")
library("kernlab")
library("ggplot2")
library("RANN")
library("latex2exp")
library("class")
library("xtable")

DataTorusGen <- function(m, n, R, r) {
  phi <- rep (2 * pi * runif(m), rep(n,m))
  psi <- 2 * pi * runif(n * m)
  x <- (R  + r * cos(phi)) * cos(psi)
  y <- (R  + r * cos(phi)) * sin(psi)
  z <- r * sin(phi)
  re <- list(data=cbind(x,y,z), mani=cbind(phi,psi))
  return(re)
}

DataSwissGen <- function(m, n) {
  phi <- rep (3 * pi * runif(m), rep(n,m)) + 1.5 * pi
  psi <- 10 * runif(n * m)
  x <- phi * cos(phi)
  y <- phi * sin(phi)
  z <- psi
  re <- list(data=cbind(x,y,z), mani=cbind(phi,psi))
  return(re)
}

DataSwissGenAlter <- function(m, n) {
  phi <- rep (10 * runif(m), rep(n,m))
  psi <- 3 * pi * runif(n * m) + 1.5 * pi
  x <- psi * cos(psi)
  y <- psi * sin(psi)
  z <- phi
  re <- list(data=cbind(x,y,z), mani=cbind(phi,psi))
  return(re)
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

AIDiffusion <-function(data, m, n, sigma, alpha, l, K) {
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
  D=diag(1/colSums(conkernel)^alpha)
  conkernel = D %*% conkernel %*% D
  Dsum=sqrt(colSums(conkernel))
  D=diag(1/Dsum)
  conkernel = D %*% conkernel %*% D
  eigs <- eigen(conkernel, symmetric=TRUE)
  K_eigenvectors <- eigs$vectors[ , 2:(K+1)]
  KK_eigenvectors = matrix(0, n*m, K)
  for (k in 1:K) {
    KK_eigenvectors[,k]=rep(exp(-eigs$values[k+1]*l) * K_eigenvectors[,k]/Dsum, rep(n,m))
  }
  return(KK_eigenvectors)
}

#####################################################
## Torus
#####################################################

m=400
n=3
set.seed(1)
dd = DataTorusGen(m,n,10,5)
#rr <- AILaplacian(dd$data, m, n, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 2)
col <- rainbow(30)[as.numeric(cut(dd$mani[,1],breaks = 30))]

datplot <- data.frame(x1=rr[,1],x2=rr[,2],col=dd$mani[,1])
ggplot(datplot, aes(x=x1, y=x2, color=col)) + 
  geom_point() + 
  scale_color_gradientn(colours=rainbow(100)) +
  theme_bw() +
  labs(colour = "phi", x = "1st Eigenvector", y = "2nd Eigenvector")

## sample size s

trains = c(50,100,200,300)
nummethod = 4
times=100
testing=100
n=3
Re=matrix(0,nrow = length(trains), ncol=nummethod)
tRe=matrix(0,nrow = times, ncol=nummethod)
set.seed(1)
for (i in 1:length(trains)) {
  training=trains[i]
  m=training+testing
  for (t in 1:times) {
    dd = DataTorusGen(m,n,10,5)
    Xd = dd$data[1:training,]
    Xt = dd$data[(training+1):m,]
    Yd = rbinom(training,1,abs(sin(dd$mani[1:training,1])))
    Yt = rbinom(testing,1,abs(sin(dd$mani[(training+1):m,1])))
    
    Ree<-class:::knn(train = Xd, test = Xt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,1]=sum(Ree==Yt)/testing
    
    rr <- AILaplacian(dd$data, m, n, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,2]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,3]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,4]=sum(Ree==Yt)/testing
    
    print(t)
  }
  for(j in 1:nummethod)
  {
    Re[i,j]=mean(tRe[,j])
  }
}
Re
xtable(t(1-Re), digits = 3, type = "latex")

## response function
responsescale = c(1,2,3,4)
nummethod = 4
times=100
testing=100
training=300
m=training+testing
n=3
Re=matrix(0,nrow = length(responsescale), ncol=nummethod)
tRe=matrix(0,nrow = times, ncol=nummethod)
set.seed(1)
for (i in 1:length(trains)) {
  scale=responsescale[i]
  for (t in 1:times) {
    dd = DataTorusGen(m,n,10,5)
    Xd = dd$data[1:training,]
    Xt = dd$data[(training+1):m,]
    Yd = rbinom(training,1,abs(sin(scale*dd$mani[1:training,1])))
    Yt = rbinom(testing,1,abs(sin(scale*dd$mani[(training+1):m,1])))
    
    Ree<-class:::knn(train = Xd, test = Xt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,1]=sum(Ree==Yt)/testing
    
    rr <- AILaplacian(dd$data, m, n, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,2]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,3]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,4]=sum(Ree==Yt)/testing
    
    print(t)
  }
  for(j in 1:nummethod)
  {
    Re[i,j]=mean(tRe[,j])
  }
}
Re
xtable(t(1-Re), digits = 3, type = "latex")

#####################################################
## Swiss roll 2
#####################################################

## sample size s

trains = c(50,100,200,300)
nummethod = 4
times=100
testing=100
n=3
Re=matrix(0,nrow = length(trains), ncol=nummethod)
tRe=matrix(0,nrow = times, ncol=nummethod)
set.seed(1)
for (i in 1:length(trains)) {
  training=trains[i]
  m=training+testing
  for (t in 1:times) {
    dd = DataSwissGenAlter(m,n)
    Xd = dd$data[1:training,]
    Xt = dd$data[(training+1):m,]
    Yd = rbinom(training,1,abs(sin(dd$mani[1:training,1])))
    Yt = rbinom(testing,1,abs(sin(dd$mani[(training+1):m,1])))
    
    Ree<-class:::knn(train = Xd, test = Xt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,1]=sum(Ree==Yt)/testing
    
    rr <- AILaplacian(dd$data, m, n, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,2]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,3]=sum(Ree==Yt)/testing
    
    rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
    XXd = rr[1:training,]
    XXt = rr[(training+1):m,]
    
    Ree<-class:::knn(train = XXd, test = XXt, cl = factor(Yd), k = 10)
    Ree<-as.numeric(as.character(Ree))
    tRe[t,4]=sum(Ree==Yt)/testing
    
    print(t)
  }
  for(j in 1:nummethod)
  {
    Re[i,j]=mean(tRe[,j])
  }
}
Re
xtable(t(1-Re), digits = 3, type = "latex")

#####################################################
## Manifold Comparison
#####################################################

m=300
n=3
set.seed(1)
dd = DataTorusGen(m,n,10,5)
rr <- AILaplacian(dd$data, m, n, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
col <- rainbow(30)[as.numeric(cut(dd$mani[,1],breaks = 30))]

datplot <- data.frame(x1=rr[,1],x2=rr[,2],col=dd$mani[,1])
ggplot(datplot, aes(x=x1, y=x2, color=col)) + 
  geom_point() + 
  scale_color_gradientn(colours=rainbow(100)) +
  theme_bw() +
  labs(colour = "phi", x = "1st Eigenvector", y = "2nd Eigenvector")

set.seed(1)
dd = DataSwissGen(m,n)
rr <- AILaplacian(dd$data, m, n, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
col <- rainbow(30)[as.numeric(cut(dd$mani[,1],breaks = 30))]

datplot <- data.frame(x1=rr[,1],x2=rr[,2],col=dd$mani[,1])
ggplot(datplot, aes(x=x1, y=x2, color=col)) + 
  geom_point() + 
  scale_color_gradientn(colours=rainbow(100)) +
  theme_bw() +
  labs(colour = "phi", x = "1st Eigenvector", y = "2nd Eigenvector")

set.seed(1)
dd = DataSwissGenAlter(m,n)
rr <- AILaplacian(dd$data, m, n, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1/2, 0.1, 2)
rr <- AIDiffusion(dd$data, m, n, 0.1, 1, 0.1, 2)
col <- rainbow(30)[as.numeric(cut(dd$mani[,1],breaks = 30))]

datplot <- data.frame(x1=rr[,1],x2=rr[,2],col=dd$mani[,1])
ggplot(datplot, aes(x=x1, y=x2, color=col)) + 
  geom_point() + 
  scale_color_gradientn(colours=rainbow(100)) +
  theme_bw() +
  labs(colour = "phi", x = "1st Eigenvector", y = "2nd Eigenvector")



