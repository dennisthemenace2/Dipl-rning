########### Clear environment
#rm(list = ls())

##create data...

options(error = recover)


sentencenet <- setRefClass("sentencenet",
                           fields = list(U = 'matrix',H='matrix',L = 'matrix', Dh = 'numeric', h_t='matrix'),
                           methods = list(
                             predict = function(pathToDir){
                               
                             },
                             init=function(classes, V){
                               .self$U = matrix(rnorm(.self$Dh* classes,0,0.01),classes,.self$Dh)
                               .self$H = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                               .self$L = matrix(rnorm(.self$Dh* V ,0,0.01),.self$Dh, V)
                             },
                             gradient = function(X,Y,backwards =3){
                               dU = matrix(0, nrow = nrow(.self$U) ,ncol = ncol(.self$U)  )
                               dH = matrix(0,.self$Dh, .self$Dh )
                               #   dL = matrix(0, nrow = nrow(.self$L) ,ncol = ncol(.self$L) )
                               dLidx = c()
                               dLlist = list()
                               
                               Y_pred = .self$forward(X)
                               err = (  Y_pred -Y ) 
                               #print(err)
                               costs= 0
                               ##fix i to last step
                               # i = nrow(X)+1
                               for(i in (nrow(X)+1):2){ ##only ione stime step
                                 #    print('for each row')
                                 #    print(i)
                                 costs = costs -log( Y_pred[i-1, Y[i-1,]==1 ])
                                 delta =  err[i-1,] %*%.self$U  * (.self$h_t[i,] * (1-.self$h_t[i,]))
                                 dU = dU + matrix(err[i-1,] %o% .self$h_t[i,], nrow = nrow(.self$U) ,ncol = ncol(.self$U) ) ##check this %o%
                                 
                                 for(s in 1:backwards){
                                   ts = i - s
                                   
                                   if(ts <= 0){ ## make this better
                                     break
                                   }
                                   #  print('timestep')
                                   # print(ts)
                                   dH = dH + matrix(matrix(delta) %o% .self$h_t[ts,],nrow= .self$Dh, ncol= .self$Dh)
                                   if(any(dLidx ==X[ts]) ){
                                     idx = which(dLidx ==X[ts])
                                     dLlist[[idx]] = dLlist[[idx]] + delta
                                   }else{
                                     dLidx = c(dLidx,X[ts])
                                     dLlist[[length(dLlist)+1]] = delta
                                   }
                                   
                                   
                                   #  dL[,X[ts]] = dL[,X[ts]] + delta
                                   # Update delta
                                   delta =  delta  %*% .self$H  * (.self$h_t[ts,] * (1-.self$h_t[ts,]))
                                 }
                               }
                               #browser()
                               list(costs ,dU,list('idx'=dLidx,'dL'=dLlist) ,dH)
                             },
                             train = function(X,Y,backwards =3,iterations = 1000,lr = 0.001){
                               ## init matricies
                               if(class(X)!='list'){
                                 cat( c('X should be a of type list') )
                                 return();
                               }
                               cat(c(length(X),' training sequences\n' ))
                               if(class(Y[[1]])=='matrix'){
                                 if(ncol(Y[[1]]) == nrow(.self$U)){
                                   cat( c('With classes:',nrow(.self$U) , 'and ',.self$Dh ,'hidden units', 'and vocabal length:', ncol(.self$L),'\n') )
                                 }else{
                                   cat( c('Network not correctly initalizied') ) 
                                   return()
                                 }
                               }else{
                                 cat( c('the elements of this list should be type matrix\n') )
                                 return();
                               }
                               
                               
                               totalCosts = c()
                               for( i in 1:iterations){
                                 costs = 0
                                 for(l in 1:length(X)){##for all training seq
                                   xl = X[[l]]
                                   yl = Y[[l]]
                                   
                                   ret = gradient(xl,yl,backwards)
                                   cost = ret[[1]]
                                   costs = costs + cost
                                   dU =  ret[[2]]
                                   dL = ret[[3]]
                                   dH = ret[[4]]
                                   .self$U=.self$U - (lr* dU)
                                   ##sparse updates fuer L
                                   
                                   #.self$L=.self$L - (lr* dL)
                                   for(kk in 1:length(dL$idx)){
                                     .self$L[,dL$idx[kk]]=.self$L[,dL$idx[kk]] - (lr* dL$dL[[kk]]) 
                                   }
                                   
                                   .self$H=.self$H - (lr* dH)
                                   #  print(dL)
                                   
                                 }
                                 totalCosts = c(totalCosts, costs)
                                 ##take one step
                                 
                                 cat(c('iteration:', i, ' costs:',costs,'\n'))
                               }
                               totalCosts
                             },
                             generate=function(n,init){
                               
                               Y= c()
                               h = matrix(0,.self$Dh,1) ##seed is zeros
                               x = init
                               for(i in 1:n){
                                 z1 = .self$H %*% h + .self$L %*%x
                                 h = .self$sigmoid(z1 )
                                 x = .self$softmax(t(.self$U %*% h) )
                                 Y = c(Y,which.max(x) )
                               }
                               Y
                               
                             },
                             chkGradient= function(){
                               
                               epsilon = 0.00001
                               tol =     0.0000001 
                               
                               X= matrix(c(1,2) ,nrow=2 )
                               Y= matrix(c(0,0,1,0,0,1),ncol=3 ) #3 classes in the end final label only
                               
                               
                               ##init network
                               .self$init(ncol(Y), 2 ) ##classed and alphabet length
                               
                               Ul = .self$U  
                               
                               res = gradient(X,Y)
                               
                               ##check U
                               for(y in 1:ncol(Ul)){
                                 for(x in 1:nrow(Ul)){
                                   .self$U = Ul
                                   .self$U[x,y] = Ul[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$U[x,y] = Ul[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   gradU = res[[2]]
                                   
                                   if(abs(gradU[x,y]-gradaprox)>tol  ){
                                     cat(c(x,'x',y,  ' gradU:',gradU[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               ##check L
                               .self$U = Ul
                               Ll = .self$L
                               for(y in 1:ncol(Ll)){
                                 for(x in 1:nrow(Ll)){
                                   .self$L = Ll
                                   .self$L[x,y] = Ll[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$L[x,y] = Ll[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   gradL = res[[3]]
                                   ikx = which(gradL$idx==y)
                                   if(abs(gradL$dL[[ikx]]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradL:',gradL$dL[[ikx]],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               ##check H
                               .self$L = Ll
                               Hl = .self$H
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$H = Hl
                                   .self$H[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$H[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[4]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradH:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                             },
                             forward=function(X){
                               ##for each row
                               Y= matrix(,ncol=nrow(.self$U),nrow=nrow(X) )
                               .self$h_t = matrix(0,nrow(X)+1,.self$Dh)
                               
                               for(i in 1:nrow(X)){
                                 z1 = .self$H %*% .self$h_t[i,] + .self$L[,X[i,]] 
                                 h = .self$sigmoid(z1 )
                                 .self$h_t[i+1,] = h
                                 
                                 z2 = softmax(t(.self$U %*% h) )
                                 #  z2 = which.max(z2) ##get only idx
                                 Y[i,] =z2
                               }
                               # browser()
                               Y
                             },
                             sigmoid = function(x){
                               x = 1/(1+exp(-x))
                             },
                             sigmoid_der = function(f){
                               f = f*(1-f)
                             },
                             softmax = function(a){
                               exp(a) / rowSums(exp(a)) 
                             },
                             initialize = function(hiddenLayer = 10) {
                               .self$Dh <<- hiddenLayer
                             }
                             
                           ))
###test

sent = sentencenet(10)
#sent$chkGradient()

###simple test
##create test data set

txt = c('aber das ist doch gut',
        'das ist gut',
        'gut ist das',
        'gut aber',
        'schlecht ist das',
        'schlecht doch gut',
        'aber doch schlecht',
        'ist schlecht')
ty = c(1,1,1,1,2,2,2,2)

datasample = data.frame('txt'=txt, 'y'=ty)

words = unique(unlist(datasample$txt))
words = strsplit(unlist(words), " ")
words = unique(unlist(words))

allX =list()
allY = list()


for(i in 1:nrow(datasample) ){
  
  if(!is.na(datasample$y[i]) ){
    ##create example
    txt = datasample$txt[i]
    uw = strsplit(txt, " ")[[1]]
    if(length(uw)>0){
      
      Y =matrix(0,ncol=2,nrow =length(uw) )
      Y[,datasample$y[i]] = 1
      
      X = c()
      for(w in 1:length(uw)){
        idx = which(uw[w]==words )
        if(length(idx)>0 ){
          X = c(X,idx)
        }
      }  
      if(length(X)>0){
        X = matrix(X)
        allX[[length(allX)+1]] = X
        allY[[length(allY)+1]] = Y
      }
    }
    
  }
  cat(c('sample:', i, ' : ',nrow(datasample),'\n') )
  
}


sent$init(2,length(words) ) ##classed and alphabet length

costs = sent$train(allX,allY,iterations = 250,lr = 0.1)
plot(1:length(costs),costs)


##find word labels

wordlabels =c()
for(i in 1:length(words) ){
  res = sent$forward(as.matrix(i) )
  wordlabels =c(wordlabels,which.max(res))
}


plainvec =sent$L

temp = as.matrix(plainvec - rowMeans(plainvec) )
covariance = 1.0 / ncol(plainvec) * temp %*% t(temp)
s= svd(covariance)
coord = t(temp) %*% s$u[,1:2] 
plot(coord,col=wordlabels)

for(i in 1:length(words)){
  text(coord[i,1],coord[i,2],words[i] )
}



