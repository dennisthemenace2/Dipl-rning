########### Clear environment
#rm(list = ls())

## bidirectional implementation of the elman network. added bias terms as well, but I would argue that there is little contribution of them.
##the idea is pretty simple
# h_f+1 = sigmoid(H_f h_f + L[x_t])
# h_b+1 = sigmoid(H_b h_b + L[reverse(x)[t] ])
# y_hat = softmax(U h_stack(h_f,h_b) )
#
## just run foward and backward through the sencente, than stack both and muliply with larger U matrix to brake it down to each class
#
# I guess it adds more stability to the prediction
#

##create data...
options(stringsAsFactors = FALSE)
options(error = recover)


fbsentencenet <- setRefClass("fbsentencenet",
                           fields = list(U = 'matrix',H_f='matrix',H_b='matrix',L = 'matrix', Dh = 'numeric', h_t='matrix',h_b='matrix',b1_f='matrix',b1_b='matrix',b2='matrix'),
                           methods = list(
                             init=function(classes, V){
                               .self$U = matrix(rnorm(.self$Dh* classes,0,0.01),classes,.self$Dh*2)
                               .self$H_f = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                               .self$H_b = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                               .self$L = matrix(rnorm(.self$Dh* V ,0,0.01),.self$Dh, V)
                               
                               .self$b1_f = matrix(rnorm(.self$Dh ,0,0.01), .self$Dh,1)
                               .self$b1_b = matrix(rnorm(.self$Dh ,0,0.01), .self$Dh,1)
                               .self$b2 = matrix(rnorm(classes ,0,0.01), 1,classes)
                             },
                             gradient = function(X,Y,backwards =4){
                               dU = matrix(0, nrow = nrow(.self$U) ,ncol = ncol(.self$U)  )
                               dH_f = matrix(0,.self$Dh, .self$Dh )
                               dH_b = matrix(0,.self$Dh, .self$Dh )
                               
                               dB2= matrix(0,nrow = nrow(.self$b2) ,ncol = ncol(.self$b2) )
                               dB1_f= matrix(0, .self$Dh,1 )
                               dB1_b= matrix(0, .self$Dh,1 )
                               
                               
                               
                               #  dL = matrix(0, nrow = nrow(.self$L) ,ncol = ncol(.self$L) )
                               dLidx = c()
                               dLlist = list()
                               
                               revX = rev(X)
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
                                 errU = (err[i-1,] %*%.self$U)
                                 dB2 = dB2+  err[i-1,]
                                 
                                 delta_f =  errU[1:.self$Dh]  * (.self$h_t[i,] * (1-.self$h_t[i,]))
                                 delta_b =  errU[(.self$Dh+1):(.self$Dh*2) ]  * (.self$h_b[i,] * (1-.self$h_b[i,]))
                                 
                              #   dU = dU + matrix(err[i-1,] %*% t(.self$h_t[i,]), nrow = nrow(.self$U) ,ncol = ncol(.self$U) ) ##check this %o%
                                 dU = dU + matrix(err[i-1,] %*% t(c(.self$h_t[i,],.self$h_b[i,] ) ), nrow = nrow(.self$U) ,ncol = ncol(.self$U) ) ##check this %o%
                                 
                                 for(s in 1:backwards){
                                   ts = i - s
                                   
                                   if(ts <= 0){ ## make this better
                                     break
                                   }
                                   #  print('timestep')
                                   # print(ts)
                                   dH_f = dH_f + matrix(matrix(delta_f) %*% t(.self$h_t[ts,]),nrow= .self$Dh, ncol= .self$Dh)
                                   dH_b = dH_b + matrix(matrix(delta_b) %*% t(.self$h_b[ts,]),nrow= .self$Dh, ncol= .self$Dh)
                                   
                             
                                   dB1_f = dB1_f +matrix(delta_f)
                                   dB1_b = dB1_b +matrix(delta_b)
                                   
                                   
                                   if(any(dLidx ==X[ts]) ){
                                     idx = which(dLidx ==X[ts])
                                     dLlist[[idx]] = dLlist[[idx]] + delta_f
                                   }else{
                                     dLidx = c(dLidx,X[ts])
                                     dLlist[[length(dLlist)+1]] = delta_f
                                   }
                                   
                                   if(any(dLidx ==revX[ts]) ){
                                     idx = which(dLidx ==revX[ts])
                                     dLlist[[idx]] = dLlist[[idx]] + delta_b
                                   }else{
                                     dLidx = c(dLidx,revX[ts])
                                     dLlist[[length(dLlist)+1]] = delta_b
                                   }
                                   
                                   
                                   #  dL[,X[ts]] = dL[,X[ts]] + delta
                                   # Update delta
                                   delta_f =  delta_f  %*% .self$H_f  * (.self$h_t[ts,] * (1-.self$h_t[ts,]))
                                   delta_b =  delta_b  %*% .self$H_b  * (.self$h_b[ts,] * (1-.self$h_b[ts,]))
                                 }
                               }
                               #browser()
                               list(costs ,dU,list('idx'=dLidx,'dL'=dLlist) ,dH_f,dH_b,dB1_f,dB1_b,dB2)
                               #     list(costs ,dU,dL,dH)
                             },
                             train = function(X,Y,backwards =4,iterations = 1000,lr = 0.001){
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
                               
                               
                               #.self$init(ncol(X))
                               totalCosts = c()
                               for( i in 1:iterations){
                                 #dU = matrix(0,ncol = ncol(.self$U),nrow = nrow(.self$U) )
                                 #  dL = matrix(0,ncol = ncol(.self$L),nrow = nrow(.self$L) )
                                 #  dL = NULL
                                 #  dH = matrix(0,ncol = ncol(.self$H),nrow = nrow(.self$H) )
                                 costs = 0
                                 for(l in 1:length(X)){##for all training seq
                                   xl = X[[l]]
                                   yl = Y[[l]]
                                   
                                   ret = gradient(xl,yl,backwards)
                                   cost = ret[[1]]
                                   costs = costs + cost
                                   dU = ret[[2]]
                                   dL = ret[[3]]
                                   dH_f = ret[[4]]
                                   dH_b = ret[[5]]
                                   
                                   db1_f = ret[[6]]
                                   db1_b = ret[[7]]
                                   db2 = ret[[8]]
                                   
                                   .self$U=.self$U - (lr* dU)
                                   ##sparse updates fuer L
                                   #.self$L=.self$L - (lr* dL)
                                   for(kk in 1:length(dL$idx)){
                                     .self$L[,dL$idx[kk]]=.self$L[,dL$idx[kk]] - (lr* dL$dL[[kk]]) 
                                   }
                                   
                                   .self$H_f=.self$H_f - (lr* dH_f)
                                   .self$H_b=.self$H_b - (lr* dH_b)
                                   
                                   .self$b1_f= .self$b1_f - (lr* db1_f)
                                   .self$b1_b=.self$b1_b - (lr* db1_b)
                                   .self$b2=.self$b2 - (lr* db2)
                                   
                                 }
                                 totalCosts = c(totalCosts, costs)
                                 ##take one step
                                 
                                 cat(c('iteration:', i, ' costs:',costs,'\n'))
                               }
                               totalCosts
                             },
                             chkGradient= function(){
                               
                               epsilon = 0.00001
                               tol =     0.000001 
                               
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
                                   gradL = res[[3]]
                                   ikx = which(gradL$idx==y)
                                   if(abs(gradL$dL[[ikx]][x]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradL:',gradL$dL[[ikx]],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               ##check H_f
                               .self$L = Ll
                               Hl = .self$H_f
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$H_f = Hl
                                   .self$H_f[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$H_f[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[4]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradH_f:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               
                               ##check H_b
                               .self$H_f =  Hl
                               Hl = .self$H_b
                               
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$H_b = Hl
                                   .self$H_b[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$H_b[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[5]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradH_b:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               
                               ##check b1_f
                              .self$H_b = Hl 
                               Hl = .self$b1_f
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$b1_f = Hl
                                   .self$b1_f[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$b1_f[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[6]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradB1_f:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               ##check b1_b
                               .self$b1_f = Hl 
                               Hl = .self$b1_b 
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$b1_b = Hl
                                   .self$b1_b[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$b1_b[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[7]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradB1_b:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                               
                               ##check b2
                               .self$b1_b = Hl 
                               Hl = .self$b2
                               for(y in 1:ncol(Hl)){
                                 for(x in 1:nrow(Hl)){
                                   .self$b2 = Hl
                                   .self$b2[x,y] = Hl[x,y]+epsilon
                                   res1 = gradient(X,Y)
                                   .self$b2[x,y] = Hl[x,y]-epsilon
                                   res2 = gradient(X,Y)
                                   
                                   gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                                   #  print(res1[[1]])
                                   gradH = res[[8]]
                                   
                                   if(abs(gradH[x,y]-gradaprox)>tol ){
                                     cat(c(x,'x',y,  ' gradB2:',gradH[x,y],' approximation:',gradaprox,'\n') )
                                   }
                                 }
                               }
                               
                             },
                             forward=function(X){
                               ##for each row
                               Y= matrix(,ncol=nrow(.self$U),nrow=nrow(X) )
                               .self$h_t = matrix(0,nrow(X)+1,.self$Dh)
                               .self$h_b = matrix(0,nrow(X)+1,.self$Dh)
                               
                               for(i in 1:nrow(X)){
                                 z1 = .self$H_f %*% .self$h_t[i,] + .self$L[,X[i,]] 
                                 
                                 z1b = .self$H_b %*% .self$h_b[i,] + .self$L[,X[nrow(X)-i+1,]] 
                                 
                                 
                                 h = .self$sigmoid(z1+.self$b1_f )
                                 .self$h_t[i+1,] = h
                                 
                                 hb = .self$sigmoid(z1b+.self$b1_b )
                                 .self$h_b[i+1,] = hb
                                 
                                 both = rbind(h,hb)
                                 
                                 z2 = t(.self$U %*% both)
                                 y_hat = softmax(z2+.self$b2 )
                                 #  z2 = which.max(z2) ##get only idx
                                 Y[i,] =y_hat
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

#sent = sentencenet(10)
#sent$chkGradient()
sent = fbsentencenet(10)
sent$chkGradient()

###simple test
##create test data set

txt = c('die katze hat zwei augen',
        'ein hund hat vier beine',
        'katze und hund haben beide vier beine',
        'ein katze und hund sind beides haustiere',
        'haben vier beine und zwei augen',
        'eine spinne hat sechs augen',
        'eine ameise hat sechs beine',
        'spinnen und ameisen sind keine haustiere',
        'spinnen und ameisen habe beide sechs beine',
        'haben sechs augen und sechs beine')
ty = c(1,1,1,1,1,2,2,2,2,2)

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