########### Clear environment
rm(list = ls())

##create data...

options(error = recover)

x = seq(0,10,1)
y = 0.5 *x +1.2

y = (y-min(y))/(max(y)-min(y))

####
#
#

lstm <- setRefClass("lstm",
                     fields = list(
                                   Wi = 'matrix',Ui ='matrix',
                                   Wf = 'matrix',Uf ='matrix',
                                   Wo = 'matrix',Uo ='matrix',
                                   Wc = 'matrix',Uc ='matrix',
                                   c_t= 'matrix',
                                   ai_t = 'matrix',
                                   af_t = 'matrix',
                                   ao_t = 'matrix',
                                   ac_t ='matrix',
                                   Dh = 'numeric' ),
                     methods = list(
                       init=function(dimx){
                         
                         .self$Ui = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uf = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uo = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uc = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         
                         .self$Wi = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wf = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wo = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wc = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         
                       },
                       gradient = function(X,Y){
                       
                         dWo = matrix(0,  nrow = nrow(.self$Wo) ,ncol = ncol(.self$Wo) )
                         dUo = matrix(0,  nrow = nrow(.self$Uo) ,ncol = ncol(.self$Uo) )
                  
                         dWi = matrix(0,  nrow = nrow(.self$Wi) ,ncol = ncol(.self$Wi) )
                         dUi = matrix(0,  nrow = nrow(.self$Ui) ,ncol = ncol(.self$Ui) )
                         
                         dWf = matrix(0,  nrow = nrow(.self$Wf) ,ncol = ncol(.self$Wf) )
                         dUf = matrix(0,  nrow = nrow(.self$Uf) ,ncol = ncol(.self$Uf) )
                         
                         dWc = matrix(0,  nrow = nrow(.self$Wc) ,ncol = ncol(.self$Wc) )
                         dUc = matrix(0,  nrow = nrow(.self$Uc) ,ncol = ncol(.self$Uc) )
                         
                         gferr = matrix(0,nrow(X)+1,.self$Dh)
                         gierr = matrix(0,nrow(X)+1,.self$Dh)
                         cierr = matrix(0,nrow(X)+1,.self$Dh)
                         goerr = matrix(0,nrow(X)+1,.self$Dh)
                         
                         stateerr  = matrix(0,nrow(X)+1,.self$Dh)
                         sourceerr = matrix(0,nrow(X)+1,.self$Dh)
                         Uerr = matrix(0,nrow(X)+1,.self$Dh)
                         PrevStateerr = matrix(0,nrow(X)+1,.self$Dh)
                         
                         
                         Y_pred = .self$forward(X)
                         err = ( Y - Y_pred )

                       #  prevstateerr = matrix(0,1,.self$Dh)
                         
                         costs= sum(err^2)*0.5
                         for(i in (nrow(X)+1):2){
                           delta =-err[i-1]
                      
                           delta = delta + Uerr[i,]
                           
                           goerr[i,]    = delta * .self$c_t[i,] * (.self$ao_t[i,]*(1- .self$ao_t[i,]))
                           stateerr[i,] = delta * .self$ao_t[i,] + PrevStateerr[i,]
                             
                           gferr[i,] = (.self$af_t[i,]*(1- .self$af_t[i,])) *stateerr[i,]*.self$c_t[i-1,]
                           gierr[i,] = (.self$ai_t[i,]*(1- .self$ai_t[i,]))*stateerr[i,]* .self$ac_t[i,]
                           cierr[i,] = (1-.self$ac_t[i,]^2) *stateerr[i,]* .self$ai_t[i,]
                           
                           sourceerr[i-1,] = gierr[i,] %*% .self$Wi
                           sourceerr[i-1,] = sourceerr[i-1,] + gferr[i,] %*% .self$Wf
                           sourceerr[i-1,] = sourceerr[i-1,] + goerr[i,] %*% .self$Wo
                           sourceerr[i-1,] = sourceerr[i-1,] + cierr[i,] %*% .self$Wc
                           
                           Uerr[i-1,] = gierr[i,] %*% .self$Ui
                           Uerr[i-1,] = Uerr[i-1,] + gferr[i,] %*% .self$Uf
                           Uerr[i-1,] = Uerr[i-1,] + goerr[i,] %*% .self$Uo
                           Uerr[i-1,] = Uerr[i-1,] + cierr[i,] %*% .self$Uc
                            
                           PrevStateerr[i-1,] = stateerr[i,] %*% .self$af_t[i,]
                             
                           dUi = dUi + matrix( gierr[i,] %o% .self$c_t[i-1,],nrow= nrow(dUi) , ncol= ncol(dUi) )
                           dWi = dWi + matrix( gierr[i,] %o% X[i-1,],nrow= nrow(dWi) , ncol= ncol(dWi) )
                           
                           dUo = dUo + matrix(goerr[i,] %o% .self$c_t[i-1,],nrow= nrow(dUo) , ncol= ncol(dUo) )
                           dWo = dWo + matrix(goerr[i,] %o% X[i-1,],nrow= nrow(dWo) , ncol= ncol(dWo) )
                         
                           dUf = dUf + matrix( gferr[i,] %o% .self$c_t[i-1,],nrow= nrow(dUf) , ncol= ncol(dUf) )
                           dWf = dWf + matrix( gferr[i,]%o% X[i-1,],nrow= nrow(dWf) , ncol= ncol(dWf) )
                          
                           dWc = dWc + matrix(cierr[i,] %o% X[i-1,],nrow= nrow(dWc) , ncol= ncol(dWc) )
                           dUc = dUc + matrix(cierr[i,] %o% .self$c_t[i-1,],nrow= nrow(dUc) , ncol= ncol(dUc) )
                           
                         }
                                      
                         ret =list(costs ,dWi,dUi,dWo,dUo,dWf,dUf,dWc,dUc)
                         ##clip gradient
                       #  THRESH = 0.1
                      #   idx = which(unlist(ret) [2:length(ret)] > THRESH )
                      #   if(length(idx)>0){
                      #     idx = idx +1
                      #     ret[idx] = (THRESH/abs(unlist(ret[idx])) )*unlist(ret[idx])
                      #     print('clip gradient')
                      #     print(idx)
                         #  browser()
                      
                         #   }
                        
                         
                         ret
                       },
                       train = function(X,Y,iterations = 1000,lr = 0.001){
                         ## init matricies
                         if(class(X)!='list'){
                           cat( c('X should be a of type list') )
                           return;
                         }
                         cat(c(length(X),' training sequences\n' ))
                         
                         #.self$init(ncol(X))
                         totalCosts = c()
                         for( i in 1:iterations){
                           # dU = matrix(0,ncol = ncol(.self$U),nrow = nrow(.self$U) )
                           #  dL = matrix(0,ncol = ncol(.self$L),nrow = nrow(.self$L) )
                           #  dH = matrix(0,ncol = ncol(.self$H),nrow = nrow(.self$H) )
                           costs = 0
                           for(l in 1:length(X)){##for all training seq
                             xl = X[[l]]
                             yl = Y[[l]]
                             
                             ret = gradient(xl,yl)
                           #  print(ret)
                             
                             cost = ret[[1]]
                             costs = costs + cost
                             
                             dWi= ret[[2]]
                             dUi= ret[[3]]
                             dWo= ret[[4]]
                             dUo= ret[[5]]
                             dWf= ret[[6]]
                             dUf= ret[[7]]
                             dWc= ret[[8]]
                             dUc= ret[[9]]
                             
                     #        .self$U=.self$U - (lr* dU)
                    #         .self$bu=.self$bu - (lr* dbu)
                             
                             .self$Wi=.self$Wi - (lr* dWi)
                             .self$Ui=.self$Ui - (lr* dUi)
                          #   .self$bi=.self$bi - (lr* dbi)
                             
                             .self$Wo=.self$Wo - (lr* dWo)
                             .self$Uo=.self$Uo - (lr* dUo)
                          #   .self$bo=.self$bo - (lr* dbo)
                             
                             .self$Wf=.self$Wf - (lr* dWf)
                             .self$Uf=.self$Uf - (lr* dUf)
                        #     .self$bf=.self$bf - (lr* dbf)
                             
                             .self$Wc=.self$Wc - (lr* dWc)
                             .self$Uc=.self$Uc - (lr* dUc)
                      #       .self$bc=.self$bc - (lr* dbc)
                             
                             #  print(dL)
                             
                           }
                           totalCosts = c(totalCosts, costs)
                           ##take one step
                           
                           cat(c('iteration:', i, ' costs:',costs,'\n'))
                         }
                         totalCosts
                       },
                       chkGradient= function(){
                         
                         epsilon = 0.00001
                         tol =     0.0000001 
                     #    X= matrix(c(10,1,5,7,4,2) )
                    #     Y= matrix(c(5,4,1,8,4,2) )
                         X= matrix(c(0.1,0.2) )
                         Y= matrix(c(0.5,0.6) )
                         
                         
                         res = gradient(X,Y)
                         
                        
                         Wil = .self$Wi
                         for(y in 1:ncol(Wil)){
                           for(x in 1:nrow(Wil)){
                             .self$Wi = Wil
                             .self$Wi[x,y] = Wil[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wi[x,y] = Wil[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = as.matrix( res[[2]] )
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Wil:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         .self$Wi = Wil
                         Uil = .self$Ui
                         for(y in 1:ncol(Uil)){
                           for(x in 1:nrow(Uil)){
                             .self$Ui = Uil
                             .self$Ui[x,y] = Uil[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Ui[x,y] = Uil[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = res[[3]]
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Uil:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         
                         .self$Ui = Uil
                         Wol = .self$Wo
                         for(y in 1:ncol(Wol)){
                           for(x in 1:nrow(Wol)){
                             .self$Wo = Wol
                             .self$Wo[x,y] = Wol[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wo[x,y] = Wol[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = res[[4]]
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Wol:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         .self$Wo = Wol
                         Uol = .self$Uo
                         for(y in 1:ncol(Uol)){
                           for(x in 1:nrow(Uol)){
                             .self$Uo = Uol
                             .self$Uo[x,y] = Uol[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Uo[x,y] = Uol[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = res[[5]]
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Uol:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         
                         ##check Wf
                         .self$Uo = Uol
                         Wfl = .self$Wf
                         for(y in 1:ncol(Wfl)){
                           for(x in 1:nrow(Wfl)){
                             .self$Wf = Wfl
                             .self$Wf[x,y] = Wfl[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wf[x,y] = Wfl[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradL = res[[6]]
                             
                             if(abs(gradL[x,y]-gradaprox)>tol ){
                               cat(c(x,'x',y,  ' Wfl:',gradL[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         .self$Wf = Wfl
                         Ufl = .self$Uf
                         for(y in 1:ncol(Ufl)){
                           for(x in 1:nrow(Ufl)){
                             .self$Uf = Ufl
                             .self$Uf[x,y] = Ufl[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Uf[x,y] = Ufl[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradL = res[[7]]
                             
                             if(abs(gradL[x,y]-gradaprox)>tol ){
                               cat(c(x,'x',y,  ' Ufl:',gradL[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         ##check Wf
                         .self$Uf = Ufl
                         Wcl = .self$Wc
                         for(y in 1:ncol(Wcl)){
                           for(x in 1:nrow(Wcl)){
                             .self$Wc = Wcl
                             .self$Wc[x,y] = Wcl[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wc[x,y] = Wcl[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradL = as.matrix(res[[8]])
                             
                             if(abs(gradL[x,y]-gradaprox)>tol ){
                               cat(c(x,'x',y,  ' Wcl:',gradL[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         .self$Wc = Wcl
                         
                         Ucl = .self$Uc
                         for(y in 1:ncol(Ucl)){
                           for(x in 1:nrow(Ucl)){
                             .self$Uc = Ucl
                             .self$Uc[x,y] = Ucl[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Uc[x,y] = Ucl[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradL = res[[9]]
                             .self
                             if(abs(gradL[x,y]-gradaprox)>tol ){
                               cat(c(x,'x',y,  ' Ufl:',gradL[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         
                         #   .self$Uc = Ucl
                         
                         
                       },
                    generate=function(n,init){
                      
                      h = matrix(0,.self$Dh,1) ##seed is zeros
                      hatc  =  matrix(0,.self$Dh,1) ##seed is zeros
                      
                      x = init
                      Y = c()
                      for(i in 1:n){
                        it = .self$Ui %*% h+ .self$Wi %*%x
                        ft = .self$Uf %*% h + .self$Wf %*%x
                        ot = .self$Uo %*% h + .self$Wo %*%x
                        ct = .self$Uc %*% h + .self$Wc %*%x
                        
                        ai = .self$sigmoid(it)
                        af = .self$sigmoid(ft)
                        ao = .self$sigmoid(ot)
                        ac = tanh(ct)
                        
                        hatc = af * hatc + ai*ac
                        h =  ao * hatc
                        
                        Y = c(Y,h)
                      }
                      Y
                      
                    },
                       forward=function(X){
                         ##for each row
                         Y= c()
                        # .self$h_t = matrix(0,nrow(X)+1,.self$Dh)
                         .self$c_t = matrix(0,nrow(X)+1,.self$Dh)
                         
                      #   .self$c_t[1,] = rnorm( .self$Dh,0,0.1)
                         
                         .self$ai_t = matrix(0,nrow(X)+1,.self$Dh)
                         .self$af_t = matrix(0,nrow(X)+1,.self$Dh)
                         .self$ao_t = matrix(0,nrow(X)+1,.self$Dh)
                         .self$ac_t = matrix(0,nrow(X)+1,.self$Dh)
                         
                         
                         for(i in 1:nrow(X)){
                           x = X[i,]
                           if(is.na(x)){
                             cat(c('NA encoutered\n') )
                             x = Y[length(Y)]
                           }
                           it = .self$Ui %*% .self$c_t[i,] + .self$Wi %*%x
                           ft = .self$Uf %*% .self$c_t[i,] + .self$Wf %*%x
                           ct = .self$Uc %*% .self$c_t[i,] + .self$Wc %*%x
                           ot = .self$Uo %*% .self$c_t[i,] + .self$Wo %*%x
                           
                           ao = .self$sigmoid(ot)
                           ai = .self$sigmoid(it)
                           af = .self$sigmoid(ft)

                           ac = tanh(ct)
                           
                           .self$ai_t[i+1,] = ai
                           .self$af_t[i+1,] = af
                           .self$ao_t[i+1,] = ao
                           .self$ac_t[i+1,] = ac
                           
                           hatc = af * .self$c_t[i,] + ai*ac
                           .self$c_t[i+1,] = hatc
                           
                           z2 = ao * hatc
                           
                           if(is.na(z2) ){
                             browser()
                           }
                           Y = c(Y,z2)
                         }
                       
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
mem = lstm(1)
set.seed(1234) 
mem$init(1)
#mem$forward(matrix(1))

mem$chkGradient()

#costs = mem$train(list(as.matrix(x)) ,list(as.matrix(y)) ,iterations = 1000,lr = 0.0001)
#plot(1:length(costs), costs)
#y_pred = mem$forward(as.matrix(x))
#plot(x,y)
#points(x,y_pred,col='red')

#gen = mem$generate(10,2)
#plot(1:length(gen),gen)





