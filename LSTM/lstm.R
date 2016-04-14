########### Clear environment
rm(list = ls())

##create data...

options(error = recover)

x = seq(0,10,1)
y = 0.5 *x +1.2

lstm <- setRefClass("lstm",
                     fields = list(U  = 'matrix',bu ='matrix',
                                   Wi = 'matrix',Ui ='matrix',bi ='matrix',
                                   Wf = 'matrix',Uf ='matrix',bf ='matrix',
                                   Wo = 'matrix',Uo ='matrix',bo ='matrix',
                                   Wc = 'matrix',Uc ='matrix',bc ='matrix',
                                   c_t= 'matrix',h_t='matrix',
                                   ai_t = 'matrix',
                                   af_t = 'matrix',
                                   ao_t = 'matrix',
                                   ac_t ='matrix',
                                   Dh = 'numeric' ),
                     methods = list(
                       init=function(dimx){
                         .self$U = matrix(rnorm(.self$Dh* dimx,0,0.01),dimx,.self$Dh)
                         .self$bu = matrix(rnorm(.self$Dh* 1,0,0.01),1,dimx)
                         
                         .self$Ui = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uf = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uo = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         .self$Uc = matrix(rnorm(.self$Dh*.self$Dh ,0,0.01),.self$Dh, .self$Dh)
                         
                         .self$Wi = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wf = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wo = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         .self$Wc = matrix(rnorm(.self$Dh* dimx ,0,0.01),.self$Dh, dimx)
                         
                         .self$bi = matrix(rnorm(.self$Dh* .self$Dh,0,0.01),.self$Dh,dimx)
                         .self$bf = matrix(rnorm(.self$Dh* .self$Dh,0,0.01),.self$Dh,dimx)
                         .self$bo = matrix(rnorm(.self$Dh* .self$Dh,0,0.01),.self$Dh,dimx)
                         .self$bc = matrix(rnorm(.self$Dh* .self$Dh,0,0.01),.self$Dh,dimx)
                         
                       },
                       gradient = function(X,Y,backwards =10){
                         dU = matrix(0,  nrow = nrow(.self$U) ,ncol = ncol(.self$U) )
                         dbu = matrix(0,  nrow = nrow(.self$bu) ,ncol = ncol(.self$bu) )
                         
                         dWo = matrix(0,  nrow = nrow(.self$Wo) ,ncol = ncol(.self$Wo) )
                         dUo = matrix(0,  nrow = nrow(.self$Uo) ,ncol = ncol(.self$Uo) )
                         dbo = matrix(0,  nrow = nrow(.self$bo) ,ncol = ncol(.self$bo) )

                         dWi = matrix(0,  nrow = nrow(.self$Wi) ,ncol = ncol(.self$Wi) )
                         dUi = matrix(0,  nrow = nrow(.self$Ui) ,ncol = ncol(.self$Ui) )
                         dbi = matrix(0,  nrow = nrow(.self$bi) ,ncol = ncol(.self$bi) )
                         
                         dWf = matrix(0,  nrow = nrow(.self$Wf) ,ncol = ncol(.self$Wf) )
                         dUf = matrix(0,  nrow = nrow(.self$Uf) ,ncol = ncol(.self$Uf) )
                         dbf = matrix(0,  nrow = nrow(.self$bf) ,ncol = ncol(.self$bf) )
                         
                         dWc = matrix(0,  nrow = nrow(.self$Wc) ,ncol = ncol(.self$Wc) )
                         dUc = matrix(0,  nrow = nrow(.self$Uc) ,ncol = ncol(.self$Uc) )
                         dbc = matrix(0,  nrow = nrow(.self$bc) ,ncol = ncol(.self$bc) )
                         
                         
                         Y_pred = .self$forward(X)
                         err = ( Y - Y_pred )

                         costs= sum(err^2)*0.5
                         for(i in (nrow(X)+1):2){

                           delta =err[i-1] %*%-.self$U  #* (1- .self$h_t[i,]^2) 
                           dU = dU + (err[i-1] %o% -.self$h_t[i,]) ##check this %o%
                           dbu = dbu + (-err[i-1]) ##
                           
                           
                           deltao = delta 
                           deltai = delta 
                           deltaf = delta 
                           deltac = delta 
                           
                           for(s in 1:backwards){
                             ts = i - s
                             
                             if(ts <= 0){ ## make this better
                               break
                             }
                             #  print('timestep')
                             # print(ts)
                             
                             deltao =  (deltao * tanh(.self$c_t[ts+1,]) *(1- .self$ao_t[ts+1,]^2) )  
                             deltai =  (deltai * tanh(.self$ao_t[ts+1,]) *(1- .self$c_t[ts+1,]^2) *tanh(.self$ac_t[ts+1,]) *(1- .self$ai_t[ts+1,]^2) )  
                             deltaf =  (deltaf * tanh(.self$ao_t[ts+1,])*(1- .self$c_t[ts+1,]^2) * .self$c_t[ts,]*(1- .self$af_t[ts+1,])  )
                             deltac =  (deltac * tanh(.self$ao_t[ts+1,]) *(1-.self$c_t[ts+1,]^2)*tanh(.self$ai_t[ts+1,]) * (1- .self$ac_t[ts+1,]^2) )
                           
                             dUi = dUi + matrix(deltai %o% .self$h_t[ts,],nrow= nrow(dUi) , ncol= ncol(dUi) )
                             dWi = dWi + matrix(deltai %o% X[ts,],nrow= nrow(dWi) , ncol= ncol(dWi) )
                             dbi = dbi + t(deltai)
                          #   print(dWi)
                           #  deltai = deltai %*% .self$Wi
                             
                             dUo = dUo + matrix(deltao %o% .self$h_t[ts,],nrow= nrow(dUo) , ncol= ncol(dUo) )
                             dWo = dWo + matrix(deltao %o% X[ts,],nrow= nrow(dWo) , ncol= ncol(dWo) )
                             dbo = dbo + t(deltao)
                             
                             dUf = dUf + matrix(deltaf %o% .self$h_t[ts,],nrow= nrow(dUf) , ncol= ncol(dUf) )
                             dWf = dWf + matrix(deltaf %o% X[ts,],nrow= nrow(dWf) , ncol= ncol(dWf) )
                             dbf = dbf + t(deltaf)
                             
                             dUc = dUc + matrix(deltac %o% .self$h_t[ts,],nrow= nrow(dUc) , ncol= ncol(dUc) )
                             dWc = dWc + matrix(deltac %o% X[ts,],nrow= nrow(dWc) , ncol= ncol(dWc) )
                             dbc = dbc + t(deltac)

                            #print(dWi)
                            #cat(c('inner:',ts,'\n') )
                            # dH = dH + matrix(matrix(delta) %o% .self$h_t[ts,],nrow= .self$Dh, ncol= .self$Dh)
                            # dL = dL + matrix( matrix(delta) %o% X[ts,],nrow =.self$Dh, ncol = ncol(X)  )
                             # Update delta
                          #   delta =  delta  %*% .self$H  * (.self$h_t[ts,] * (1-.self$h_t[ts,]))
                           }
                         }
                        
                         
                         
                         
                    #     if(is.na(costs) | any( dWo >100)){
                     #      browser()
                    #     }
                         #ret =list(costs ,dU,dbu,dWi,dUi,dWo,dUo,dWf,dUf,dWc,dUc,dbi,dbo,dbf,dbc)
                         ret =list(costs ,dU,dbu,dWi,dUi,dWo,dUo,dWf,dUf,dWc,dUc)
                         ##clip gradient
                         THRESH = 0.1
                         idx = which(unlist(ret) [2:length(ret)] > THRESH )
                         if(length(idx)>0){
                           idx = idx +1
                           ret[idx] = (THRESH/abs(unlist(ret[idx])) )*unlist(ret[idx])
                           print('clip gradient')
                           print(idx)
                        #   browser()
                         }
                         idx = which(abs(unlist(ret) [2:length(ret)]) < 0.000000000000001 )
                         if(length(idx)>0){
                           idx = idx +1
                           ret[idx] = 0
                           print('set gradient to zero')
                           print(idx)
                         }
                         
                         ret
                       },
                       train = function(X,Y,backwards =3,iterations = 1000,lr = 0.001){
                         ## init matricies
                         if(class(X)!='list'){
                           cat( c('X should be a of type list') )
                           return;
                         }
                         cat(c(length(X),' training sequences\n' ))
                         if(class(X[[1]])=='matrix'){
                           if(ncol(X[[1]]) == nrow(.self$U)){
                             cat( c('With dimension:',nrow(.self$U) , 'and ',ncol(.self$U),'hidden units\n') )
                           }else{
                             cat( c('Network not correctly initalizied, i will do this for u with dimension:',ncol(X[[1]]),'\n') )
                             .self$init(ncol(X[[1]]))
                           }
                         }else{
                           cat( c('the elements of this list should be type matrix\n') )
                           return;
                         }
                         
                         
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
                             
                             ret = gradient(xl,yl,backwards)
                           #  print(ret)
                             
                             cost = ret[[1]]
                             costs = costs + cost
                             
                             dU = ret[[2]]
                             dbu= ret[[3]]
                             dWi= ret[[4]]
                             dUi= ret[[5]]
                             dWo= ret[[6]]
                             dUo= ret[[7]]
                             dWf= ret[[8]]
                             dUf= ret[[9]]
                             dWc= ret[[10]]
                             dUc= ret[[11]]
                          #   dbi= ret[[12]]
                          #   dbo= ret[[13]]
                          #   dbf= ret[[14]]
                          #   dbc= ret[[15]]
                             
                             
                             .self$U=.self$U - (lr* dU)
                             .self$bu=.self$bu - (lr* dbu)
                             
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
                       generate=function(n,init){
           
                         h = matrix(0,.self$Dh,1) ##seed is zeros
                         hatc  =  matrix(0,.self$Dh,1) ##seed is zeros
                         #hatc  =  matrix(rnorm(.self$Dh,0,0.01),.self$Dh,1) ##seed is zeros
                         
                         x = init
                         Y = c()
                         for(i in 1:n){
                           it = .self$Ui %*% h+ .self$Wi %*%x
                           ft = .self$Uf %*% h + .self$Wf %*%x
                           ot = .self$Uo %*% h + .self$Wo %*%x
                           ct = .self$Uc %*% h + .self$Wc %*%x
                           
                           ai = tanh(it)
                           af = tanh(ft)
                           ao = tanh(ot)
                           ac = tanh(ct)
                           
                           
                           hatc = af * hatc + ai*ac
                           h =  ao * tanh(hatc)
                           x = .self$U %*% h  + .self$bu
                           Y = c(Y,x)
                         }
                         Y
                         
                       },
                       chkGradient= function(){
                         
                         epsilon = 0.00001
                         tol =     0.000000001 
                         X= matrix(c(10,1,5,7,4,2) )
                         Y= matrix(c(5,4,1,8,4,2) )
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
                         
                         ##check bu
                         .self$U = Ul
                         bul = .self$bu
                         for(y in 1:ncol(bul)){
                           for(x in 1:nrow(bul)){
                             .self$bu = bul
                             .self$bu[x,y] = bul[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$bu[x,y] = bul[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = res[[3]]
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' bu:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         .self$bu = bul
                         Wil = .self$Wi
                         for(y in 1:ncol(Wil)){
                           for(x in 1:nrow(Wil)){
                             .self$Wi = Wil
                             .self$Wi[x,y] = Wil[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wi[x,y] = Wil[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradU = res[[4]]
                             
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
                             gradU = res[[5]]
                             
                             #if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Uil:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             #}
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
                             gradU = res[[6]]
                             
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
                             gradU = res[[7]]
                             
                             if(abs(gradU[x,y]-gradaprox)>tol  ){
                               cat(c(x,'x',y,  ' Uol:',gradU[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         
                         ##check Wf
                         .self$bu = bul
                         Wfl = .self$Wf
                         for(y in 1:ncol(Wfl)){
                           for(x in 1:nrow(Wfl)){
                             .self$Wf = Wfl
                             .self$Wf[x,y] = Wfl[x,y]+epsilon
                             res1 = gradient(X,Y)
                             .self$Wf[x,y] = Wfl[x,y]-epsilon
                             res2 = gradient(X,Y)
                             
                             gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                             gradL = res[[8]]
                             
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
                             gradL = res[[9]]
                             
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
                             gradL = res[[10]]
                             
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
                             gradL = res[[11]]
                             .self
                             if(abs(gradL[x,y]-gradaprox)>tol ){
                               cat(c(x,'x',y,  ' Ufl:',gradL[x,y],' approximation:',gradaprox,'\n') )
                             }
                           }
                         }
                         
                         
                      #   .self$Uc = Ucl
                      #   bil = .self$bi
                      #   for(y in 1:ncol(bil)){
                      #     for(x in 1:nrow(bil)){
                      #       .self$bi = bil
                      #       .self$bi[x,y] = bil[x,y]+epsilon
                      #       res1 = gradient(X,Y)
                      #       .self$bi[x,y] = bil[x,y]-epsilon
                      #       res2 = gradient(X,Y)
                             
                      #       gradaprox  =  (res1[[1]] - res2[[1]])/(2*epsilon)
                      #       gradL = res[[12]]
                             
                      #       if(abs(gradL[x,y]-gradaprox)>tol ){
                      #         cat(c(x,'x',y,  ' bi:',gradL[x,y],' approximation:',gradaprox,'\n') )
                      #       }
                      #     }
                      #   }
                         
                       },
                       forward=function(X){
                         ##for each row
                         Y= c()
                         .self$h_t = matrix(0,nrow(X)+1,.self$Dh)
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
                           it = .self$Ui %*% .self$h_t[i,] + .self$Wi %*%x #+ .self$bi
                           ft = .self$Uf %*% .self$h_t[i,] + .self$Wf %*%x #+ .self$bf
                           ot = .self$Uo %*% .self$h_t[i,] + .self$Wo %*%x #+ .self$bo
                           ct = .self$Uc %*% .self$h_t[i,] + .self$Wc %*%x #+ .self$bc
                           
                           ai = tanh(it)
                           af = tanh(ft)
                           ao = tanh(ot)
                           ac = tanh(ct)
                           .self$ai_t[i+1,] = it
                           .self$af_t[i+1,] = ft
                           .self$ao_t[i+1,] = ot
                           .self$ac_t[i+1,] = ct
                           
                           
                           hatc = af * .self$c_t[i,] + ai*ac
                           .self$c_t[i+1,] = hatc
                           
                             
                           h = ao * tanh(hatc)
                           .self$h_t[i+1,] = h
                           
                           z2 = .self$U %*% h + .self$bu
                           
                           if(is.na(z2) ){
                             browser()
                           }
                           Y = c(Y,z2)
                         }
                         
                         if( any(.self$h_t[2:nrow(.self$h_t),]>100) ){
                           browser()
                         }
                         
                         Y
                       },
                       sigmoid = function(x){
                         x = 1/(1+exp(x))
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

costs = mem$train(list(as.matrix(x)) ,list(as.matrix(y)) ,iterations = 100,lr = 0.001,backwards=1)
plot(1:length(costs), costs)
y_pred = mem$forward(as.matrix(x))
plot(x,y)
points(x,y_pred,col='red')

##
#costs = mem$train(allSeq$x ,allSeq$y,iterations = 100,lr = 0.0001,backwards = 1)

#ts = allMoodTs[1,]
#ts = as.matrix(ts)
#y = mem$forward(ts)
#plot(1:length(y),y,ylim=c(0,10))
#points(1:length(ts),ts,col='red')



