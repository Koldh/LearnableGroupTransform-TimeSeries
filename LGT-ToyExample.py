from pylab import *
import cPickle
import theano as th
import lasagne
import sys
import scipy.io
from scipy.io import wavfile
import numpy as np
import glob




class LGT_toy:
        def __init__(self,x_shape,N,J,Q,num_knots=2**8,t1=-1.,t2=1.,sigma=0.2,log_=1,init_='classic'):
                x             = th.tensor.fmatrix('x')
                y             = th.tensor.ivector('y')
                filter_size   = int(2**9)
                num_filters   = int(J*Q)
                stride        = 1
                pad           = 'valid'
                #WARPING TIME
                if init_ == 'dyadic':
                    W_firstlayer         = th.shared(ones((1,int(J*Q)*num_knots)).astype('float32'))
                    W_positiv_firstlayer = th.tensor.abs_(W_firstlayer)
                    b                    = th.shared(ones((int(J*Q)*num_knots)).astype('float32'))
                    w__                  = th.shared(asarray([2**(-float(j)/Q) for j in range(J*Q)]).astype('float32'))
                    w__positiv  = th.tensor.abs_(w__)
                    fill        = th.shared(zeros((J*Q,num_knots-1)).astype('float32'))
                    W__         = th.tensor.concatenate([w__positiv.reshape((-1,1)),fill],axis=1)
                    W_positiv   = th.tensor.abs_(W__)
                    W__         = W_positiv.dimshuffle('x',0,1)
                    b_          = th.shared(zeros(int(J*Q)).astype('float32'))


                elif init_ == 'random':
                    b            = th.shared(zeros(int(J*Q)*num_knots).astype('float32'))
                    W_firstlayer = th.shared(randn(1,int(J*Q)*num_knots).astype('float32')*(sqrt(6.)/sqrt(0.5*(1+int(J*Q)*num_knots))))
                    W_positiv_firstlayer = th.tensor.abs_(W_firstlayer)
                    W_ =  th.shared(randn(int(J*Q),num_knots).astype('float32')*(sqrt(6.)/sqrt(num_knots+int(J*Q))))
                    W_positiv = th.tensor.abs_(W_)
                    W__ =  W_positiv.dimshuffle('x',0,1)
                    b_ =  th.shared(zeros(int(J*Q)).astype('float32'))
                elif init_ == 'classic':
                    b            = th.shared(randn(int(J*Q)*num_knots).astype('float32'))
                    W_firstlayer = th.shared(randn(1,int(J*Q)*num_knots).astype('float32')*(sqrt(6.)/sqrt(0.5*(1+int(J*Q)*num_knots))))
                    W_ =  th.shared(randn(int(J*Q),num_knots).astype('float32')*(sqrt(6.)/sqrt(num_knots+int(J*Q))))
                    W__ =  W_.dimshuffle('x',0,1)
                    b_ =  th.shared(randn(int(J*Q)).astype('float32'))



                time              = th.shared(linspace(t1,t2,filter_size).astype('float32').reshape((-1,1)))
                layers_time       = [lasagne.layers.InputLayer((filter_size,1),time)]
                layers_time.append(lasagne.layers.ReshapeLayer(layers_time[-1],(filter_size,1)))
                layers_time.append(((lasagne.layers.DenseLayer(layers_time[-1], num_knots*int(J*Q),W=W_firstlayer,b=b,nonlinearity=lasagne.nonlinearities.leaky_rectify))))
                layers_time.append(lasagne.layers.ReshapeLayer(layers_time[-1],(filter_size,int(J*Q),num_knots)))
                output_time = ((lasagne.layers.get_output(layers_time[-1])*W__).sum(axis=-1)+b_).T

                if init_ == 'dyadic':
                    params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)+[w__,fill,b_,W_firstlayer,b]
                elif init_ == 'random':
                    params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)+[W_,b_,W_firstlayer,b]
                elif init_ == 'classic':
                    params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)+[W_,b_,W_firstlayer,b]
                output_time = th.tensor.sort(output_time,axis=1)
                time_warped       = output_time-output_time.mean(axis=1,keepdims=True)
                self.get_warped = th.function([],time_warped)


                f          = (filter_size)/2.
                psi_real   =  th.tensor.cast(pi**(-1/4.)*th.tensor.cos((2*pi*f*(time_warped)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')
                psi_imag   = th.tensor.cast(pi**(-1/4.)*th.tensor.sin((2*pi*f*(time_warped)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')
                self.get_filters = th.function([],[psi_real,psi_imag])


                layers_warp       = [lasagne.layers.InputLayer(x_shape,x)]
                layers_warp.append(lasagne.layers.ReshapeLayer(layers_warp[-1],(x_shape[0],1,x_shape[1])))
                real_layer        = lasagne.layers.Conv1DLayer(layers_warp[-1],num_filters=num_filters,filter_size=filter_size,W=psi_real.dimshuffle(0,'x',1),stride=int(stride),pad=pad,nonlinearity=None,b=None)
                imag_layer        = lasagne.layers.Conv1DLayer(layers_warp[-1],num_filters=num_filters,filter_size=filter_size,W=psi_imag.dimshuffle(0,'x',1),stride=int(stride),pad=pad,nonlinearity=None,b=None)
                warp              = th.tensor.sqrt(th.tensor.pow(lasagne.layers.get_output(real_layer),2)+th.tensor.pow(lasagne.layers.get_output(imag_layer),2))
                self.get_represent = th.function([x],warp)
                warp_shape        = (x_shape[0],int(J*Q),x_shape[1]-filter_size+1)



                layers            = [lasagne.layers.InputLayer(warp_shape,warp)]
                shape             = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])))
                layers.append(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]))
                layers.append(lasagne.layers.DenseLayer(layers[-1],1,nonlinearity=lasagne.nonlinearities.sigmoid))


                output      = lasagne.layers.get_output(layers[-1])
                output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
                loss = lasagne.objectives.binary_crossentropy(output,y).mean()
                accu_train = lasagne.objectives.binary_accuracy(output,y).mean()
                accu = lasagne.objectives.binary_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
                learning_rate = th.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params+params_time,learning_rate)
                updates_time  = lasagne.updates.adam(loss,params_time,learning_rate)
                self.predict  = th.function([x],output_test)
                self.train    = th.function([x,y,learning_rate],loss,updates=updates)
                self.train_time    = th.function([x,y,learning_rate],loss,updates=updates_time)
                self.test     = th.function([x,y],accu)
                self.acc_train = th.function([x,y],accu_train)





list_file = glob.glob('./toyexample_filter_c_1e-07noise_0.7.pkl')
for name in list_file:
    if 'lr_' not in name:
        f    = open(name,'rb')
        data = cPickle.load(f)
        f.close()
        print name
        fil_as, fil_des = data
        X  = asarray(concatenate([fil_as,fil_des]))
        labels = concatenate([ones(100),zeros(100)])

        p = permutation(200)
        data_p = real(X[p]).astype('float32')
        labels_p = labels[p].astype('int32')

        data_train  = data_p[:100]
        data_test   = data_p[100:]
        label_train = labels_p[:100]
        label_test  = labels_p[100:]
        epoch  =50
        for run in xrange(1):
            for l_r in [0.01]:
                for batch_size in [10]:
                    x_shape = (batch_size,2**13)
                    nb_batchs = 100/batch_size
                    MODEL = LGT_toy(x_shape,1,8,8,num_knots=2**8,t1=-1.,t2=1.,sigma=.02,log_=1)
                    loss = []
                    accu_mean = []
                    accu_test_mean = []
                    repres   = []
                    filters = []
                    warp = []
                    for e in xrange(epoch):
                        p= permutation(100)
                        data_train = data_train[p]
                        label_train = label_train[p]
                        accu_train = []
                        accu_test  = []
                        if e==0 or e==49:
                            repres.append(MODEL.get_represent(data_test[0:batch_size]))
                            filters.append(MODEL.get_filters())
                            warp.append(MODEL.get_warped())
                        for k in xrange(nb_batchs):
                            loss.append(MODEL.train(data_train[k*batch_size:(k+1)*batch_size],label_train[k*batch_size:(k+1)*batch_size],l_r))
                            print loss[-1]
                        for c in xrange(nb_batchs):    
                            accu_train.append(MODEL.acc_train(data_train[c*batch_size:(c+1)*batch_size],label_train[c*batch_size:(c+1)*batch_size]))
                            accu_test.append(MODEL.test(data_test[c*batch_size:(c+1)*batch_size],label_test[c*batch_size:(c+1)*batch_size]))
                        print ' ' 
                        print 'ACCU TRAIN: ', mean(accu_train)
                        print 'ACCU TEST:  ',mean(accu_test)
                        print ' '
                        accu_mean.append(mean(accu_train))
                        accu_test_mean.append(mean(accu_test))
                    f =  open(name+'sortforplotwarping_filters64_RUN'+str(run)+'lr_'+str(l_r)+'.pkl','wb')
                    cPickle.dump([loss,accu_mean,accu_test_mean,repres,filters,warp,label_test[0:batch_size]],f)
                    f.close()


