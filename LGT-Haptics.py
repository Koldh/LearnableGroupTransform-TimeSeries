from pylab import *
import theano as th
import lasagne
import sys
import scipy.io
from scipy.io import wavfile



class LGT_haptics:
        def __init__(self,x_shape,N,J,Q,num_knots=2**8,t1=-1.,t2=1.,sigma=0.02,log_=1,init_='random',nonlin='sqrt',grad=False):
                x             = th.tensor.fmatrix('x')
                y             = th.tensor.imatrix('y')
                filter_size   = int(128)
                num_filters   = int(J*Q)
                stride        = 1
                pad           = 'valid'
                #WARPING TIME
                time              = th.shared(linspace(t1,t2,filter_size).astype('float32').reshape((-1,1)))
                if init_ == 'random':
                        layers_time       = [lasagne.layers.InputLayer((filter_size,1),time)]
                        layers_time.append(lasagne.layers.ReshapeLayer(layers_time[-1],(filter_size,1,1)))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1], num_knots,nonlinearity=lasagne.nonlinearities.leaky_rectify,b=randn(num_knots).astype('float32')/10.)))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1],num_filters, nonlinearity=None)))
                        params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)
                        time_warped       = th.tensor.sort(lasagne.layers.get_output(layers_time[-1]),axis=0)
                        time_warped -= time_warped.mean(axis=0)

                elif init_ == 'dyadic':
                        print 'dyadic'
                        W_ = th.shared(asarray([2**(-j/Q) for j in xrange(J*Q)]).astype('float32').reshape((-1,1)))
                        fill = th.shared(zeros((J*Q,num_knots-1)).astype('float32'))
                        W__ = th.tensor.concatenate([W_,fill],axis=1).transpose()
                        layers_time       = [lasagne.layers.InputLayer((filter_size,1),time)]
                        layers_time.append(lasagne.layers.ReshapeLayer(layers_time[-1],(filter_size,1,1)))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1],num_knots,W=th.shared(ones((num_knots)).astype('float32').reshape((1,-1))),nonlinearity=lasagne.nonlinearities.leaky_rectify,b=ones(num_knots).astype('float32'))))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1],num_filters,W=W__,nonlinearity=None)))
                        params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)+[W_,fill]
                        time_warped       = th.tensor.sort(lasagne.layers.get_output(layers_time[-1]),axis=0)
                        time_warped -= time_warped.mean(axis=0)

                elif init_ == 'constrained':
                        layers_time       = [lasagne.layers.InputLayer((filter_size,1),time)]
                        layers_time.append(lasagne.layers.ReshapeLayer(layers_time[-1],(filter_size,1,1)))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1], num_knots,nonlinearity=lasagne.nonlinearities.leaky_rectify,b=randn(num_knots).astype('float32')/10.)))
                        layers_time.append((lasagne.layers.DenseLayer(layers_time[-1],num_filters, nonlinearity=None)))
                        params_time       = lasagne.layers.get_all_params(layers_time[-1],trainable=True)
                        alpha         = th.shared(asarray([2**(-float(j)/Q) for j in range(J*Q)]).astype('float32'))
                        time_warped       = th.tensor.sort(lasagne.layers.get_output(layers_time[-1]),axis=0)
                        time_warped  -= th.tensor.min(time_warped,axis=0)
                        time_warped /= th.tensor.max(time_warped,axis=0)
                        alpha_positiv = th.tensor.abs_(alpha)
                        time_warped  *= 2.
                        time_warped  -= 1.
                        time_warped  *= alpha_positiv.dimshuffle('x',0)



                grad_ = th.tensor.max((time_warped[10:,:]-time_warped[:-10,:])/(th.shared(linspace(-1.,1.,filter_size)[10:])-th.shared(linspace(-1.,1.,filter_size)[:-10])).dimshuffle(0,'x'),axis=0)
                frequency_sorting = grad_.argsort()

                f          = (filter_size)/2.
                #WARPING FILTER
                if grad == False:
                    psi_real   =  th.tensor.cast(pi**(-1/4.)*th.tensor.cos((2*pi*f*(time_warped)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')
                    psi_imag   = th.tensor.cast(pi**(-1/4.)*th.tensor.sin((2*pi*f*(time_warped)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')
                elif grad == True:
                    psi_real   =  th.tensor.cast(pi**(-1/4.)*th.tensor.cos((2*pi*f*(time_warped/grad_)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')
                    psi_imag   = th.tensor.cast(pi**(-1/4.)*th.tensor.sin((2*pi*f*(time_warped/grad_)*.44))*th.tensor.exp((-1/2.)*(time_warped/sigma)**2),'float32')


                psi_real = th.tensor.transpose(psi_real)
                psi_imag = th.tensor.transpose(psi_imag)
                psi_real = psi_real[frequency_sorting]
                psi_imag = psi_imag[frequency_sorting]
                psi_real_argm = psi_real.argmax(axis=1)
                self.get_filter = th.function([],[psi_real,psi_imag])
                self.get_time = th.function([],th.tensor.transpose(time_warped))
                layers_warp       = [lasagne.layers.InputLayer(x_shape,x)]
                layers_warp.append(lasagne.layers.ReshapeLayer(layers_warp[-1],(x_shape[0],1,x_shape[1])))

                real_layer        = lasagne.layers.Conv1DLayer(layers_warp[-1],num_filters=num_filters,filter_size=filter_size,W=psi_real.dimshuffle(0,'x',1),stride=int(stride),pad=pad,nonlinearity=None,b=None)
                imag_layer        = lasagne.layers.Conv1DLayer(layers_warp[-1],num_filters=num_filters,filter_size=filter_size,W=psi_imag.dimshuffle(0,'x',1),stride=int(stride),pad=pad,nonlinearity=None,b=None)
                warp              = th.tensor.sqrt(th.tensor.pow(lasagne.layers.get_output(real_layer),2)+th.tensor.pow(lasagne.layers.get_output(imag_layer),2)+0.001)
                self.get_represent = th.function([x],warp)
                warp_shape        = (x_shape[0],int(J*Q),x_shape[1]-filter_size+1)
               
                layers            = [lasagne.layers.InputLayer(warp_shape,warp)]
                shape             = lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.ReshapeLayer(layers[-1],(shape[0],1,shape[1],shape[2])))
                shape2= lasagne.layers.get_output_shape(layers[-1])
                layers.append(lasagne.layers.Pool2DLayer(layers[-1],stride=(1,2**6),pool_size=(1,2**7),mode='average_exc_pad'))
                layers.append(lasagne.layers.BatchNormLayer(layers[-1],axes=[0,1,3]))
                layers.append(lasagne.layers.DenseLayer(layers[-1],5,nonlinearity=lasagne.nonlinearities.softmax))
                output      = lasagne.layers.get_output(layers[-1])
                output_test = lasagne.layers.get_output(layers[-1],deterministic=True)
                loss = lasagne.objectives.categorical_crossentropy(output,y).mean()
                accu_train = lasagne.objectives.categorical_accuracy(output,y).mean()
                accu = lasagne.objectives.categorical_accuracy(output_test,y).mean()
                print("NUMBER OF PARAMS",lasagne.layers.count_params(layers[-1]))
                params        = lasagne.layers.get_all_params(layers[-1],trainable=True)
                learning_rate = th.tensor.scalar()
                updates       = lasagne.updates.adam(loss,params,learning_rate)
                updates_time  = lasagne.updates.adam(loss,params_time,learning_rate)
                self.predict  = th.function([x],output_test)
                self.train    = th.function([x,y,learning_rate],loss,updates=updates)
                self.train_time    = th.function([x,y,learning_rate],loss,updates=updates_time)
                self.test     = th.function([x,y],accu)
                self.acc_train = th.function([x,y],accu_train)

