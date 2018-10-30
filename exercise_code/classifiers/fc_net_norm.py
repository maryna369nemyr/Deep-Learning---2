import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        
        self.classes = range(num_classes)

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################   
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim) # +0.0        
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward 
    
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        X_new  = np.reshape(X, (X.shape[0], -1))
        N, _ = X_new.shape
        
        aff1 =X_new.dot(W1) + b1.T 
        hl= np.multiply(aff1, aff1>np.zeros_like(aff1))
        
        aff2 = hl.dot(W2) + b2.T 
        scores = aff2       
        
        temp = np.exp(aff2)
        den = np.sum(temp, axis =1)  
        rl = np.transpose(np.divide(np.transpose(temp),den))
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
            
        c = self.classes
        y_enc = []
        for i in range(len(y)):
            for el in c:
                #print(el, y[i])
                zero = np.zeros_like(c)
                if(el == y[i]):
                    zero[el]=1
                    break;
            y_enc.append(zero)
        y_hot_enc =  np.array(y_enc)  
        


        loss = np.trace(np.matmul(np.transpose(y_hot_enc), (np.log(rl)))) 
        loss = (-1.0/N)* loss
        loss = loss + 0.5* self.reg*  ( np.sum(np.power(W1,2)) + np.sum(np.power(W2,2)))
        
        #grads
        grad_y_pred = 1/N * (rl - y_hot_enc)
        
        
        grad_w2 = np.dot(hl.T, grad_y_pred) + self.reg*W2 
        grad_b2 = (1/N)*(np.matmul(np.ones_like(rl).T,(rl - y_hot_enc)))
               
        prod = np.dot(grad_y_pred,W2.T)
        grad_h = prod
       
        
        my_zeros = np.zeros_like(hl)       
        grad_relu = my_zeros
        grad_relu[np.where(hl>my_zeros)] = prod[np.where(hl>my_zeros)]  
       
        grad_w1= np.matmul(X_new.T, grad_relu) + self.reg*W1 
        grad_b1 = np.matmul(np.ones_like(grad_relu).T, grad_relu) 
        
        grads['W1'] = grad_w1
        grads['W2'] = grad_w2
        grads['b1'] = grad_b1[0]
        grads['b2'] = grad_b2[0]
        
        #print(grad_b1[0]) 
        #print(grad_b2[0])
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet_mod(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.bn_params = {} # scale and shift parameter
        
        self.classes = range(num_classes)

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        for i in range(self.num_layers):
            w = 'W' + str(i+1)
            b = 'b'+ str (i+1)
            gamma = 'gamma' + str(i+1) # scale parameter
            beta = 'beta' + str(i+1) # shift parameter
            self.bn_params[gamma] = 1.0
            self.bn_params[beta] = 0.0
            if(i==0):
                self.params[w] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
                self.params[b] = np.zeros( hidden_dims[i])
                continue;
            if(i==(self.num_layers-1)):
                self.params[w] = weight_scale * np.random.randn(hidden_dims[i-1],num_classes)
                self.params[b] = np.zeros(num_classes)
                continue;                
            self.params[w] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
            self.params[b] = np.zeros(hidden_dims[i])

   
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    
    def affine_batchnorm_relu_forward(x,w,b):
            a, fc_cache = affine_forward(x, w, b)
            outbn, bn_cache = batchnorm_forward(a) # a = dx,dw,db = bn_cache
            out, relu_cache = relu_forward(outbn)
            cache = (fc_cache, bn_cache, relu_cache)
            return out, cache
        
    def affine_relu_backward(dout, cache):
            fc_cache, bn_cache, relu_cache = cache
            doutbn = relu_backward(dout, relu_cache)
            da = batchnorm_backward(doutbn, bn_cache)
            dx, dw, db = affine_backward(da, fc_cache)
            return dx, dw, db
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        #print(X.shape, len(self.params))
        affines = [] # XW1 + b1, H1W2 + b2, H2W3 + b3
        caches = [] # x_in, W_in, b_in
        activations = [] # Relu (x_out), Relu, Softmax 
        affine_relu_caches = [] # affine_in = LW_in + b_in, W_in, b_in 
        outs2 = []
        
        
        affine_batchnorm_relu_caches = [] 
        outs3 = []
        
        k=0
        #temp = False
        #for i in range(self.params)
        for par, value in self.params.items():
            if(k==0):
                out = X
                out2 = X
                out3 = X
            if(k%2 == 0):
                #print(par)
                k+=1
                w = value 
                continue;
            else:
                #print(par)
                k+=1
                b = value
                
                out, cache = affine_forward(out,w,b)
                caches.append(cache)
                ##
                out2, cache2=  affine_relu_forward(out2, w, b)
                outs2.append(out2)
                affine_relu_caches.append(cache2)
                ##
                
                ##
                out3, cache3=  affine_relu_forward(out3, w, b)
                outs3.append(out3)
                affine_batchnorm_relu_caches.append(cache3)
                ##
                
                if(k == len(self.params)):
                    affines.append(out)
                    break;
                else:
                    out, cache = relu_forward(out) # in this cache we have affines xw + b 
                    affines.append(cache)
                    activations.append(out)
        scores = out
        
        temp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        den = np.sum(temp, axis =1)  
        rl = np.transpose(np.divide(np.transpose(temp),den))
        
        activations.append(rl)                    
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        
        #loss
        N = X.shape[0]
        
        c = self.classes
        y_enc = []
        for i in range(len(y)):
            for el in c:
                #print(el, y[i])
                zero = np.zeros_like(c)
                if(el == y[i]):
                    zero[el]=1
                    break;
            y_enc.append(zero)
        y_hot_enc =  np.array(y_enc)  
        


        loss = np.trace(np.matmul(np.transpose(y_hot_enc), (np.log(rl)))) 
        loss = (-1.0/N)* loss
        
        k=0
        for par, value in self.params.items():
            if(k%2 == 0):
                loss = loss + 0.5* self.reg* np.sum(np.power(value,2))
                k+=1
       
        ##################
        #################
        dout = 1/N * (rl - y_hot_enc)
        last_len = int(len(self.params)/2)

        
        for i in reversed(range(last_len)):
            if(i== last_len-1):
                #print(i)
                w_str = 'W' + str(i+1)
                b_str = 'b'+ str (i+1)
                dnext, dw, db = affine_backward(dout, caches[i]) 
                
                grads[w_str] = dw + self.reg*self.params[w_str] 
                grads[b_str] = db
                dout = dnext
               
            else:
                w_str = 'W' + str(i+1)
                b_str = 'b'+ str (i+1)
                """
                # without batchnorm
                dnext, dw, db = affine_relu_backward(dout, affine_relu_caches[i]) 
                
                grads[w_str] = dw + self.reg*self.params[w_str] 
                grads[b_str] = db
                dout = dnext
                """
                dnext, dw, db = affine_batchnorm_relu_backward(dout, affine_batchnorm_relu_caches[i]) 
                
                grads[w_str] = dw + self.reg*self.params[w_str] 
                grads[b_str] = db
                dout = dnext

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
