import numpy as np

from code.layers import *
from code.fast_layers import *
from code.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    stride = 1
    pad = (filter_size - 1)/2
    H_out = (1 + (H + 2 * pad - filter_size) / 2)
    W_out = (1 + (W + 2 * pad - filter_size) / 2)
    
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(H_out * W_out * num_filters, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X2, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    X3, cache2 = affine_relu_forward(X2, W2, b2)
    scores, cache3 = affine_forward(X3, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dout = softmax_loss(scores, y)
    
    reg_W1_loss = 0.5 * self.reg * np.sum(self.params['W1']**2)
    W1_reg = self.reg * self.params['W1']
    reg_W2_loss = 0.5 * self.reg * np.sum(self.params['W2']**2)
    W2_reg = self.reg * self.params['W2']
    reg_W3_loss = 0.5 * self.reg * np.sum(self.params['W3']**2)
    W3_reg = self.reg * self.params['W3']   
    loss = data_loss + reg_W1_loss + reg_W2_loss + reg_W3_loss
    
    dx3, dw3, db3 = affine_backward(dout, cache3)
    dx2, dw2, db2 = affine_relu_backward(dx3, cache2)
    dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache1)
    
    grads['W1'] = W1_reg + dw1
    grads['W2'] = W2_reg + dw2
    grads['W3'] = W3_reg + dw3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class ThreeLayerConvNetWithBatchnorm(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - batchnorm - relu - 2x2 max pool - affine - batchnorm - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = {}
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    stride = 1
    pad = (filter_size - 1)/2
    H_out = (1 + (H + 2 * pad - filter_size) / stride) /2
    W_out = (1 + (W + 2 * pad - filter_size) / stride) /2
    
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(H_out * W_out * num_filters, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    bn_param1 = {'mode': 'train'}
    bn_param2 = {'mode': 'train'}
    self.bn_params['bn_param1'] = bn_param1
    self.bn_params['bn_param2'] = bn_param2
    
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #x, w, b, conv_param, pool_param, gamma, beta, bn_param
    X2, cache1  = conv_batch_relu_pool_forward(X, W1, b1, conv_param, pool_param, self.params['gamma1'], self.params['beta1'], self.bn_params['bn_param1'])
    X3, cache2 = affine_batchnorm_relu_forward(X2, W2, b2, self.params['gamma2'], self.params['beta2'], self.bn_params['bn_param2'])
    scores, cache3 = affine_forward(X3, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if mode == 'test':
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dout = softmax_loss(scores, y)
    
    reg_W1_loss = 0.5 * self.reg * np.sum(self.params['W1']**2)
    W1_reg = self.reg * self.params['W1']
    reg_W2_loss = 0.5 * self.reg * np.sum(self.params['W2']**2)
    W2_reg = self.reg * self.params['W2']
    reg_W3_loss = 0.5 * self.reg * np.sum(self.params['W3']**2)
    W3_reg = self.reg * self.params['W3']   
    loss = data_loss + reg_W1_loss + reg_W2_loss + reg_W3_loss
    
    dx3, dw3, db3 = affine_backward(dout, cache3)
    dx2, dw2, db2, dgamma2, dbeta2 = affine_batchnorm_relu_backward(dx3, cache2)
    # dx2, dw2, db2 = affine_relu_backward(dx3, cache2)
    dx1, dw1, db1, dgamma1, dbeta1 = conv_batch_relu_pool_backward(dx2, cache1)
    
    grads['W1'] = W1_reg + dw1
    grads['W2'] = W2_reg + dw2
    grads['W3'] = W3_reg + dw3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
 
pass
