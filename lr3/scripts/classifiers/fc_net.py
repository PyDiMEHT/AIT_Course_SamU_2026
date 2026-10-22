from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


import numpy as np
from scripts.layers import affine_forward, affine_backward, relu_forward, relu_backward
from scripts.layers import softmax_loss
# (подставьте правильные пути импорта из вашего проекта)

class TwoLayerNet(object):

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net.            #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net.                  #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # affine -> relu -> affine -> (softmax в тесте не нужен)
        h1, cache1 = affine_forward(X, W1, b1)      # (N, H)
        r1, cache_r1 = relu_forward(h1)             # (N, H)
        scores, cache2 = affine_forward(r1, W2, b2) # (N, C)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net.                 #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # ---- loss ----
        data_loss, dscores = softmax_loss(scores, y)          # (N, C)
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # ---- backward ----
        # второй affine-слой
        dr1, dW2, db2 = affine_backward(dscores, cache2)      # dr1: (N, H)
        dW2 += self.reg * W2                                  # + L2-градиент

        # ReLU
        dh1 = relu_backward(dr1, cache_r1)                    # (N, H)

        # первый affine-слой
        _, dW1, db1 = affine_backward(dh1, cache1)            # dW1: (D, H)
        dW1 += self.reg * W1                                  # + L2-градиент

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
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
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dims = [input_dim] + list(hidden_dims) + [num_classes]

        for i in range(self.num_layers):
            W = np.random.randn(dims[i], dims[i + 1]) * weight_scale
            b = np.zeros(dims[i + 1])
            self.params['W%d' % (i + 1)] = W
            self.params['b%d' % (i + 1)] = b

        if self.normalization in ('batchnorm', 'layernorm'):
            # гамма/бета для всех скрытых слоёв (последний affine — без нормы)
            for i in range(self.num_layers - 1):
                self.params['gamma%d' % (i + 1)] = np.ones(dims[i + 1])
                self.params['beta%d'  % (i + 1)] = np.zeros(dims[i + 1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = []                    # по одному на слой: (affine, norm, relu, dropout)
        h = X
        for i in range(1, self.num_layers):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            h, fc_cache = affine_forward(h, W, b)

            norm_cache = None
            if self.normalization == 'batchnorm':
                h, norm_cache = batchnorm_forward(
                    h, self.params['gamma%d' % i], self.params['beta%d' % i],
                    self.bn_params[i - 1])
            elif self.normalization == 'layernorm':
                h, norm_cache = layernorm_forward(
                    h, self.params['gamma%d' % i], self.params['beta%d' % i],
                    self.bn_params[i - 1])

            h, relu_cache = relu_forward(h)

            drop_cache = None
            if self.use_dropout:
                h, drop_cache = dropout_forward(h, self.dropout_param)

            caches.append((fc_cache, norm_cache, relu_cache, drop_cache))

        # последний affine → scores
        W, b = self.params['W%d' % self.num_layers], self.params['b%d' % self.num_layers]
        scores, last_fc_cache = affine_forward(h, W, b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W%d' % i] ** 2)
        loss = data_loss + reg_loss

        # ---- backward: последний affine ----
        dW = self.params['W%d' % self.num_layers]
        dh, grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = \
            affine_backward(dscores, last_fc_cache)
        grads['W%d' % self.num_layers] += self.reg * dW

        # ---- backward: скрытые слои (в обратном порядке) ----
        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, norm_cache, relu_cache, drop_cache = caches[i - 1]

            if self.use_dropout:
                dh = dropout_backward(dh, drop_cache)

            dh = relu_backward(dh, relu_cache)

            dgamma = dbeta = None
            if self.normalization == 'batchnorm':
                dh, dgamma, dbeta = batchnorm_backward(dh, norm_cache)
            elif self.normalization == 'layernorm':
                dh, dgamma, dbeta = layernorm_backward(dh, norm_cache)

            dh, dW, db = affine_backward(dh, fc_cache)

            grads['W%d' % i] = dW + self.reg * self.params['W%d' % i]
            grads['b%d' % i] = db
            if self.normalization is not None:
                grads['gamma%d' % i] = dgamma
                grads['beta%d'  % i] = dbeta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
