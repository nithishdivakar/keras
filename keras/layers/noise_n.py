from __future__ import absolute_import
from ..engine import Layer
from .. import backend as K


class Binomial(Layer):
    '''
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Binomial, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.random_binomial(shape=K.shape(x), p=x)

    def get_config(self):
        config = {'binomial':'binomial'}
        base_config = super(Binomial, self).get_config()
        return dict(list(base_config.items())+ list(config.items()))

