import tensorflow.compat.v1 as tf


class Dense(object):
    def __init__(self, dim_in, dim_out, name='dense'):
        self.dim_in = dim_in
        self.dim_out = dim_out
        with tf.variable_scope(name):
            self.kernel = tf.get_variable('kernel', [dim_in, dim_out])
            self.bias = tf.Variable(tf.zeros([dim_out]), name='bias')
    
    def __call__(self, inputs, activation=lambda a: a):
        return activation(tf.matmul(inputs, self.kernel) + self.bias)

    def get_weights(self):
        return {self.kernel.name: self.kernel}

    def get_biases(self):
        return {self.bias.name: self.bias}

    
class GCN(Dense):
    # https://tkipf.github.io/graph-convolutional-networks/
    # https://arxiv.org/pdf/1609.02907.pdf
    def __init__(self, dim_in, dim_out, name='gcn'):
        super(GCN, self).__init__(dim_in, dim_out, name=name)
    
    def __call__(self, inputs, activation=lambda a: a, mask=1.0, sparse=False):
        if sparse and mask != 1.0:
            self.degree = tf.sparse.reduce_sum(mask, axis=1, keepdims=True)**-0.5
            self.alpha = mask*self.degree*tf.transpose(self.degree)
            return activation(tf.sparse.matmul(self.alpha, tf.matmul(inputs, self.kernel)) + self.bias)
        self.degree = tf.reduce_sum(mask, axis=1, keepdims=True)**-0.5
        self.alpha = mask*self.degree*tf.transpose(self.degree)
        return activation(tf.matmul(self.alpha, tf.matmul(inputs, self.kernel)) + self.bias)


class GATHead(Dense):
    # https://github.com/PetarV-/GAT/blob/master/utils/layers.py
    def __init__(self, dim_in, dim_out, residual=False, name='gathead'):
        super(GATHead, self).__init__(dim_in, dim_out, name=name)
        with tf.variable_scope(name):
            self.attention_self = Dense(dim_out, 1, name='self')
            self.attention_other = Dense(dim_out, 1, name='other')
            self.residual = Dense(dim_in, dim_out, name='residual') if residual else None

    def __call__(self, inputs, activation=lambda a: a, mask=1.0, sparse=False):
        x = tf.matmul(inputs, self.kernel)
        u, v = self.attention_self(x), self.attention_other(x)
        if sparse and mask != 1.0:
            logits = tf.sparse.add(mask*u, mask*tf.transpose(v))
            logits = tf.SparseTensor(logits.indices, tf.nn.leaky_relu(logits.values), logits.shape)
            self.alpha = tf.sparse.softmax(logits)
            out = tf.sparse.matmul(self.alpha, x) + self.bias
        else:
            logits = tf.nn.leaky_relu(u + tf.transpose(v))
            self.alpha = mask*tf.exp(logits - tf.reduce_max(logits, 1))
            self.alpha /= tf.reduce_sum(self.alpha, 1, keepdims=True)
            out = tf.matmul(self.alpha, x) + self.bias
        return activation(out if self.residual is None else out + self.residual(inputs))

    def get_weights(self):
        weights = {
            self.kernel.name: self.kernel,
            self.attention_self.kernel.name: self.attention_self.kernel,
            self.attention_other.kernel.name: self.attention_other.kernel,
        }
        if self.residual is not None:
            weights[self.residual.kernel.name] = self.residual.kernel
        return weights

    def get_biases(self):
        biases = {
            self.bias.name: self.bias,
            self.attention_self.bias.name: self.attention_self.bias,
            self.attention_other.bias.name: self.attention_other.bias,
        }
        if self.residual is not None:
            biases[self.residual.bias.name] = self.residual.bias
        return biases


class GAT(object):
    # https://arxiv.org/pdf/1710.10903.pdf
    def __init__(self, dim_in, dim_out, n_head, residual=False, concat=True, name='gat'):
        with tf.variable_scope(name):
            self.heads = [GATHead(dim_in, dim_out, residual=residual, name=str(i)) for i in range(n_head)]
        self.dim_in = dim_in
        self.n_head = n_head
        self.concat = concat
        self.dim_out = dim_out*n_head if concat else dim_out

    def __call__(self, inputs, activation=lambda a: a, mask=1.0, sparse=False):
        if self.concat:
            return tf.concat([h(inputs, activation=activation, mask=mask, sparse=sparse) for h in self.heads], 1)
        return activation(tf.reduce_mean([h(inputs, mask=mask, sparse=sparse) for h in self.heads], 0))

    def get_weights(self):
        weights = {}
        for h in self.heads:
            weights.update(h.get_weights())
        return weights

    def get_biases(self):
        biases = {}
        for h in self.heads:
            biases.update(h.get_biases())
        return biases
