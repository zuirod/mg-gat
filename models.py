import os

from hyperopt import hp
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import svds
import tensorflow.compat.v1 as tf

from datasets import Dataset
import layers
import metrics


class Model(object):
    def __init__(self, session, dataset, **config):
        # tensorflow Session object
        self.session = session
        # datasets.Dataset object
        self.dataset = dataset
        # dict of hyperparameters, etc.
        self.config = config

        # inputs
        self.id = tf.placeholder(tf.int64, [None])
        self.user_id = tf.placeholder(tf.int64, [None])
        self.item_id= tf.placeholder(tf.int64, [None])
        # labels
        self.r_true = tf.placeholder(tf.float32, [None])
        
        # define model parameters
        self.weights, self.biases = self._params()
        # define rating computation and scale to dataset range
        self.r_pred = self.dataset.min + self.dataset.range*tf.sigmoid(self._r_pred())
        # define loss
        self.loss = self._loss()
        # define rmse metric for update monitoring
        self.rmse = tf.reduce_mean((self.r_pred - self.r_true)**2)**0.5
        
        # Adam optimization with default learning rate
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def _params(self):
        # see example models below
        weights, biases = {}, {}
        return weights, biases

    def _r_pred(self):
        # see example models below
        raise NotImplementedError()

    def _loss(self):
        self.mse = tf.reduce_mean((self.r_pred - self.r_true)**2)
        self.reg = self._reg()
        # alpha hyperparameter controls regularization, range 0 to 1.
        self.alpha = float(self.config.get('alpha', 0))
        # normalize mean square error so alpha is not dependent on dataset range
        return (1 - self.alpha)*self.mse/self.dataset.range**2 + self.alpha*self.reg

    def _reg(self):
        user_graph = self.dataset.side_info.get('user_graph', None)
        item_graph = self.dataset.side_info.get('item_graph', None)
        
        # compute L2 and graph regularization for all weights
        self.reg_l2, self.reg_graph = 0, 0
        for w in self.weights.values():
            self.reg_l2 += tf.reduce_sum(w**2)
            if w.shape[0] == self.dataset.n_user:
                self.reg_graph += self._graph_reg(user_graph, w)
            elif w.shape[0] == self.dataset.n_item:
                self.reg_graph += self._graph_reg(item_graph, w)
            else:
                self.reg_graph += self._graph_reg(None, w)
        
        # beta hyperparameter controls L2 vs graph reg, range 0 to 1.
        self.beta = float(self.config.get('beta', 0))
        return (1 - self.beta)*self.reg_l2 + self.beta*self.reg_graph

    def _graph_reg(self, g, w):
        if g is None:
            # if no graph is provided, use L2 reg
            return tf.reduce_sum(w**2)
        if len(w.shape) == 1:
            w = tf.reshape(w, (-1, 1))
        # normed controls whether or not to use normed graph laplacian
        normed = self.config.get('normed', False)
        # sparse controls whether or not to use sparse tensors
        if self.config.get('sparse', True):
            s = laplacian(sp.coo_matrix(g), normed=normed).astype(np.float32)
            s = tf.sparse.reorder(tf.SparseTensor(np.array([s.row, s.col]).T, s.data, s.shape))
            return tf.linalg.trace(tf.matmul(w, tf.sparse.matmul(s, w), transpose_a=True))
        s = tf.constant(laplacian(g.A if sp.issparse(g) else g, normed=normed), dtype=tf.float32)
        return tf.linalg.trace(tf.matmul(w, tf.matmul(s, w), transpose_a=True))

    def run(self, ops, batch):
        return self.session.run(ops, {
            self.id: batch.index,
            self.user_id: batch.user_id,
            self.item_id: batch.item_id,
            self.r_true: batch.rating,
        })

    def train(self, max_updates=100000, n_check=100, patience=float('inf'), batch_size=None):
        self.session.run(tf.global_variables_initializer())
        best = {'updates': 0, 'loss': float('inf'),  'rmse_tune': float('inf')}
        for i in range(max_updates):
            # update
            opt, loss = self.run([self.opt, self.loss], self.dataset.get_batch(mode='train', size=batch_size))
            if i % n_check == 0 or i == max_updates - 1:
                # monitoring
                rmse_tune = self.run(self.rmse, self.dataset.get_batch(mode='tune', size=None))
                if len(self.dataset.tune) == 0 or rmse_tune < best['rmse_tune']:
                    rmse_test = self.run(self.rmse, self.dataset.get_batch(mode='test', size=None))
                    best = {'updates': i, 'loss': loss, 'rmse_tune': rmse_tune, 'rmse_test': rmse_test}
                print(best)
                if (i - best['updates'])//n_check > patience:
                    # early stopping
                    break
        return best
    
    def test(self):
        return self._metrics(self.dataset.get_batch(mode='test', size=None))
   
    def _metrics(self, batch):
        r_pred = self.run(self.r_pred, batch)
        r_true, user_ids, item_ids = batch[['rating', 'user_id', 'item_id']].values.T
        return {
            'rmse': metrics.bootstrap(metrics.rmse, r_pred, r_true, user_ids, item_ids),
#             'mae': metrics.bootstrap(metrics.mae, r_pred, r_true, user_ids, item_ids),
#             'spearman': metrics.bootstrap(metrics.spearman, r_pred, r_true, user_ids, item_ids),
#             'fcp': metrics.bootstrap(metrics.fcp, r_pred, r_true, user_ids, item_ids),
#             'bpr': metrics.bootstrap(metrics.bpr, r_pred, r_true, user_ids, item_ids),
       }

    def _schema(self):
        # parameters to save
        return {
            'weights': self.weights,
            'biases': self.biases,
        }

    def _mask(self, g):
        # define the mask used for GAT
        if g is None:
            # no masking
            return 1.0
        if self.config.get('sparse', True):
            shape = g.shape
            g = sp.coo_matrix(g, shape=shape, dtype=np.float32)
            g = tf.sparse.reorder(tf.SparseTensor(np.array([g.row, g.col]).T, g.data, shape))
            return tf.sparse.add(tf.sparse.eye(*shape), g)
        return tf.eye(*g.shape) + tf.constant(g.A if sp.issparse(g) else g, dtype=tf.float32)

    def _normalized_aggregation(self, g, w):
        # define aggregation used for SVD++
        if g is None:
            return 0.0
        if self.config.get('sparse', True):
            g = sp.coo_matrix(g, dtype=np.float32)
            g = tf.sparse.reorder(tf.SparseTensor(np.array([g.row, g.col]).T, g.data, g.shape))
            return tf.sparse.matmul(g, w)/(tf.sparse.reduce_sum(g, axis=1, keepdims=True)**0.5 + 1e-10)
        g = tf.constant(g.A if sp.issparse(g) else g, dtype=tf.float32)
        return tf.matmul(g, w)/(tf.reduce_sum(g, axis=1, keepdims=True)**0.5 + 1e-10)


class SVD(Model):
    # GRALS: https://www.cs.utexas.edu/~rofuyu/papers/grmf-nips.pdf
    def _params(self):
        self.rank = int(self.config.get('rank', 1))
        weights = {
            'user_factor': tf.get_variable('user_factor', [self.dataset.n_user, self.rank]),
            'item_factor': tf.get_variable('item_factor', [self.dataset.n_item, self.rank]),
            'user_bias': tf.Variable(tf.zeros([self.dataset.n_user])),
            'item_bias': tf.Variable(tf.zeros([self.dataset.n_item])),
        }
        biases = {'bias': tf.Variable(0.0)}
        self.user_factor = tf.nn.embedding_lookup(weights['user_factor'], self.user_id)
        self.item_factor = tf.nn.embedding_lookup(weights['item_factor'], self.item_id)
        self.user_bias = tf.nn.embedding_lookup(weights['user_bias'], self.user_id)
        self.item_bias = tf.nn.embedding_lookup(weights['item_bias'], self.item_id)
        self.bias = biases['bias']
        return weights, biases

    def _r_pred(self):
        return (tf.reduce_sum(self.user_factor*self.item_factor, 1)
            + self.user_bias + self.item_bias + self.bias)
    

class SVDpp(SVD):
    # SVD++: https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf
    def _params(self):
        self.config['beta'] = 0.0
        weights, biases = super(SVDpp, self)._params()
        weights['item_factor_pp'] = tf.get_variable('item_factor_pp', [self.dataset.n_item, self.rank])
        g = sp.coo_matrix((self.dataset.data.is_train, (self.dataset.data.user_id, self.dataset.data.item_id)))
        user_factor_pp = self._normalized_aggregation(g, weights['item_factor_pp'])
        self.user_factor = tf.nn.embedding_lookup(weights['user_factor'] + user_factor_pp, self.user_id)
        return weights, biases


class MGCNN(Model):
    # sRGCNN: https://github.com/fmonti/mgcnn
    def _params(self):
        self.config['normed'] = True
        self.config['beta'] = 1.0
        self.rank = int(self.config.get('rank', 1))
        self.n_conv_feat = int(self.config.get('n_conv_feat', 32))
        self.ord_row = int(self.config.get('ord_row', 5))
        self.ord_col = int(self.config.get('ord_col', 5))
        self.num_iterations = int(self.config.get('num_iterations', 10))

        ##################################definition of the NN variables#####################################

        #definition of the weights for extracting the global features
        self.W_conv_W = tf.get_variable("W_conv_W", shape=[self.ord_col*self.rank, self.n_conv_feat])
        self.b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.W_conv_H = tf.get_variable("W_conv_H", shape=[self.ord_row*self.rank, self.n_conv_feat])
        self.b_conv_H = tf.Variable(tf.zeros([self.n_conv_feat,]))

        #recurrent N parameters
        self.W_f_u = tf.get_variable("W_f_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_i_u = tf.get_variable("W_i_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_o_u = tf.get_variable("W_o_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_c_u = tf.get_variable("W_c_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_f_u = tf.get_variable("U_f_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_i_u = tf.get_variable("U_i_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_o_u = tf.get_variable("U_o_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_c_u = tf.get_variable("U_c_u", shape=[self.n_conv_feat, self.n_conv_feat])
        self.b_f_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_i_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_o_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_c_u = tf.Variable(tf.zeros([self.n_conv_feat,]))

        self.W_f_m = tf.get_variable("W_f_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_i_m = tf.get_variable("W_i_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_o_m = tf.get_variable("W_o_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.W_c_m = tf.get_variable("W_c_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_f_m = tf.get_variable("U_f_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_i_m = tf.get_variable("U_i_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_o_m = tf.get_variable("U_o_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.U_c_m = tf.get_variable("U_c_m", shape=[self.n_conv_feat, self.n_conv_feat])
        self.b_f_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_i_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_o_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
        self.b_c_m = tf.Variable(tf.zeros([self.n_conv_feat,]))

        #output parameters
        self.W_out_W = tf.get_variable("W_out_W", shape=[self.n_conv_feat, self.rank]) 
        self.b_out_W = tf.Variable(tf.zeros([self.rank,]))
        self.W_out_H = tf.get_variable("W_out_H", shape=[self.n_conv_feat, self.rank]) 
        self.b_out_H = tf.Variable(tf.zeros([self.rank,]))

        #########definition of the NN
        #definition of W and H
        d, shape = self.dataset.train, (self.dataset.n_user, self.dataset.n_item)
        matrix_train = sp.coo_matrix((d.rating, (d.user_id, d.item_id)), shape=shape)
        u, s, vt = svds(matrix_train.astype(float), k=self.rank)
        self.W = tf.constant(u*s**0.5, dtype=tf.float32)
        self.H = tf.constant(vt.T*s**0.5, dtype=tf.float32)

        #RNN
        self.h_u = tf.zeros([self.dataset.n_user, self.n_conv_feat])
        self.c_u = tf.zeros([self.dataset.n_user, self.n_conv_feat])
        self.h_m = tf.zeros([self.dataset.n_item, self.n_conv_feat])
        self.c_m = tf.zeros([self.dataset.n_item, self.n_conv_feat])

        for k in range(self.num_iterations):
            #extraction of global features vectors
            self.final_feat_users = self._mono_conv(
                self.dataset.side_info['user_graph'], self.ord_row, self.W, self.W_conv_W, self.b_conv_W)
            self.final_feat_movies = self._mono_conv(
                self.dataset.side_info['item_graph'], self.ord_col, self.H, self.W_conv_H, self.b_conv_H)

            #here we have to split the features between users and movies LSTMs

            #users RNN
            self.f_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_f_u) + tf.matmul(self.h_u, self.U_f_u) + self.b_f_u)
            self.i_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_i_u) + tf.matmul(self.h_u, self.U_i_u) + self.b_i_u)
            self.o_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_o_u) + tf.matmul(self.h_u, self.U_o_u) + self.b_o_u)

            self.update_c_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_c_u) + tf.matmul(self.h_u, self.U_c_u) + self.b_c_u)
            self.c_u = tf.multiply(self.f_u, self.c_u) + tf.multiply(self.i_u, self.update_c_u)
            self.h_u = tf.multiply(self.o_u, tf.sigmoid(self.c_u))

            #movies RNN
            self.f_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_f_m) + tf.matmul(self.h_m, self.U_f_m) + self.b_f_m)
            self.i_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_i_m) + tf.matmul(self.h_m, self.U_i_m) + self.b_i_m)
            self.o_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_o_m) + tf.matmul(self.h_m, self.U_o_m) + self.b_o_m)

            self.update_c_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_c_m) + tf.matmul(self.h_m, self.U_c_m) + self.b_c_m)
            self.c_m = tf.multiply(self.f_m, self.c_m) + tf.multiply(self.i_m, self.update_c_m)
            self.h_m = tf.multiply(self.o_m, tf.sigmoid(self.c_m))

            #compute update of matrix X
            self.delta_W = tf.tanh(tf.matmul(self.c_u, self.W_out_W) + self.b_out_W) #N x rank_W_H
            self.delta_H = tf.tanh(tf.matmul(self.c_m, self.W_out_H) + self.b_out_H) #M x rank_W_H

            self.W += self.delta_W
            self.H += self.delta_H
            
        self.user_factor = tf.nn.embedding_lookup(self.W, self.user_id)
        self.item_factor = tf.nn.embedding_lookup(self.H, self.item_id)
        
        return {'W': self.W, 'H': self.H}, {}

    def _r_pred(self):
        return tf.reduce_sum(self.user_factor*self.item_factor, 1)

    def _mono_conv(self, g, ord_conv, A, W, b):
        if self.config.get('sparse', True):
            L = sp.coo_matrix(laplacian(g, normed=True) - sp.eye(*g.shape), dtype=np.float32)
            L = tf.sparse.reorder(tf.SparseTensor(np.array([L.row, L.col]).T, L.data, L.shape))
            matmul = tf.sparse.matmul
        else:
            L = tf.constant(laplacian(g, normed=True) - np.eye(*g.shape), dtype=tf.float32)
            matmul = tf.matmul
        feat = []
        for k in range(ord_conv):
            if k == 0:
                feat.append(A)
            elif k == 1:
                feat.append(matmul(L, A))
            else:
                feat.append(matmul(L, 2*feat[k - 1]) - feat[k - 2])
        return tf.nn.relu(tf.matmul(tf.concat(feat, 1), W) + b)
    

class GATSVD(SVD):
    # MG-GAT: our model
    def _params(self):
        weights, biases = super(GATSVD, self)._params()
        data, shape = self.dataset.data, (self.dataset.n_user, self.dataset.n_item)
        implicit = sp.coo_matrix((data.is_train, (data.user_id, data.item_id)), shape=shape)
        self.k = int(self.config.get('k', 1))
        u, s, vt = svds(implicit.astype(float), k=self.k)
        self.user_features = tf.constant(u*s**0.5, dtype=tf.float32)
        self.item_features = tf.constant(vt.T*s**0.5, dtype=tf.float32)
        user_features = self.dataset.side_info.get('user_features', None)
        item_features = self.dataset.side_info.get('item_features', None)
        if user_features is not None:
            self.user_features = tf.concat(
                [self.user_features, tf.constant(user_features.values, dtype=tf.float32)], 1)
        if item_features is not None:
            self.item_features = tf.concat(
                [self.item_features, tf.constant(item_features.values, dtype=tf.float32)], 1)
        self.n_head = int(self.config.get('n_head', 1))
        self.activation_in = tf.keras.activations.get(self.config.get('activation_in', 'softsign'))
        self.activation_out = tf.keras.activations.get(self.config.get('activation_out', 'hard_sigmoid'))
        self.residual = bool(self.config.get('residual', True))
        self.user_in = layers.GAT(
            self.user_features.shape[1], self.n_head*self.rank, 1, concat=False, residual=self.residual, name='user_in')
        self.user_out = layers.Dense(self.user_in.dim_out, self.rank, name='user_out')
        self.item_in = layers.GAT(
            self.item_features.shape[1], self.n_head*self.rank, 1, concat=False, residual=self.residual, name='item_in')
        self.item_out = layers.Dense(self.item_in.dim_out, self.rank, name='item_out')
        for layer in [self.user_in, self.user_out, self.item_in, self.item_out]:
            weights.update(layer.get_weights())
            biases.update(layer.get_biases())
        self.user_mask = self._mask(self.dataset.side_info.get('user_graph', None))
        self.item_mask = self._mask(self.dataset.side_info.get('item_graph', None))
        sparse = bool(self.config.get('sparse', True))
        self.user_factor_pp = self.user_out(self.user_in(
            self.user_features, self.activation_in, self.user_mask, sparse), self.activation_out)
        self.item_factor_pp = self.item_out(self.item_in(
            self.item_features, self.activation_in, self.item_mask, sparse), self.activation_out)
        self.user_factor = tf.nn.embedding_lookup(weights['user_factor'] + self.user_factor_pp, self.user_id)
        self.item_factor = tf.nn.embedding_lookup(weights['item_factor'] + self.item_factor_pp, self.item_id)
        return weights, biases

    def _schema(self):
        return {
            'weights': self.weights,
            'biases': self.biases,
            'outputs': {
                'user_alpha': self.user_in.heads[0].alpha,
                'item_alpha': self.item_in.heads[0].alpha,
                'user_factor_pp': self.user_factor_pp,
                'item_factor_pp': self.item_factor_pp,
                'user_factor': self.weights['user_factor'] + self.user_factor_pp,
                'item_factor': self.weights['item_factor'] + self.item_factor_pp,
            },
        }


if __name__ == '__main__':
    # Arguments
    MODEL = GATSVD # SVD, SVDpp, MGCNN, or GATSVD
    CWD = os.getcwd()
    DATASET_PATH = CWD + '/data/datasets/MovieLens100K' # check data/datasets for options
    METRICS_PATH = CWD + '/data/results/metrics'
    RAY_RESULTS = CWD + '/data/results/ray_results'
    BATCH_SIZE = None # change to integer to use minibatches
    TUNE = False # False to use saved hyperparameters, True to search
    N_SAMPLES = 500 # number of hyperparameter sets to try if TUNE == True
    ACTIVATIONS = [
        'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu',
        'selu', 'sigmoid', 'softplus', 'softsign', 'tanh',
    ]
    DATASET = Dataset.load(DATASET_PATH)
    NAME = '{}_{}'.format(MODEL.__name__, DATASET.name)
    N_FOLDS = 10 if len(DATASET.tune) == 0 else 1 # 10-fold cross validation
    
    
    if TUNE:
        def validation(config):
            DATASET = Dataset.load(DATASET_PATH)
            if len(DATASET.tune) == 0:
                FOLDS = np.random.randint(N_FOLDS, size=DATASET.n_data - len(DATASET.test))

            # add DATASET.max so np.mean(metrics['rmse_tune']) decreases with each iteration
            # which is necessary for AsyncHyperBandScheduler
            metrics = {'updates': [], 'rmse': [DATASET.max]}
            for j in range(N_FOLDS):
                data = DATASET.data[['user_id', 'item_id', 'rating', 'is_test', 'is_tune']]
                if len(DATASET.tune) == 0:
                    data.loc[data.is_test == False, 'is_tune'] = FOLDS == j
                dataset = Dataset(data, **DATASET.side_info)
                with tf.Graph().as_default():
                    with tf.Session() as session:
                        model = MODEL(session, dataset, **config)
                        best = model.train(patience=10, batch_size=BATCH_SIZE)
                metrics['updates'].append(best['updates'])
                metrics['rmse'].append(best['rmse_tune'])
                tune.report(updates=np.mean(metrics['updates']), rmse=np.mean(metrics['rmse']))
        
        tune.run(
            validation,
            name=NAME,
            search_alg=HyperOptSearch({
                    'alpha': hp.loguniform('alpha', -20, 0),
                    'beta': hp.loguniform('beta', -20, 0),
                    'rank': hp.qloguniform('rank', 0, 5, 1),
                    'k': hp.qloguniform('k', 0, 3, 1),
                    'n_head': hp.qloguniform('n_head', 0, 2, 1),
                    'activation_in': hp.choice('activation_in', ACTIVATIONS),
                    'activation_out': hp.choice('activation_out', ACTIVATIONS),
                    'residual': hp.choice('residual', [True, False]),
                }, metric='rmse', mode='min'),
            scheduler=AsyncHyperBandScheduler(metric='rmse', mode='min', max_t=N_FOLDS),
            resources_per_trial={"gpu": 1},
            local_dir=RAY_RESULTS,
            num_samples=N_SAMPLES)

    analysis = tune.Analysis('{}/{}'.format(RAY_RESULTS, NAME))
    config = analysis.get_best_config(metric='rmse', mode='min')
    df = analysis.dataframe()
    config['max_updates'] = int(df[df['config/alpha'] == config['alpha']]['updates'])
    print(config)

    data = DATASET.data[['user_id', 'item_id', 'rating', 'is_test']]
    dataset = Dataset(data, **DATASET.side_info)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = MODEL(session, dataset, **config)
            model.train(max_updates=config['max_updates'], batch_size=BATCH_SIZE)
            best = pd.DataFrame(model.test())
            print(best.describe())
            best.to_csv('{}/{}.csv'.format(METRICS_PATH, NAME), index=False)
