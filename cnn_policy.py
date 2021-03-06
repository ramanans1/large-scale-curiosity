import tensorflow as tf
from distributions import make_pdtype
#from baselines.common.tf_util import load_variables, save_variables
import functools
import os.path as osp
import numpy as np

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim, get_session

class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.bool_actionclip = True #TODO Need to make this flexible
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        #self.ac_range = ac_range
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space) #RS: Should give a continuous action space, given  a continuous action env
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=False)
            self.features = unflatten_first_dim(self.flat_features, sh)

            with tf.variable_scope(scope, reuse=False):
                x = fc(self.flat_features, units=hidsize, activation=activ)
                x = fc(x, units=hidsize, activation=activ)
                pdparam = fc(x, name='pd', units=pdparamsize, activation=tf.nn.tanh)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
            pdparam = unflatten_first_dim(pdparam, sh)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.a_samp = self.clip_action(self.a_samp) if self.bool_actionclip else self.a_samp
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)
            self.pd_logstd = pd.logstd
            self.pd_std = pd.std
            self.pd_mean = pd.mean

    def get_features(self, x, reuse):
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def clip_action(self,action):

        forward = tf.stop_gradient(tf.clip_by_value(action, -7.0, 7.0))
        action = action - tf.stop_gradient(action) + forward

        return action

    def get_ac_value_nlp(self, ob):
        a, vpred, nlp, logstd, std, mean = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.pd_logstd, self.pd_std, self.pd_mean],
                          feed_dict={self.ph_ob: ob[:, None]})
        #print('---LOGSTD--',logstd[:,0])
        #print('---STD--',std[:,0])
        #print('---MEAN--',mean[:,0])
        return a[:,0], vpred[:, 0], nlp[:, 0]

    def get_ac_value_nlp_eval(self, ob):
        a, vpred, nlp = getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ((ob,),)})

        return a[:,0], vpred[:, 0], nlp[:, 0]
