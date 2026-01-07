from __future__ import annotations

import numpy as np
import copy
import os
import pickle
import types
from collections import deque, Counter

import bitarray
import bitarray.util
import numpy as np
import scipy.special as special
import npn
import time
import json
import sys
import tracemalloc
import tensorflow as tf

import tf_keras as keras
import keras.backend as K
import gc
# from tensorflow import keras

# import keras
# from tensorflow import keras
# import tensorflow.keras
import sys
# replace your path
sys.path.append('./')

from tensorflow_models import nlp
from gtac.tensorflow_transformer import Seq2SeqTransformer, CustomSchedule, masked_loss, masked_accuracy
from gtac.utils import *

class MCTSNode:
    INIT_MAX_VALUE = -1000

    def __init__(self, parent, t, action, prob=None, info=None, puct_explore_ratio=1.):
        self.t = t
        self.parent = parent
        self.action = action
        self.explored = False
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.total_value = 0
        self.info = info
        self.v = None
        self.prob = prob
        self.max_value = self.INIT_MAX_VALUE
        self.puct_explore_ratio = puct_explore_ratio

    @property
    def value(self):  # Q
        return self.total_value / self.visits if self.visits != 0 else 100

    @property
    def puct(self):
        return self.value + self.puct_explore_ratio * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)

    def __repr__(self):     # sum reward: from the root to the end, value: future reward from (excluding) the current node to the end
        repr = "(%s%s, visits: %d, avg sum reward: %.2f, max sum reward: %d, value: %s, seq: %s)" % \
               (self.action, " (Done)" if self.info['done'] else "", self.visits, self.value, self.max_value, self.v, self.info['env'].tokens)
        if self.prob is not None:
            repr = repr[:-1] + ", prob: %.2f, puct: %.2f)" % (self.prob, self.puct)
        return repr


def ucb(node: MCTSNode):
    """UCB (Upper Confidence Bound) function"""
    return node.value + np.sqrt(np.log(node.parent.visits) / node.visits) 