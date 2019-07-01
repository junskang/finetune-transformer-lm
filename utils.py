import os
import re
import sys
import json
import math
import time
import unicodedata
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tqdm import tqdm
from functools import partial

def encode_dataset(*splits, encoder):
    encoded_splits = []
    for split in splits[0]: # train, valid, test
        fields = []
        for field in split: # context, claim, pm
            if isinstance(field[0], str):
                field = encoder.encode(field)
            else:
                print ("\n[%s][%s]"%(type(field[0]), field[0][:100]))

            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits

def stsb_label_encoding(labels, nclass=6):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype(np.float32)
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    if a dimension is "None", use the static(?) one
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def np_softmax(x, t=1):
    x = x/t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex/np.sum(ex, axis=-1, keepdims=True)

def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n)*scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)

def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)

def _np_init(shape, dtype, partition_info, w):
    return w

def np_init(w):
    return partial(_np_init, w=w)

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))

def flatten(outer):
    return [el for inner in outer for el in inner]

def remove_none(l):
    return [e for e in l if e is not None]

def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign

def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def print_args(args):
    print("DESC:%s" % (args.desc))
    print("DATASET:%s" % (args.dataset))
    print("LOG_DIR:%s" % (args.log_dir))
    print("SAVE_DIR:%s" % (args.save_dir))
    print("DATA_DIR:%s" % (args.data_dir))
    print("SUBMISSION_DIR:%s" % (args.submission_dir))
    print("SUBMIT:%s" % (args.submit))
    print("ANALYSIS:%s" % (args.analysis))
    print("SEED:%s" % (args.seed))
    print("N_ITER:%s" % (args.n_iter))
    print("N_BATCH:%s" % (args.n_batch))
    print("MAX_GRAD_NORM:%s" % (args.max_grad_norm))
    print("LR:%s" % (args.lr))
    print("LR_WARMUP:%s" % (args.lr_warmup))
    print("N_CTX:%s" % (args.n_ctx))
    print("N_EMBD:%s" % (args.n_embd))
    print("N_HEAD:%s" % (args.n_head))
    print("N_LAYER:%s" % (args.n_layer))
    print("EMBD_PDROP:%s" % (args.embd_pdrop))
    print("ATTN_PDROP:%s" % (args.attn_pdrop))
    print("RESID_PDROP:%s" % (args.resid_pdrop))
    print("CLF_PDROP:%s" % (args.clf_pdrop))
    print("L2:%s" % (args.l2))
    print("VECTOR_L2:%s" % (args.vector_l2))
    print("N_GPU:%s" % (args.n_gpu))
    print("OPT:%s" % (args.opt))
    print("AFN:%s" % (args.afn))
    print("LR_SCHEDULE:%s" % (args.lr_schedule))
    print("ENCODER_PATH:%s" % (args.encoder_path))
    print("BPE_PATH:%s" % (args.bpe_path))
    print("N_TRANSFER:%s" % (args.n_transfer))
    print("LM_COEF:%s" % (args.lm_coef))
    print("B1:%s" % (args.b1))
    print("B2:%s" % (args.b2))
    print("E:%s" % (args.e))