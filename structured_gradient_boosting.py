# Copyright 2018 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured Gradient Tree Boosting

This module contains methods for fitting gradient boosted regression trees for
structured prediction problems.

"""

# Author: Yi Yang
# Email: yyang464@bloomberg.net 
#        yiyangnlp@gmail.com

from __future__ import print_function
from __future__ import division

from abc import abstractmethod

from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.tree.tree import DecisionTreeRegressor

import numbers, cPickle
import numpy as np

from time import time
from multiprocessing import Process, Queue

class StructuredGradientBoosting(object):
    """Structured Gradient Boosting. (S-MART) """

    def __init__(self, n_estimators, beam_width, learning_rate, # SGTB params
            min_samples_split, min_samples_leaf, max_depth,     # Tree params
            ent_ent_feat_dict, num_thread, random_state=1234):

        # sklearn tree related
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self.estimators = np.empty((0, 0), dtype=np.object)
        self.n_estimated = 0

        # structured learning
        self.ent_ent_feat_dict = ent_ent_feat_dict
        self.num_ent_ent_feat = len(ent_ent_feat_dict.values()[0])
        self.beam_width = beam_width # width of beam for global training and testing
        self.num_thread = num_thread

    def fit_stage(self, i, X, y):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        # induce regression tree on residuals
        tree = DecisionTreeRegressor(
            criterion='friedman_mse',
            splitter='best',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=0.,
            max_features=None,
            max_leaf_nodes=None,
            random_state=self.random_state,
            presort=False)

        tree.fit(X, y, check_input=False, X_idx_sorted=None)

        # add tree to ensemble
        self.estimators[i, 0] = tree
        self.n_estimated = i + 1


    def fit(self, train_set, dev_set):
        """Fit the gradient boosting model.

       """
        # init state
        self.estimators = np.empty((self.n_estimators, 1),
                                    dtype=np.object)

        # fit the boosting stages
        n_stages = self.fit_stages(train_set, dev_set)

        return self

    # MAIN METHODs
    def fit_stages(self, train_set, dev_set):
        """Iteratively fits the stages.

        """
        X, y, indices, ent_ids = train_set

        # perform boosting iterations
        for i in range(self.n_estimators):
            start_time = time()
            batches = self.split(indices, self.num_thread, True)
            q = Queue()
            procs = []
            for batch in batches:
                proc = Process(target=self.get_func_grad, args=(batch, X, y,
                    ent_ids, q))
                procs.append(proc)
                proc.start()
            result_list = []
            for _ in xrange(len(batches)):
                result_list.append(q.get())
            for proc in procs:
                proc.join()
            X_aug, y_aug, sum_match_cnt, sum_gold_cnt = self.merge(result_list)
            train_acc = float(sum_match_cnt*100) / sum_gold_cnt
            X_aug, y_aug = np.array(X_aug, dtype='float32'), np.array(y_aug, dtype='float32')
            end_time = time()

            # report dev accuracy every 25 iterations
            if (i+1) % 25 == 0: 
                dev_X, dev_y, dev_indices, dev_ent_ids = dev_set
                dev_acc = self.get_acc(dev_X, dev_y, dev_indices, dev_ent_ids)

                print ("Iter %d  Takes %.4f sec.  Train acc %.2f  Dev acc %.2f" %(i+1, end_time-start_time, train_acc, dev_acc))

            # fit next stage of trees
            self.fit_stage(i, X_aug, y_aug)

        return i + 1


    def beam_search(self, doc, X, ent_ids, gold_seq=None):
        """ Beam search, used for training if 'gold_seq' is given, otherwise testing. 
            'aggregated_score_dict' is used to save previously computed scores of
            sequence prefices.
        """
        feat_seq_logprobs = []
        aggregated_score_dict={} 
        prev_beam=[[]]

        for i, ent_list in enumerate(doc): # ent_list is the ent indices for a mention
            if gold_seq: 
                inner_gold_seq = gold_seq[:i+1]
            next_beam = [[]]
            sorted_list = []
            for prev_seq in prev_beam:
                for ent_idx in ent_list:
                    local_feats = X[ent_idx]
                    score, feats = self.get_score(local_feats, ent_idx,
                            prev_seq, ent_ids)
                    if len(prev_seq) == 0:
                        aggregated_score = score
                    else:
                        aggregated_score = aggregated_score_dict[tuple(prev_seq)] + score
                    curr_seq = prev_seq + [ent_idx]
                    aggregated_score_dict[tuple(curr_seq)] = aggregated_score
 
                    max_score = -float('inf')
                    max_tup = None
                    for next_seq in next_beam:
                        additional_score = 0.
                        if len(next_seq) > 0:
                            additional_score = aggregated_score_dict[tuple(next_seq + [ent_idx])]
                        if aggregated_score + additional_score > max_score:
                            max_score = aggregated_score + additional_score
                    max_tup = (feats, curr_seq, aggregated_score, max_score)
                    sorted_list.append(max_tup)
                    if gold_seq and tuple(max_tup[1]) == tuple(inner_gold_seq):
                        inner_gold_tuple = sorted_list[-1]
            sorted_list = sorted(sorted_list, key=lambda p : p[3], reverse=True)
            gold_in = False
            final_beam = []
            for tup in sorted_list[:self.beam_width]:
                final_beam.append(tup)
                if gold_seq and tuple(tup[1]) == tuple(inner_gold_seq):
                    gold_in = True
            if gold_seq and not gold_in:
                final_beam[-1] = inner_gold_tuple
            inner_feat_seq_logprobs = []
            prev_beam = []
            for tup in final_beam:
                prev_beam.append(tup[1])
                inner_feat_seq_logprobs.append((tup[0], tup[1], tup[2]))
            feat_seq_logprobs.append(inner_feat_seq_logprobs)
        return feat_seq_logprobs

    def compute_func_grad(self, feat_seq_logprobs, y):
        """ Compute functional gradients and evaluation statistics """
        new_X, func_grads = [], []
        prev_X, prev_grads = [], []
        final_prob = 0.
        for inner_feat_seq_logprobs in feat_seq_logprobs:
            temp_X, temp_grads = [], []
            z_score = 0.
            gold_in = False
            for _, _, logprob in inner_feat_seq_logprobs:
                z_score += np.exp(logprob)
            if z_score > 0:
                final_prob = np.exp(inner_feat_seq_logprobs[0][-1]) / z_score
            for feats, seq, logprob in inner_feat_seq_logprobs:
                prob = 0.
                if z_score > 0.: prob = np.exp(logprob) / z_score
                label = np.prod(y[np.array(seq, dtype='int32')])
                if label == 1: gold_in = True
                temp_X.append(feats)
                temp_grads.append(label - prob)
            if not gold_in: 
                break
            new_X += temp_X
            func_grads += temp_grads
            prev_X, prev_grads = temp_X, temp_grads

        pred_seq = feat_seq_logprobs[-1][0][1]
        return new_X, func_grads, pred_seq, final_prob

    def get_score(self, local_feats, ent_idx, prev_seq, ent_ids):
        ent_id = ent_ids[ent_idx]
        if len(prev_seq) == 0:
            global_feats = np.zeros(2 * self.num_ent_ent_feat, dtype='float32') # max pool + avg pool
        else:
            global_feat_list = []
            for prev_ent_idx in prev_seq:
                prev_ent_id = ent_ids[prev_ent_idx]
                feat_val = [0.] * self.num_ent_ent_feat
                if (prev_ent_id, ent_id) in self.ent_ent_feat_dict: 
                    feat_val = self.ent_ent_feat_dict[(prev_ent_id, ent_id)]
                elif (ent_id, prev_ent_id) in self.ent_ent_feat_dict: 
                    feat_val = self.ent_ent_feat_dict[(ent_id, prev_ent_id)]
                global_feat_list.append(feat_val)
            global_feat_mat = np.array(global_feat_list, dtype='float32')
            avg_pooled = np.mean(global_feat_mat, 0)
            max_pooled = np.max(global_feat_mat, 0)
            global_feats = np.concatenate([avg_pooled, max_pooled])
        feats = np.concatenate([local_feats, global_feats])
        refeats = feats.reshape((1, len(feats)))
        score = self.decision_function(refeats)
        score = score[0, 0]
        return score, feats

    def decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        score = np.zeros((X.shape[0], 1), dtype=np.float64)
        if self.n_estimated > 0: 
            predict_stages(self.estimators[:self.n_estimated], X, self.learning_rate, score)
        return score


    # MULTI-THREAD
    def split(self, indices, part=8, permutate=False):
        if permutate:
            permuted_idx = np.random.permutation(len(indices))
            indices = [indices[idx] for idx in permuted_idx]
        result = []
        batch_size = (len(indices) + part - 1) / part
        batch_size = int(batch_size)
        for j in xrange(part):
            docs = indices[j*batch_size:(j+1)*batch_size]
            result.append(docs)
        return result

    def merge(self, result_list):
        X_aug, y_aug = [], []
        sum_match_cnt, sum_gold_cnt = 0, 0
        for result in result_list:
            for p1, p2, match_cnt, gold_cnt in result:
                X_aug += p1
                y_aug += p2
                sum_match_cnt += match_cnt
                sum_gold_cnt += gold_cnt
        return X_aug, y_aug, sum_match_cnt, sum_gold_cnt

    def get_func_grad(self, batch_data, X, y, ent_ids, q, train_flag=True):
        result = []
        for doc, gold_ids in batch_data:
            if train_flag:
                gold_seq = []
                for ent_list in doc:
                    gold_idx = -1
                    for ent_idx in ent_list:
                        if y[ent_idx] == 1:
                            gold_idx = ent_idx
                            break
                    gold_seq.append(gold_idx)
            else:
                gold_seq = None
            
            feat_seq_logprobs = self.beam_search(doc, X, ent_ids, gold_seq=gold_seq)
            new_X, func_grad, pred_seq, _ = self.compute_func_grad(feat_seq_logprobs, y)

            # in KB acc
            match_cnt = np.sum(y[np.array(pred_seq, dtype='int32')])
            gold_cnt = len(gold_ids)

            result.append((new_X, func_grad, match_cnt, gold_cnt))
        q.put(result)


    # EVALUATION
    def get_acc(self, X, y, indices, ent_ids):
        batches = self.split(indices, self.num_thread)
        q = Queue()
        procs = []
        for batch in batches:
            proc = Process(target=self.get_func_grad, args=(batch,
                X, y, ent_ids, q, False))
            procs.append(proc)
            proc.start()
        result_list = []
        for _ in xrange(len(batches)):
            result_list.append(q.get())
        for proc in procs:
            proc.join()
        _, _, sum_match_cnt, sum_gold_cnt = self.merge(result_list)
        acc = float(sum_match_cnt*100)/sum_gold_cnt
        return acc
