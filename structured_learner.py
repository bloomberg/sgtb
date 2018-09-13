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

"""
Named entity disambiguation by Structured Gradient Tree Boosting.
"""

# Author: Yi Yang
# Email: yyang464@bloomberg.net 
#        yiyangnlp@gmail.com

import argparse, json, time, gzip
import numpy as np

from structured_gradient_boosting import StructuredGradientBoosting

def train_model(train_docs, dev_docs, test_docs, ent_ent_feat_dict, params):
    print ("PARAM SETTINGS")
    print (params)
    train_set = make_idx_data(train_docs, params['num_candidate'], skip=True)
    dev_set = make_idx_data(dev_docs, params['num_candidate'])
    test_set = make_idx_data(test_docs, params['num_candidate'])

    clf = StructuredGradientBoosting(max_depth=params['max_depth'],
                                     learning_rate=params['learning_rate'],
                                     n_estimators=params['n_estimators'],
                                     min_samples_split=params['min_samples_split'],
                                     min_samples_leaf=params['min_samples_leaf'],
                                     ent_ent_feat_dict=ent_ent_feat_dict,
                                     beam_width=params['beam_width'],
                                     num_thread=params['num_thread'])
    
    start_time = time.time()
    clf = clf.fit(train_set, dev_set)
    end_time = time.time()
    print ("Training take %.2f secs" %(end_time - start_time))

    test_X, test_y, test_indices, test_ent_ids = test_set
    test_acc = clf.get_acc(test_X, test_y, test_indices, test_ent_ids)
    print ("Test acc %.2f" %(test_acc))


def make_idx_data(docs, ncand=30, skip=False):
    """
    Convert data to fit sklearn regression trees.

    Inputs
    -------
        docs: a document list '[[[(mention_str, offset, wikiID), [(entity, label), [feature]]]]]'
        ncand: number of entity candidates for a mention
        skip: whether to skip mentions whose gold entities are not in candidates
              (used for training data only)
        
    Outputs
    -------
        X: a local feature matrix, label and mention indices arraries
        y: a label array
        indices: a list of pair '(a list of mention indices, a list of gold entity ids)' 
        ent_ids: wikiID for entities (used for querying entity-entity features)
    """
    X, y, indices, ent_ids = [], [], [], []
    i = 0
    for doc in docs:
        doc_idx = []
        gold_ids, skip_ids = [], [] 
        for mentcand in doc:
            ment_idx = []
            flag = False
            tX, ty, tids = [], [], []
            for entcand in mentcand[1][:ncand]:
                tX.append(entcand[1])
                ty.append(entcand[0][1])
                if ty[-1] == 1: flag = True
                tids.append(entcand[0][0])
                ment_idx.append(i)
                i += 1
            if skip and not flag:
                i = len(y)
                continue
            else:
                X += tX
                y += ty
                ent_ids += tids
            if len(ment_idx) > 0: 
                doc_idx.append(ment_idx)
                gold_ids.append(mentcand[0][-1])
            else: # must be a false negative
                skip_ids.append(mentcand[0][-1]) 
        if len(doc_idx) > 0: 
            # append skip_ids after gold_ids, in order to properly evaluate
            # note len(doc_idx) != len(gold_ids+skip_ids)
            indices.append((doc_idx, gold_ids+skip_ids))
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')
    return X, y, indices, ent_ids

def main():
    parser = argparse.ArgumentParser(description="""Named entity disambiguation with 
                                                    Structured Gradient Tree Boosting""")
    parser.add_argument('--dataset', type=str, default="data/AIDA-PPR-processed.json",
                        help="""Processed dataset file in json format. A document is represented as 
                                '[[[(mention, offset, goldEntity), [(entity,label), [feature]]]]]'""")
    parser.add_argument('--train-dev-split-idx', type=int, default=946,
                        help="Number of training instances.")
    parser.add_argument('--dev-test-split-idx', type=int, default=1162,
                        help="Number of training and development instances.")
    parser.add_argument('--num-candidate', type=int, default=30,
                        help="Number of entity candidates for each mention.")
    parser.add_argument('--entity-features', type=str, default="data/ent_ent_feats.txt.gz",
                        help="""Pre-computed feature vectors for entity-entity pairs.
                                Format: 'ent1 ent2<TAB>feat1 feat2 feat3'""")
    parser.add_argument('--num-epoch', type=int, default=500,
                        help="Number of iterations, aka, number of ensembled trees.")
    parser.add_argument('--beam-width', type=int, default=4,
                        help="Beam width, used by beam search.")
    parser.add_argument('--learning-rate', type=float, default=1.,
                        help="Learning rate. It is fixed to 1.")
    parser.add_argument('--max-depth', type=int, default=3,
                        help="Maximum depth of a regression tree.")
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help="Minimum samples required for splitting a node.")
    parser.add_argument('--min-samples-leaf', type=int, default=1,
                        help="Minimum instances required to be a leaf node.")
    parser.add_argument('--num-thread', type=int, default=8,
                        help="SGTB can be easily parallelized. Number of threads.")
    args = parser.parse_args()

    params = {
              'n_estimators'      : args.num_epoch, 
              'beam_width'        : args.beam_width,
              'learning_rate'     : args.learning_rate,
              'max_depth'         : args.max_depth, 
              'min_samples_split' : args.min_samples_split,
              'min_samples_leaf'  : args.min_samples_leaf,
              'num_candidate'     : args.num_candidate,
              'num_thread'        : args.num_thread
             }

    # data
    processed_docs = []
    with open(args.dataset, 'rb') as f:
        for line in f:
            processed_docs.append(json.loads(line.strip()))
    train_docs, dev_docs, test_docs = processed_docs[ : args.train_dev_split_idx],\
                processed_docs[args.train_dev_split_idx : args.dev_test_split_idx],\
                processed_docs[args.dev_test_split_idx : ]

    # entity-entity features
    ent_ent_feat_dict = {}
    with gzip.open(args.entity_features, 'rb') as f:
        for line in f:
            ep, feat_str = line.split('\t')
            e1, e2 = ep.split()
            feats = map(float, feat_str.split()) 
            ent_ent_feat_dict[(e1,e2)] = feats
 
    # train and evaluate
    train_model(train_docs, dev_docs, test_docs, ent_ent_feat_dict, params)


if __name__ == '__main__':
    main()

