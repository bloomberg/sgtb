# Structured Gradient Tree Boosting

Author: Yi Yang

Contact: yyang464@bloomberg.net


## Basic description

This is the Python implementation of the structured gradient tree boosting model 
for collective named entity disambiguation, described in

    Yi Yang, Ozan Irsoy, and Kazi Shefaet Rahman 
    "Collective Entity Disambiguation with Structured Gradient Tree Boosting"
    NAACL 2018

[[pdf]](https://arxiv.org/pdf/1802.10229.pdf)

BibTeX

    @inproceedings{yang2018collective,
      title={Collective Entity Disambiguation with Structured Gradient Tree Boosting},
      author={Yang, Yi and Irsoy, Ozan and Rahman, Kazi Shefaet},
      booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      volume={1},
      pages={777--786},
      year={2018}
    }



## Data

The preprocessed [AIDA-CoNLL](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/)
data ('AIDA-PPR-processed.json') is available in the [data](data) folder:
* The entity candidates are generated based on the [PPRforNED](https://github.com/masha-p/PPRforNED) candidate
generation system.
* The system uses 19 local features, including 3 prior features, 4 NER features,
2 entity popularity features, 4 entity type features, and 6 context features. 
Please look into the paper for details.

The system also uses entity-entity features, which can be quickly computed
on-the-fly. Here, we provide pre-computed entity-entity features (3 features
per entity pair) for the AIDA-CoNLL dataset, which is available in the 
[data](data) folder ('ent_ent_feats.txt.gz').


## Reproduce results

You can reproduce the SGTB-BSG results by running:
```
python structured_learner.py --num-thread=16 --num-epoch=250
```
I got 95.32 accuracy on the test dataset. Training took 35 min on 16 threads.
