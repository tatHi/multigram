# MULTIGRAM LANGUAGE MODEL
Python implementation of [multigram language model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.6619&rep=rep1&type=pdf) for unsupervised word segmentation.
The system trains the language model with [Online EM algorithm](https://www.aclweb.org/anthology/N09-1069.pdf).

## Quick Start
```
$ python train.py -h
usage: train.py [-h] -d DATA [-td TESTDATA] [-me MAXEPOCH] [-ml MAXLENGTH]
                [-mf MINFREQ] [-os OUTPUTSUFFIX]
                [-tm {viterbi,viterbiStepWise,viterbiBatch,EM}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  text data for training
  -td TESTDATA, --testData TESTDATA
                        text data for checking log-prob as test. if not
                        specified, use training set
  -me MAXEPOCH, --maxEpoch MAXEPOCH
                        max training epoch
  -ml MAXLENGTH, --maxLength MAXLENGTH
                        maximum length of word
  -mf MINFREQ, --minFreq MINFREQ
                        minimum frequency of word
  -os OUTPUTSUFFIX, --outputSuffix OUTPUTSUFFIX
                        output is dumped as timestamp_suffix.pickle if suffix
                        is given otherwise timestamp.pickle
  -tm {viterbi,viterbiStepWise,viterbiBatch,EM}, --trainMode {viterbi,viterbiStepWise,viterbiBatch,EM}
                        method to train multigram language modelt
```


For example, run the following command:
```
$ python train.py -d [path to training corpus]
```
then, you obtain the learned language model (lm.pickle) and estimated word segmentation (seg.txt) in `results/[timestamp]/`.


## Estimate Probabilities for BERT
```
$ python trainBERTvocab.py -h
usage: trainBERTvocab.py [-h] -d DATA -p PRETRAIN [-me MAXEPOCH]
                         [-os OUTPUTSUFFIX]
                         [-tm {viterbi,viterbiStepWise,viterbiBatch,EM}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data for training lm
  -p PRETRAIN, --pretrain PRETRAIN
                        pretrained shortcut, such as bert-base-cased
  -me MAXEPOCH, --maxEpoch MAXEPOCH
                        maximum training epoch
  -os OUTPUTSUFFIX, --outputSuffix OUTPUTSUFFIX
                        output is dumped as timestamp_suffix.pickle if suffix
                        is given otherwise timestamp.pickle
  -tm {viterbi,viterbiStepWise,viterbiBatch,EM}, --trainMode {viterbi,viterbiStepWise,viterbiBatch,EM}
                        method to train multigram language model
```

## Use as pip package
To use our implementation in other projects, the useful way is running `setup.py` to call multigram from anywhere.
```
$ python setup.py develop
```
