# MULTIGRAM LANGUAGE MODEL
Python implementation of [multigram language model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.6619&rep=rep1&type=pdf) for unsupervised word segmentation.
The system trains the language model with [Online EM algorithm](https://www.aclweb.org/anthology/N09-1069.pdf).

## Quick Start 
### with train.py
At the top of this repository.
```
$ python -m multigram.train -h
usage: train.py [-h] -d DATA [-td TESTDATA] [-me MAXEPOCH] [-ml MAXLENGTH] [-mf MINFREQ]
                [-os OUTPUTSUFFIX] [-tm {viterbi,viterbiStepWise,viterbiBatch,EM}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  text data for training
  -td TESTDATA, --testData TESTDATA
                        text data for checking log-prob as test. if not specified, use training
                        set
  -me MAXEPOCH, --maxEpoch MAXEPOCH
                        max training epoch (default: 10)
  -ml MAXLENGTH, --maxLength MAXLENGTH
                        maximum length of word (default: 5)
  -mf MINFREQ, --minFreq MINFREQ
                        minimum frequency of word (default: 50)
  -os OUTPUTSUFFIX, --outputSuffix OUTPUTSUFFIX
                        output is dumped as [timestamp]_suffix.pickle if suffix is given
                        otherwise [timestamp].pickle
  -tm {viterbi,viterbiStepWise,viterbiBatch,EM}, --trainMode {viterbi,viterbiStepWise,viterbiBatch,EM}
                        method to train multigram language model (default: EM)
```


For example, run the following command:
```
$ python -m multigram.train -d [path to training corpus]
```
then, you obtain the learned language model (lm.pickle) and estimated word segmentation (seg.txt) in `results/[timestamp]/`.

### as a package
```
$ python setup.py develop
$ python
>>> from multigram import lm
>>> from multigram import train
>>> maxEpoch = 20
>>> data = [line.strip() for line in open('path/to/text')]
>>> mlm = lm.MultigramLM(data=data)
>>> mlm = train.EMTrain(mlm=mlm, data=data, maxEpoch=20)
>>> mlm.save('path/to/output')
```

## Estimate Probabilities for BERT
This module is for estimating probabilities of WordPieces distributed as a part of BERT.
We only support `transformers` published by HuggingFace.

```
$ pip install transformers
$ python -m multigram.trainBERTvocab -h
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

