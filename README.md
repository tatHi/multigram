# MULTIGRAM LANGUAGE MODEL
Python implementation of [multigram language model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=479391) for unsupervised word segmentation.
The system trains the language model with stepwise updation introduced by [Online EM algorithm](https://www.aclweb.org/anthology/N09-1069.pdf) in addition to the default EM updation.

# Requirements
```
tqdm==4.28.1
numpy==1.18.0
numba==0.48.0
sentencepiece==0.1.85
scipy==1.2.1
PyYAML==5.3.1
scikit_learn==0.23.2
(transformers==2.8.0, if you need to use trainBERTvocab.py)
```

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
  -rd RESULTDIR, --resultDir RESULTDIR
                        dir to output (default: ./)
```


For example, run the following command:
```
$ python -m multigram.train -d [path to training corpus]
```
then, you obtain the learned language model (lm.pickle) and estimated word segmentation (seg.txt) in `[RESULTDIR]/[timestamp](_[OUTPUTSUFFIX])/`.

### as a package
```
$ pip install --editable .
$ python
>>> from multigram import lm
>>> from multigram import train
>>> maxEpoch = 20
>>> data = [line.strip() for line in open('path/to/text')]
>>> mlm = lm.MultigramLM(data=data)
>>> mlm = train.EMTrain(mlm=mlm, data=data, maxIter=maxEpoch)
>>> mlm.save('path/to/output')
```

## Training Strategies
We prepared 4 training methods:
- EM and Viterbi
  - The default training schema introduced by the original [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.6619&rep=rep1&type=pdf).
  - They correspond to Eq.(5) and Eq.(6) of the paper, respectively.
  - Viterbi training is the fastest method for learning.
- Viterbi Batch and Viterbi Stepwise
  - Enhance the Viterbi training to using batch-wise and stepwise updation introduced by this [paper](https://www.aclweb.org/anthology/N09-1069.pdf).

## Estimate Probabilities for BERT
This module is for estimating probabilities of WordPieces distributed as a part of BERT.
We only support `transformers` published by [HuggingFace](https://github.com/huggingface/transformers).

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
  -rd RESULTDIR, --resultDir RESULTDIR
                        dir to output (default: ./)
```

# Use Trained Model as Tokenizer
```
>>> from multigram import lm, tokenizer
>>> mlm = lm.MultigramLM()
>>> mlm.load('/path/to/trained/model/lm.pickle')
# you can also load sentencepiece model as:
>>> mlm.loadSentencePieceModel('path/to/sentencepiece.model')
>>> tk = tokenizer.Tokenizer(mlm)

# you can use tk as SentencePieceProcessor in almost all cases.
>>> sent = 'hello world'
>>> tk.encode_as_pieces(sent) # viterbi tokenization
>>> tk.sample_encode_as_pieces(sent, -1, 0.2) # sampling tokenization with FFBS and alpha=0.2
```

