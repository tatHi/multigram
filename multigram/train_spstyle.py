# SentencePieceと同様に，指定のvocabサイズまでEM-proningを繰り返す
from . import lm
from . import mdp as dp
from . import train
from . import util
from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import os

def createTokenizer(args, data):
    mlm = lm.MultigramLM(maxLength=args.maxLength, minFreq=args.minFreq, data=data)

    if args.trainMode=='EM':
        trainfunc = train.EMTrain
    elif args.trainMode=='EMMT':
        trainfunc = train.EMTrainMultiThread
    elif args.trainMode=='viterbi':
        trainfunc = train.viterbiTrain
    elif args.trainMode=='viterbiStepWise':
        trainfunc = train.viterbiTrainStepWise
    elif args.trainMode=='viterbiBatch':
        trainfunc = train.viterbiTrainBatch

    step = 1
    while len(mlm.vocab) > args.vocabSize:
        print('STEP %d: START'%step)
        mlm = trainfunc(mlm, data, args.maxEpoch, proning=False)
        size = max(args.vocabSize, int(len(mlm.vocab) * args.shrinkRatio))
        mlm.shrinkVocab(size)
        mlm.reIndex() # discard tokens whose probability is 0.0
        mlm.randomizeTheta()
        print('STEP %d: END (VOCABSIZE=%d)'%(step, len(mlm.vocab)))
        step += 1
    
    print('final em training...')
    mlm = trainfunc(mlm, data, args.maxEpoch, proning=False)
    return mlm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='text data for training')
    parser.add_argument('-td', 
                        '--testData', 
                        default=None,
                        type=str,
                        help='text data for checking log-prob as test. if not specified, use training set')
    parser.add_argument('-me', 
                        '--maxEpoch', 
                        default=2, 
                        type=int,
                        help='max training epoch of each training step (default: 10)')
    parser.add_argument('-ml', 
                        '--maxLength', 
                        default=5, 
                        type=int,
                        help='maximum length of word (default: 5)')
    parser.add_argument('-mf', 
                        '--minFreq', 
                        default=3, 
                        type=int,
                        help='minimum frequency of word (default: 3)')
    parser.add_argument('-os',
                        '--outputSuffix',
                        default='',
                        type=str,
                        help='output is dumped as [timestamp]_suffix.pickle if suffix is given otherwise [timestamp].pickle')
    parser.add_argument('-tm',
                        '--trainMode',
                        default='EM',
                        choices=['viterbi',
                                 'viterbiStepWise',
                                 'viterbiBatch',
                                 'EM',
                                 'EMMT'],
                        help='method to train multigram language model (default: EM)')
    parser.add_argument('-vs', 
                        '--vocabSize', 
                        default=32000, 
                        type=int,
                        help='vocabulary size (default 32000)')
    parser.add_argument('-rd',
                        '--resultDir',
                        default='./',
                        help='dir to output (default: ./)')
    parser.add_argument('-sr', 
                        '--shrinkRatio', 
                        default=0.8, 
                        type=float,
                        help='ratio for shrinking in each proning step (default: 0.8, which means 20\% of vocab are discarded in each step)')
    args = parser.parse_args()

    # make results dir
    if not os.path.isdir(args.resultDir):
        os.makedirs(args.resultDir)
        print('>>> CREATE RESULTS DIR')

    # set time stamp
    timeStamp = util.getTimeStamp()
    dirName = timeStamp + ('_' + args.outputSuffix if args.outputSuffix else '')
    os.mkdir(os.path.join(args.resultDir, dirName))
    
    # dump config
    setattr(args, 'dirName', dirName)
    open(os.path.join(args.resultDir, dirName, 'config.yaml'), 'w').write(yaml.dump(vars(args)))

    # load data
    data = [line.strip() for line in open(args.data)]
    print('>>> LOAD DATA')

    # training
    print('>>> START TRAINING')
    mlm = createTokenizer(args, data)
    print('>>> FINISH TRAINING')

    # inference
    if args.testData is not None:
        print('>>> INFERENCE ON TEST DATA')
        data = [line.strip() for line in open(args.testData)]
    else:
        print('>>> INFERENCE ON TRAIN DATA')
        
    segData = [dp.viterbiSegmentation(line, mlm.makeLogProbTable(line)) for line in data]
    loglikelihood = np.log([mlm.theta[mlm.word2id[seg]] for segLine in segData for seg in segLine]).sum()
    print('log-likelihood:', loglikelihood/sum([len(segLine) for segLine in segData]))

    # dump
    with open(os.path.join(args.resultDir, dirName, 'seg.txt'), 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = os.path.join(args.resultDir, dirName, 'lm.pickle')
    mlm.save(path)
    print('>>> DUMP RESULTS')

if __name__ == '__main__':
    main()
