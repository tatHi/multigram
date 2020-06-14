from . import lm
from . import mdp as dp
from . import util
from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import os

RESULTS_DIR = '../results'

def EMTrain(mlm, data, maxIter=10, proning=True):
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        iterTheta = np.zeros(mlm.theta.shape)
        for j,line in enumerate(tqdm(data)):
            if len(line)==0: continue

            if it==0:
                unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
                idTable = mlm.makeIdTable(line, unkCharIdx=unkid)
                idTables.append(idTable)
            else:
                idTable = idTables[j]
            logProbTable = mlm.makeLogProbTable(line, idTable=idTable)
            
            # dp
            alpha, sumAlpha = dp.calcAlpha(logProbTable)
            sumBeta = dp.calcBeta(logProbTable)
            sumGamma = dp.calcGamma(logProbTable, alpha, sumAlpha)
            
            # posterior
            posterior = dp.calcPosterior(alpha, sumBeta, sumGamma)
            
            # update
            idx = np.where(idTable!=-1)
            iterTheta[idTable[idx]] += posterior[idx]

        # re-normalize
        iterTheta = iterTheta / sum(iterTheta)

        # update
        mlm.theta = iterTheta

        # proning
        if proning: mlm.proneVocab()
        
        tmpSegs = [mlm.id2word[i] for i in dp.viterbiIdSegmentation(idTable,
                                                 mlm.makeLogProbTable(data[0], idTable=idTable))]
        print(' '.join(tmpSegs))

    return mlm

def viterbiTrainBatch(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERVI %d EPOCH TRAINING'%(maxIter))
    batchSize = 256
    shuffle = True
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        iterTheta = np.zeros(mlm.theta.shape)
        
        if it==0:
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in data]

        for b in range(0, len(data), batchSize):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            

            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]
            tables = [idTables[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [i
                       for line, idTable in zip(data, idTables)
                       for i in dp.viterbiIdSegmentation(idTable,
                                                         mlm.makeLogProbTable(line, idTable=idTable))]

            # re-estimate
            batchTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                batchTheta[k] = v
            batchTheta = batchTheta / tmpVocabSize

            iterTheta += batchTheta

        # re-normalize
        iterTheta = iterTheta / sum(iterTheta)

        # update
        mlm.theta = iterTheta

        # proning
        if proning: mlm.proneVocab()
        
    return mlm

def viterbiTrainStepWise(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERVI %d EPOCH TRAINING'%(maxIter))

    decay = 0.8
    batchSize = 256
    shuffle = True

    step = 0

    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        if it==0:
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in data]

        for b in range(0, len(data), batchSize):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            
            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]
            tables = [idTables[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [i
                       for line, idTable in zip(data, idTables)
                       for i in dp.viterbiIdSegmentation(idTable,
                                                         mlm.makeLogProbTable(line, idTable=idTable))]

            # re-estimate
            eta = (step+2)**(-decay)
            step += 1

            currentTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                currentTheta[k] = v
            currentTheta = currentTheta / tmpVocabSize

            # update
            mlm.theta = (1-eta)*mlm.theta + eta*currentTheta

        # proning
        if proning: mlm.proneVocab()
        
    return mlm

def viterbiTrain(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERVI %d EPOCH TRAINING'%(maxIter))

    prevLH = 0
    idTables = []
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))

        if it==0:
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            unkid = mlm.word2id[mlm.unkToken] if mlm.unkToken else -1
            idTables = [mlm.makeIdTable(line, unkCharIdx=unkid) for line in data]

        # viterbi
        tmpSegs = [i
                   for line, idTable in zip(data, idTables)
                   for i in dp.viterbiIdSegmentation(idTable,
                                                     mlm.makeLogProbTable(line, idTable=idTable))]

        # calc loglikelihood
        loglikelihood = np.log([mlm.theta[i] for i in tmpSegs]).sum()
        print('current log-likelihood:', loglikelihood/len(tmpSegs))

        # re-estimate
        tmpVocabSize = len(tmpSegs)
        tmpUnigramCount = Counter(tmpSegs)
        currentTheta = np.zeros(mlm.theta.shape)
        for k,v in tmpUnigramCount.items():
            currentTheta[k] = v
        currentTheta = currentTheta / tmpVocabSize

        # re-normalize
        mlm.theta = currentTheta

        # proning
        if proning: mlm.proneVocab()

        print(' '.join([mlm.id2word[i] for i in tmpSegs[:100]]))        

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
                        default=10, 
                        type=int,
                        help='max training epoch')
    parser.add_argument('-ml', 
                        '--maxLength', 
                        default=5, 
                        type=int,
                        help='maximum length of word')
    parser.add_argument('-mf', 
                        '--minFreq', 
                        default=50, 
                        type=int,
                        help='minimum frequency of word')
    parser.add_argument('-os',
                        '--outputSuffix',
                        default='',
                        type=str,
                        help='output is dumped as timestamp_suffix.pickle if suffix is given otherwise timestamp.pickle')
    parser.add_argument('-tm',
                        '--trainMode',
                        default='EM',
                        choices=['viterbi',
                                 'viterbiStepWise',
                                 'viterbiBatch',
                                 'EM'],
                        help='method to train multigram language model')
    args = parser.parse_args()

    # make results dir
    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
        print('>>> CREATE RESULTS DIR')

    # set time stamp
    timeStamp = util.getTimeStamp()
    dirName = '_'.join([timeStamp, args.outputSuffix])
    os.mkdir(os.path.join(RESULTS_DIR, dirName))
    
    # dump config
    setattr(args, 'dirName', dirName)
    open(os.path.join(RESULTS_DIR, dirName, 'config.yaml'), 'w').write(yaml.dump(vars(args)))

    # load data
    data = [line.strip() for line in open(args.data)]
    print('>>> LOAD DATA')

    # training
    print('>>> START TRAINING')
    mlm = lm.MultigramLM(maxLength=args.maxLength, minFreq=args.minFreq, data=data)

    if args.trainMode=='EM':
        mlm = EMTrain(mlm, data, args.maxEpoch)
    elif args.trainMode=='viterbi':
        mlm = viterbiTrain(mlm, data, args.maxEpoch)
    elif args.trainMode=='viterbiStepWise':
        mlm = viterbiTrainStepWise(mlm, data, args.maxEpoch)
    elif args.trainMode=='viterbiBatch':
        mlm = viterbiTrainBatch(mlm, data, args.maxEpoch)
    
    mlm.reIndex() # discard tokens whose probability is 0.0
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
    with open(os.path.join(RESULTS_DIR, dirName, 'seg.txt'), 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = os.path.join(RESULTS_DIR, dirName, 'lm.pickle')
    mlm.save(path)
    print('>>> DUMP RESULTS')

if __name__ == '__main__':
    main()
