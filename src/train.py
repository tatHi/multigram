import lm
import mdp as dp
from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm
import util
import os

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
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        iterTheta = np.zeros(mlm.theta.shape)

        for b in range(0, len(data), batchSize):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            
            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [w
                       for line in lines
                       for w in dp.viterbiSegmentation(line,
                                                       mlm.makeLogProbTable(line))]

            # re-estimate
            batchTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                batchTheta[mlm.word2id[k]] = v
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

    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        indices = np.random.permutation(len(data)) if shuffle else np.arange(len(data))

        for b in range(0, len(data), batchSize):
            if len(data)-b < batchSize*0.9:
                # if the number of contents is less than 90% of batchSize, break 
                break
            
            lines = [data[indices[b]] for b in range(b, b+batchSize) if b<len(data)]

            # viterbi
            tmpSegs = [w
                       for line in lines
                       for w in dp.viterbiSegmentation(line,
                                                       mlm.makeLogProbTable(line))]

            # re-estimate
            eta = (step+2)**(-decay)
            step += 1

            currentTheta = np.zeros(mlm.theta.shape)
            tmpVocabSize = len(tmpSegs)
            tmpUnigramCount = Counter(tmpSegs)
            for k,v in tmpUnigramCount.items():
                currentTheta[mlm.word2id[k]] = v
            currentTheta = currentTheta / tmpVocabSize

            # update
            mlm.theta = (1-eta)*mlm.theta + eta*currentTheta

            # proning
            if proning: mlm.proneVocab()
        
    return mlm

def viterbiTrain(mlm, data, maxIter=10, proning=True):
    print('>>> START VITERVI %d EPOCH TRAINING'%(maxIter))

    prevLH = 0
    for it in range(maxIter):
        print('iter: %d/%d'%(it+1, maxIter))
        if it==10:
            # vocabulary pruning
            pass

        # viterbi
        tmpSegs = [w
                   for line in data
                   for w in dp.viterbiSegmentation(line,
                                                   mlm.makeLogProbTable(line))]

        # calc loglikelihood
        loglikelihood = np.log([mlm.theta[mlm.word2id[w]] for w in tmpSegs]).sum()
        print('current log-likelihood:', loglikelihood/len(tmpSegs))

        # re-estimate
        tmpVocabSize = len(tmpSegs)
        tmpUnigramCount = Counter(tmpSegs)
        currentTheta = np.zeros(mlm.theta.shape)
        for k,v in tmpUnigramCount.items():
            currentTheta[mlm.word2id[k]] = v
        currentTheta = currentTheta / tmpVocabSize

        # re-normalize
        mlm.theta = currentTheta

        # proning
        if proning: mlm.proneVocab()

        print(tmpSegs[100:200])        

    return mlm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data')
    parser.add_argument('-td', '--testData', default=None)
    parser.add_argument('-me', '--maxEpoch', default=10, type=int)
    parser.add_argument('-ml', '--maxLength', default=5, type=int)
    parser.add_argument('-mf', '--minFreq', default=50, type=int)
    args = parser.parse_args()

    # set time stamp
    timeStamp = util.getTimeStamp()
    dirName = '%s-%s'%(timeStamp,
                       '-'.join(['%s=%s'%(str(k),
                                          str(v).replace('/','#')) for k,v in vars(args).items()]))
    os.mkdir('../results/'+dirName)

    data = [line.strip() for line in open(args.data)]
    print('>>> LOAD DATA')
    mlm = lm.MultigramLM(maxLength=args.maxLength, minFreq=args.minFreq, data=data)

    mlm = EMTrain(mlm, data, args.maxEpoch)
   
    mlm.reIndex() # discard tokens whose probability is 0.0

    print('>>> FINISH TRAINING')

    # inference
    if args.testData is not None:
        data = [line.strip() for line in open(args.testData)]
        
    segData = [dp.viterbiSegmentation(line, mlm.makeLogProbTable(line)) for line in data]
    loglikelihood = np.log([mlm.theta[mlm.word2id[seg]] for segLine in segData for seg in segLine]).sum()
    print('log-likelihood:', loglikelihood/sum([len(segLine) for segLine in segData]))

    # dump
    with open('../results/%s/seg.txt'%dirName, 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = '../results/%s/lm.pickle'%dirName
    mlm.save(path)
    
    print('>>> DUMP RESULTS')


if __name__ == '__main__':
    main()
