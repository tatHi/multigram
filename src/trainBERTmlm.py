import train
import lm
from transformers import *
import argparse
from tqdm import tqdm
import util
import os
import mdp as dp
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data')
    parser.add_argument('-v', '--vocab')
    parser.add_argument('-me', '--maxEpoch', default=10, type=int)
    args = parser.parse_args()

    # set time stamp
    timeStamp = util.getTimeStamp()
    dirName = '%s-%s'%(timeStamp,
                       '-'.join(['%s=%s'%(str(k),
                                          str(v).replace('/','#')) for k,v in vars(args).items()]))
    os.mkdir('../results/'+dirName)

    data = [line.strip() for line in open(args.data)]
    print('>>> LOAD DATA')

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab)
    mlm = lm.MultigramLM(maxLength=1, minFreq=1, wordPiecePrefix='##', unkToken='[UNK]')
    mlm.setVocabFromBERTVocab(tokenizer.vocab)
    print('>>> INITIALIZE MLM WITH BERT TOKENIZER')

    mlm = train.EMTrain(mlm, data, args.maxEpoch, proning=False)

    #mlm.reIndex() # discard tokens whose probability is 0.0

    print('>>> FINISH TRAINING')

    #segData = [dp.viterbiIdSegmentation(line, mlm.makeLogProbTable(line)) for line in data]
    idTables = [mlm.makeIdTable(line, unkCharIdx=mlm.word2id[mlm.unkToken]) for line in data]
    segData = [[mlm.id2word[i] 
                        for i in dp.viterbiIdSegmentation(idTable,
                                                           mlm.makeLogProbTable(line, idTable=idTable))]
                for line, idTable in zip(data, idTables)]
    loglikelihood = np.log([mlm.theta[mlm.word2id[seg]] for segLine in segData for seg in segLine]).sum()
    print('log-likelihood:', loglikelihood/sum([len(segLine) for segLine in segData]))

    # dump
    with open('../results/%s/seg.txt'%dirName, 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = '../results/%s/lm.pickle'%dirName
    mlm.save(path)

    print('>>> DUMP RESULTS')

if __name__=='__main__':
    main()
