import train
import lm
from transformers import *
import argparse
from tqdm import tqdm
import util
import os
import mdp as dp
import numpy as np
import yaml

RESULTS_DIR = '../results'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', 
                        '--data',
                        required=True,
                        help='data for training lm')
    parser.add_argument('-p', 
                        '--pretrain',
                        required=True,
                        help='pretrained shortcut, such as bert-base-cased')
    parser.add_argument('-me', 
                        '--maxEpoch', 
                        default=10, 
                        type=int,
                        help='maximum training epoch')
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

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrain)
    mlm = lm.MultigramLM(maxLength=1, minFreq=1, wordPiecePrefix='##', unkToken='[UNK]')
    mlm.setVocabFromBERTVocab(tokenizer.vocab)
    print('>>> INITIALIZE MLM WITH BERT TOKENIZER')

    if args.trainMode=='EM':
        mlm = train.EMTrain(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbi':
        mlm = train.viterbiTrain(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbiStepWise':
        mlm = train.viterbiTrainStepWise(mlm, data, args.maxEpoch, proning=False)
    elif args.trainMode=='viterbiBatch':
        mlm = train.viterbiTrainBatch(mlm, data, args.maxEpoch, proning=False)

    # reindexing is not required because word ids should be match b/w bert and mlm

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
    with open(os.path.join(RESULTS_DIR, dirName, 'seg.txt'), 'w') as f:
        for segLine in segData:
            f.write(' '.join(segLine)+'\n')

    path = os.path.join(RESULTS_DIR, dirName, 'lm.pickle')
    mlm.save(path)
    print('>>> DUMP RESULTS')

if __name__=='__main__':
    main()
