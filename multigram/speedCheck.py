import lm as mlm
from tqdm import tqdm
import mdp

mlmPath = '/home/hiraoka.t/work/emSegmentation/multigram/results/20200208174303-aiuqaszwgw-data=#home#hiraoka.t#work#data#twitter_ja_all#train.text-testData=None-maxEpoch=20-maxLength=8-minFreq=50/lm.dict'
lm = mlm.MultigramLM()
lm.load(mlmPath)

lm.addWordToVocab('<unk>')
lm.unkToken = '<unk>'

dataPath = '/home/hiraoka.t/work/data/twitter_ja_all/train.text'
data = [line.strip() for line in open(dataPath)]

print('idTables')
idTables = [lm.makeIdTable(line) for line in tqdm(data)]

print('logProbTables')
logProbTables = [lm.makeLogProbTable(line) for line in tqdm(data)]

for idTable, logProbTable in tqdm(zip(idTables, logProbTables)):
    mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, m=5, n=5, mode='astar')

