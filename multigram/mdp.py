import numpy as np
from scipy.special import logsumexp
import random
from multiprocessing import Pool
import multiprocessing as multi
from time import time

import numba as nb
from numba import jit, f8, i8, u1, b1
from numba.typed import List

minf = float('-inf')

def calcAlpha(logProbTable):
    T, L = logProbTable.shape

    ### A
    alpha = np.zeros((T, L))
    sumAlpha = np.zeros(T)

    for t in range(T):
        for l in range(L):
            prev = sumAlpha[t-l-1] if 0 <= t-l-1 else 0
            alpha[t,l] = logProbTable[t,l] + prev
        sumAlpha[t] = logsumexp(alpha[t])
    #print(alpha)
    #print(sumAlpha)

    return alpha, sumAlpha

def calcBeta(logProbTable):
    T, L = logProbTable.shape
    beta = np.full((T, L), float('-inf'))
    sumBeta = np.zeros(T+1)

    for t in range(T)[::-1]:
        for l in range(L):
            if T < t+l+1:
                break
            #prev = sumBeta[t+l+1] if t+l+1<beta.shape[0] else 0
            prev = sumBeta[t+l+1]
            beta[t,l] = logProbTable[t+l,l] + prev
        sumBeta[t] = logsumexp(beta[t])

    return sumBeta

def calcGamma(logProbTable, alpha, sumAlpha):
    gamma = np.zeros(alpha.shape)
    sumGamma = np.zeros(alpha.shape[0])

    for t in range(gamma.shape[0]):
        for l in range(gamma.shape[1]):
            if t-l-1 < 0:
                prev = 0
            else:
                prev = sumGamma[t-l-1]
            gamma[t,l] = prev * np.exp(alpha[t,l] - sumAlpha[t])
        sumGamma[t] = 1 + sum(gamma[t])
    return np.log(sumGamma)

def calcPosterior(alpha, sumBeta, sumGamma):
    ab = alpha + sumBeta[1:].reshape(sumBeta.shape[0]-1, 1)
    bg = sumBeta[0]+sumGamma[-1]
    return np.exp(ab-bg)

def forwardFiltering(logProbTable):
    T, L = logProbTable.shape
    alpha, sumAlpha = calcAlpha(logProbTable)
    dists = alpha - sumAlpha.reshape(T,1)
    dists = np.exp(dists)
    return dists, T, L
    
def backwardSampling(dists, T, L):
    ls = []
    p = T-1

    while 0<=p:
        #dist = np.exp(alpha[p] - sumAlpha[p])
        #dist = dist[:min(alpha.shape[1], p+1)]
        lSize = min(L, p+1)
        dist = dists[p, :lSize]
        l = random.choices(range(1,lSize+1), weights=dist, k=1)       
        #l = sampledLs[p]+1
        #ls += [l]
        ls += l
        #l = random.choices(range(lSize), weights=dist, k=1)[0]        
        #l = l+1
        #ls.append(l)
        p -= l[0]
    ls = ls[::-1]
    return ls

def ffbs(logProbTable, n=1):
    dists, T, L = forwardFiltering(logProbTable)
    lss = [backwardSampling(dists, T, L) for m in range(n)]
    return lss

@jit(nb.types.Tuple((f8[:], i8[:]))(f8[:,:]), nopython=True)
def viterbiForward(logProbTable):
    size = logProbTable.shape[0]
    maxScores = np.zeros(size, dtype=np.float64)
    maxLs = np.zeros(size, dtype=np.int64)

    # forward
    for t in range(size):
        lpList = np.zeros(logProbTable.shape[1], dtype=np.float64)
        for l in range(logProbTable.shape[1]):
            lpList[l] = logProbTable[t,l] + (maxScores[t-l-1] if 0<=t-l-1 else 0)
        maxScores[t] = max(lpList)
        maxLs[t] = lpList.argmax()
    return maxScores, maxLs

def viterbi(logProbTable):
    maxScores, maxLs = viterbiForward(logProbTable)
    
    # backward
    ls = []
    t = len(maxLs)-1
    while 0<=t:
        l = maxLs[t] + 1
        ls.append(l)
        t -= l
    ls = ls[::-1]

    return ls

def tokenizeByLength(line, ls):
    segs = ['']*len(ls)
    pointer = 0
    for i,l in enumerate(ls):
        #segs.append(line[pointer:pointer+l])
        segs[i] = line[pointer:pointer+l]
        pointer += l
    return segs

def tokenize(line, logProbTable, sampling):
    ls = ffbs(logProbTable, 1)[0] if sampling else viterbi(logProbTable)
    segs = tokenizeByLength(line, ls)
    return segs    

def samplingSegmentation(line, logProbTable):
    return tokenize(line, logProbTable, sampling=True)

def samplingIdSegmentation(idTable, logProbTable, n=1):
    ls = ffbs(logProbTable, n)[0]
    ids = getIds(idTable, ls)
    return ids

def wrapSamplingIdSegmentation(xs):
    return samplingIdSegmentation(*xs)

def samplingIdSegmentationMultiProc(idTables, logProbTables):
    p = Pool(multi.cpu_count())
    segIds = p.map(wrapSamplingIdSegmentation, zip(idTables, logProbTables))
    p.close()
    return segIds

def viterbiSegmentation(line, logProbTable):
    return tokenize(line, logProbTable, sampling=False)

def viterbiIdSegmentation(idTable, logProbTable):
    ls = viterbi(logProbTable)
    c = 0
    ids = []
    for l in ls:
        c += l
        ids.append(idTable[c-1, l-1])
    return ids

def nbestSegmentation(line, logProbTable, n, mode='astar'):
    #print(line)
    #print(logProbTable)

    if mode=='astar':
        # forward
        maxScores, _ = viterbiForward(logProbTable)
        # backward
        segs = [tokenizeByLength(line, ls) for ls, _ in nbestAstarBakckward(maxScores, logProbTable, n=n)]
        return segs
    elif mode=='point':
        # viterbi
        bestSeg = viterbi(logProbTable)
        # point estimation
        lss = nbestPointEstimation(bestSeg, logProbTable, n=n)
    else:
        print('mode should be {astar, point}')
        exit()

    segs =  [tokenizeByLength(line, ls) for ls in lss]

    return segs

def nbestIdSegmentation(idTable, logProbTable, n, mode='astar'):
    if mode=='astar':
        # forward
        maxScores, _ = viterbiForward(logProbTable)
        # backward
        lss = nbestAstarBakckward(maxScores, logProbTable, n=n)
    elif mode=='point':
        # viterbi
        bestSeg = viterbi(logProbTable)
        # point estimation
        lss = nbestPointEstimation(bestSeg, logProbTable, n=n)
    else:
        print('mode should be {aster, point}')
        exit()

    def getIds(idTable, ls):
        c = 0
        ids = []
        for l in ls:
            c += l
            ids.append(idTable[c-1, l-1])
        return ids

    idss = [getIds(idTable, ls) for ls in lss]

    return idss

def mSampleFromNBestSegmentation(line, logProbTable, m, n, mode='astar'):
    if mode!='astar':
        print('mode %s is not implemented'%mode)
        exit()
    assert m<=n, 'mSamplingFromNbestSegmentation requires: m <= n'

    # forward
    maxScores, _ = viterbiForward(logProbTable)

    # nbest backward
    ress = [(tokenizeByLength(line, ls), logP) for ls, logP in nbestAstarBakckward(maxScores, logProbTable, n=n)]
    segs, logPs = zip(*ress)

    size = len(segs)
    if size <= m:
        return segs

    # m-sampling
    dist = logPs - logsumexp(logPs)
    dist = np.exp(dist)

    if np.any(np.isnan(dist)):
        # debug
        print(dist)
        print(logProbTable)
        print(line)

    segIdx = np.random.choice(size, m, p=dist, replace=False)
    segs = [segs[si] for si in segIdx]

    return segs

def getIds(idTable, ls):
    c = 0
    ids = []
    for l in ls:
        c += l
        ids.append(idTable[c-1, l-1])
    return ids

def mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar'):
    if mode!='astar':
        print('mode %s is not implemented'%mode)
        exit()
    assert m<=n, 'mSamplingFromNbestSegmentation requires: m <= n'

    # forward
    maxScores, _ = viterbiForward(logProbTable)

    # nbest backward
    #ress = [(tokenizeByLength(line, ls), logP) for ls, logP in nbestAstarBakckward(maxScores, logProbTable, n=n)]
    ress = [(getIds(idTable, ls), logP) for ls, logP in nbestAstarBakckward(maxScores, logProbTable, n=n)]
    isegs, logPs = zip(*ress)

    size = len(isegs)
    if size <= m:
        return isegs

    # m-sampling
    dist = logPs - logsumexp(logPs)
    dist = np.exp(dist)

    if np.any(np.isnan(dist)):
        # debug
        print(dist)
        print(logProbTable)
        print(line)

    segIdx = np.random.choice(size, m, p=dist, replace=False)
    isegs = [isegs[si] for si in segIdx]

    return isegs

def nbestPointEstimation(bestSegLen, logProbTable, n):
    maxLength = logProbTable.shape[1]

    def vary(seg):
        variations = [seg[:k] 
                      + (1-seg[k],) 
                      + seg[k+1:] for k in range(len(seg))]
        return variations
    
    def calcSegScore(seg):
        score, p, l = 0, 0, 0
        for s in seg+(1,):
            if s==1:
                score += logProbTable[p,l]
                l = 0;  p += 1
            else:
                l += 1; p += 1
                if maxLength <= l:
                    return float('-inf')
        return score

    def seg2len(seg):
        ls = []
        l = 0
        for s in seg+(1,):
            l += 1
            if s==1:
                ls.append(l)
                l = 0 
        return ls

    bestSeg = tuple(s for l in bestSegLen for s in (0,)*(l-1)+(1,))[:-1]
    queue = {bestSeg: calcSegScore(bestSeg)}
    nbests = {}

    for _ in range(n):
        if not queue:
            break
        nbestSeg = sorted(queue.items(), key=lambda x:x[1], reverse=True)[0][0]
        if queue[nbestSeg]==float('-inf'): break
        nbests[nbestSeg] = queue[nbestSeg]
        del queue[nbestSeg]

        for seg in vary(nbestSeg):
            if seg not in queue and seg not in nbests:
                queue[seg] = calcSegScore(seg)

    return [seg2len(seg) for seg, score in sorted(nbests.items(), key=lambda x:x[1], reverse=True)]

def nbestAstarBakckward(viterbiScores, logProbTable, n):
    def calcNextScores(prevIdx, prevScore, path, maxLength):
        prevIdxM1 = prevIdx-1
        startIdx = max(prevIdx-maxLength, 0)

        ids = range(startIdx,prevIdx)
        wordScores = logProbTable[prevIdxM1, prevIdxM1-startIdx::-1].tolist()
        for i, wordScore in zip(ids, wordScores):
            if wordScore==minf:
                continue
            nextScore = prevScore + wordScore
            nextPriority = nextScore + viterbiScores[i]
            yield nextPriority, nextScore, i

    def _calcNextScores(prevIdx, prevScore, maxLength):
        prevIdxM1 = prevIdx-1
        startIdx = max(prevIdx-maxLength, 0)

        ids = np.arange(startIdx, prevIdx, dtype=np.int64)
        wordScores = logProbTable[prevIdxM1][prevIdxM1-ids]
        nominf = np.where(wordScores!=minf)

        ids = ids[nominf]
        nextScores = prevScore + wordScores[nominf]
        nextPriorities = nextScores + viterbiScores[ids]
        
        #return map(lambda x:x.tolist(), [nextPriorities, nextScores, ids])
        return nextPriorities, nextScores, ids

    def backtrace(ls):
        size = len(ls)
        return [ls[i]-ls[i-1] for i in range(1,size)]

    # add BOS
    viterbiScores = np.hstack([0, viterbiScores]).tolist()
    maxLength = logProbTable.shape[1]

    queue = [(0, 0, [len(viterbiScores)-1])] # initialized with endnode. requrires: (priority, score, idx to trace+)
    m = 0

    maxQsize = 0

    while queue:
        #lq = len(queue)
        #print('\rsize of queue: %d'%lq, end='')
        #maxQsize = max(maxQsize, lq)

        # pop
        _, prevScore, path = queue.pop()
        prevIdx = path[-1]

        # BOS
        if prevIdx==0:
            yield backtrace(path[::-1]), prevScore
            m += 1
            if n<=m: break
            continue

        #nextNodes = [(nextPriority, nextScore, path+[nextIdx]) 
        #             for nextPriority, nextScore, nextIdx in calcNextScores(prevIdx, prevScore, path, maxLength)]
        #queue += nextNodes
        
        queue += [(nextPriority, nextScore, path+[nextIdx]) 
                  for nextPriority, nextScore, nextIdx in calcNextScores(prevIdx, prevScore, path, maxLength)]

        # sort queue
        queue = sorted(queue)

        # limit queue size
        queue = queue[-512:]

    '''
    # back trace
    lss = [[ls[i]-ls[i-1] for i in range(1,len(ls))] for ls in nbests]
    return lss
    '''

def wrapNbestSegmentation(xs):
    return nbestSegmentation(*xs)

def nbestSegmentationMultiProc(lines, logProbTables, n, mode='astar'):
    size = len(lines)
    if size <= 512:
        return [nbestSegmentation(line, logProbTable, n, mode)
                for line, logProbTable in zip(lines, logProbTables)]

    ns = [n] * size
    modes = [mode] * size
    
    p = Pool(multi.cpu_count())
    segs = p.map(wrapNbestSegmentation, zip(lines, logProbTables, ns, modes))
    p.close()
    
    return segs

def mSamplingFromNbestSegmentationMultiProc(lines, logProbTables, m, n, mode='astar'):
    size = len(lines)
    if size <= 512:
        return [mSampleFromNBestSegmentation(line, logProbTable, m, n, mode='astar')
                for line, logProbTable in zip(lines, logProbTables)]

    ns = [n] * size
    modes = [mode] * size
    
    p = Pool(multi.cpu_count())
    segs = p.map(wrapNbestSegmentation, zip(lines, logProbTables, ns, modes))
    p.close()
    
    return segs

def nSampleWithFFBS(line, logProbTable, n):
    lss = ffbs(logProbTable, n)
    segs = [tokenizeByLength(line, ls) for ls in lss]
    return segs

def checkSpeed():
    from tqdm import tqdm
    import pickle
    path = '../results/20200210142652-sypfvsomti-data=#home#hiraoka.t#work#data#twitter_ja#twitter_ja_train_text.txt-testData=None-maxEpoch=20-maxLength=8-minFreq=25/lm.pickle'
    lm = pickle.load(open(path, 'rb'))
    data = [line.strip() for line in open('../../../data/twitter_ja/twitter_ja_train_text.txt')] 
    data = data[:1024]

    #'''
    st = time()
    logProbTables = [lm.makeLogProbTable(line) for line in data]
    segs = nbestSegmentationMultiProc(data, logProbTables, n=5, mode='astar')
    print('multi:', time()-st)
    #'''
    st = time()
    for line in data:
        logProbTable = lm.makeLogProbTable(line)
        nbest = nbestSegmentation(line, logProbTable, n=5, mode='astar')
    print('single:', time()-st)

def checkCalc():
    from tqdm import tqdm
    import pickle
    path = '../results/20200210142652-sypfvsomti-data=#home#hiraoka.t#work#data#twitter_ja#twitter_ja_train_text.txt-testData=None-maxEpoch=20-maxLength=8-minFreq=25/lm.pickle'
    lm = pickle.load(open(path, 'rb'))
    data = [line.strip() for line in open('../../../data/twitter_ja/twitter_ja_train_text.txt')] 
    data = data[:1024]

    for line in data:
        print(line)
        logProbTable = lm.makeLogProbTable(line)
        a = viterbiForward(logProbTable)
        b = viterbiForwardOrig(logProbTable)
        print(np.all(a[0]==b[0]), np.all(a[1]==b[1]))
        nbest = nbestSegmentation(line, logProbTable, n=5, mode='astar')

def checkBug():
    from tqdm import tqdm
    import pickle
    path = '../results/20200210142652-sypfvsomti-data=#home#hiraoka.t#work#data#twitter_ja#twitter_ja_train_text.txt-testData=None-maxEpoch=20-maxLength=8-minFreq=25/lm.pickle'
    lm = pickle.load(open(path, 'rb'))
    
    #lm.shrinkVocab(8000)
    lm.theta[8000:] = 0
    lm.reIndex()
    
    data = [line.strip() for line in open('../../../data/twitter_ja/twitter_ja_train_text.txt')] 
    

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=3)
   
    for line in tqdm(data):
        logProbTable = lm.makeLogProbTable(line)
        nbest = nbestSegmentation(line, logProbTable, n=5, mode='astar')
    
if __name__ == '__main__':
    #checkBug(); exit()
    #checkSpeed(); exit()
    #checkCalc(); exit()

    '''
    probTable contains probability of each token
    in the same manner as alpha like:

    ---
    a   x   x
    b   ab  x
    c   bc  abc
    d   cd  bcd
    e   de  cde
    ---
    '''
    
    probTable = [[1, 0, 0],
                 [2, 3, 0],
                 [4, 5, 6],
                 [7, 8, 9],
                 [1, 2, 3]]
    probTable = np.array(probTable[:-1])*0.01
    logProbTable = np.log(probTable)
    
    logProbTable = np.vstack([logProbTable, logProbTable[2:]])
    line = 'abcde'+'abc'
    
    print(viterbiSegmentation(line, logProbTable))
    print(nbestSegmentation(line, logProbTable, 5))
    print(mSampleFromNBestSegmentation(line, logProbTable, m=5, n=10, mode='astar'))
    print(nSampleWithFFBS(line, logProbTable, 5))

    #print(samplingSegmentation('abcde', logProbTable))
    #alpha, sumAlpha = calcAlpha(logProbTable)
    #print(alpha)
    #sumBeta = calcBeta(logProbTable)
    #print(sumBeta)
    #sumGamma = calcGamma(logProbTable, alpha, sumAlpha)
    #posterior = calcPosterior(alpha, sumBeta, sumGamma)
