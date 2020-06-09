import numpy as np
from scipy.special import logsumexp
import random
from multiprocessing import Pool
import multiprocessing as multi
from time import time

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
    
def ffbs(logProbTable):
    T, L = logProbTable.shape
    alpha, sumAlpha = calcAlpha(logProbTable)
    dists = alpha - sumAlpha.reshape(T,1)
    dists = np.exp(dists)
    #sampledLs = np.argmax(dists + np.random.gumbel(size=dists.shape), axis=1)

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

def viterbiForward(logProbTable):
    maxScores = []
    #maxLs = []
    def func(lpList):
        maxScores.append(max(lpList))
        return np.argmax(lpList, axis=0)

    maxLs = [func([logProbTable[t,l] + (maxScores[t-l-1] if 0<=t-l-1 else 0)
                   for l in range(logProbTable.shape[1])]) 
             for t in range(logProbTable.shape[0])]
    '''# forward
    for t in range(logProbTable.shape[0]):
        lpList = [logProbTable[t,l] + (maxScores[t-l-1] if 0<=t-l-1 else 0) 
                  for l in range(logProbTable.shape[1])]
        maxScores.append(max(lpList))
        maxLs.append(np.argmax(lpList))'''

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
    segs = []
    pointer = 0
    for l in ls:
        segs.append(line[pointer:pointer+l])
        pointer += l
    return segs

def tokenize(line, logProbTable, sampling):
    ls = ffbs(logProbTable) if sampling else viterbi(logProbTable)
    segs = tokenizeByLength(line, ls)
    return segs    

def samplingSegmentation(line, logProbTable):
    return tokenize(line, logProbTable, sampling=True)

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
    # add BOS
    viterbiScores = [0] + viterbiScores

    maxLength = logProbTable.shape[1]

    nbests = []
    endNode = ([len(viterbiScores)-1], 0, 0)
    queue = [endNode] # requrires: (idx to trace, priority, score)

    while queue:
        # pop
        node = queue.pop()
        path = node[0]
        prevIdx = path[-1]

        # BOS
        if prevIdx==0:
            nbests.append(path[::-1])
            if n<=len(nbests): break
            continue

        prevIdxM1 = prevIdx-1
        prevScore = node[2]
        
        for i in range(max(prevIdx-maxLength, 0), prevIdx):
            wordScore = logProbTable[prevIdxM1, prevIdxM1-i]
            if wordScore==minf:
                continue
            nextScore = prevScore + wordScore
            nextPriority = nextScore + viterbiScores[i]
            nextNode = (path+[i], nextPriority, nextScore)
            queue.append(nextNode)

        # sort queue
        queue = sorted(queue, key=lambda x:x[1])

    # back trace
    lss = [[ls[i]-ls[i-1] for i in range(1,len(ls))] for ls in nbests]
    return lss

def wrapNbestSegmentation(xs):
    return nbestSegmentation(*xs)

def nbestSegmentationMultiProc(lines, logProbTables, n, mode='astar'):
    size = len(lines)
    if size <= 256:
        return [nbestSegmentation(line, logProbTable, n, mode)
                for line, logProbTable in zip(lines, logProbTables)]

    ns = [n] * size
    modes = [mode] * size
    
    p = Pool(multi.cpu_count())
    segs = p.map(wrapNbestSegmentation, zip(lines, logProbTables, ns, modes))
    p.close()
    
    return segs

def checkSpeed():
    from tqdm import tqdm
    import pickle
    path = '../results/20200210142652-sypfvsomti-data=#home#hiraoka.t#work#data#twitter_ja#twitter_ja_train_text.txt-testData=None-maxEpoch=20-maxLength=8-minFreq=25/lm.pickle'
    lm = pickle.load(open(path, 'rb'))
    data = [line.strip() for line in open('../../../data/twitter_ja/twitter_ja_train_text.txt')] 
    data = data[:1024]

    st = time()
    logProbTables = [lm.makeLogProbTable(line) for line in data]
    segs = nbestSegmentationMultiProc(data, logProbTables, n=5, mode='astar')
    print('multi:', time()-st)

    st = time()
    for line in data:
        logProbTable = lm.makeLogProbTable(line)
        nbest = nbestSegmentation(line, logProbTable, n=5, mode='astar')
    print('single:', time()-st)

if __name__ == '__main__':
    checkSpeed()
    exit()

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

    #print(samplingSegmentation('abcde', logProbTable))
    #alpha, sumAlpha = calcAlpha(logProbTable)
    #print(alpha)
    #sumBeta = calcBeta(logProbTable)
    #print(sumBeta)
    #sumGamma = calcGamma(logProbTable, alpha, sumAlpha)
    #posterior = calcPosterior(alpha, sumBeta, sumGamma)
