import numpy as np
from collections import defaultdict
import pickle

class MultigramLM:
    def __init__(self, maxLength=5, minFreq=4, data=None, wordPiecePrefix=None, unkToken='<unk>'):
        self.maxLength = maxLength
        self.minFreq = minFreq
        self.theta = None
        
        self.replaceSpaceMode = False
        self.wordPiecePrefix = wordPiecePrefix
        # memo set wordPiecePrefix as '' if you want to recognize whitespace as word boundary.
        self.unkToken = unkToken
        
        if data:
            self.buildVocab(data)

    def buildVocab(self, data):
        # data is a list of string
        self.unigramFreq = 0
        wordCountDict = defaultdict(lambda:0)

        for line in data:
            for i in range(len(line)):
                for l in range(min(i+1, self.maxLength)):
                    w = line[i-l:i+1]
                    wordCountDict[w] += 1
            self.unigramFreq += len(line)


        self.vocab = set(k for k,v in wordCountDict.items() if len(k)==1 or self.minFreq<=v)
        
        # add unk
        self.vocab.add(self.unkToken)

        self.word2id = {w:i for i,w in enumerate(sorted(list(self.vocab)))}
        self.id2word = {i:w for w,i in self.word2id.items()}

        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        print('>>> BUILD VOCABULARY')
        print('possible n-grams (n=%d):'%self.maxLength, len(self.vocab))

        self.theta = np.random.rand(len(self.vocab))
        self.theta = self.theta/sum(self.theta)
        print('>>> INITIALIZE THETA')

    def __addLineToVocab(self, line):
        for i in range(len(line)):
            for l in range(min(i+1, self.maxLength)):
                w = line[i-l:i+1]
                self.vocab.add(w)
        self.unigramFreq += len(line)

    def addWordToVocab(self, word, p=0.0):
        if word in self.vocab:
            return False
        
        self.vocab.add(word)
        self.word2id[word] = len(self.word2id)
        self.id2word[self.word2id[word]] = word

        # add theta as its prob=0.0
        self.theta = np.append(self.theta, p)

        return True

    def piece_to_id(self, piece):
        if piece not in self.vocab:
            piece = self.unkToken
        return self.word2id[piece]

    def id_to_piece(self, i):
        return self.id2word[i]

    def setVocabFromWord2Id(self, w2i):
        # dict = {w:p, w:p, ...}
        self.vocab = set(w2i.keys())
        self.word2id = w2i
        self.id2word = {i:w for w,i in self.word2id.items()}
        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        self.theta = np.array([1 for i in range(len(self.word2id))])
        self.theta = self.theta/self.theta.sum()

        self.maxLength = max([len(w) for w in self.vocab])

    def setVocabFromWordList(self, wordList):
        dummyUnigramDict = {w:1 for w in wordList}
        self.setVocabFromUnigramDict(dummyUnigramDict)

    def setVocabFromUnigramDict(self, unigramDict):
        # dict = {w:p, w:p, ...}
        self.vocab = set(unigramDict.keys())
        self.word2id = {w:i for i,w in enumerate(sorted(list(self.vocab)))}
        self.id2word = {i:w for w,i in self.word2id.items()}
        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        self.theta = np.array([unigramDict[self.id2word[i]] for i in range(len(self.word2id))])
        self.theta = self.theta/self.theta.sum()

        self.maxLength = max([len(w) for w in self.vocab])

    def setVocabFromBERTVocab(self, vocab):
        self.vocab = set(vocab.keys())
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(vocab):
            assert i==vocab[w], 'index mismatch'
            self.word2id[w] = i
            self.id2word[i] = w
        charVocab = set(w for w in self.vocab if len(w)==1)
        self.char2id = {w:i for i,w in enumerate(sorted(list(charVocab)))}
        self.id2char = {i:w for w,i in self.char2id.items()}

        self.unigramFreq = None
        size = len(self.vocab)
        p = 1/size
        self.theta = np.full(size, p)            

        self.maxLength = max([len(w) for w in vocab])

    def setThetaFromSentencePiece(self, path):
        sp = self.__loadSentencePieceModel(path)
        spPiece2Score = {sp.id_to_piece(i):sp.get_score(i) for i in range(sp.get_piece_size())}
        # vocabs which is not included in loaded sentencepiece
        vocabDiff = self.vocab-set(spPiece2Score.keys()) 
        print('vocab diff')
        print(vocabDiff)

        print('set vocab diff as small value, exp(-30)') 
        for w in vocabDiff:
            spPiece2Score[w] = -30
        
        print('set scores of special tokens as exp(-30)')
        for w in spPiece2Score:
            if spPiece2Score[w]==0.0:
                # scores of special tokens (<s>, </s>, <unk>) are 0.0
                spPiece2Score[w] = -30

        self.theta = [spPiece2Score[self.id2word[i]] for i in range(len(self.vocab))]
        self.theta = np.exp(self.theta)
        self.theta = self.theta / self.theta.sum()

    def __loadSentencePieceModel(self, path):
        import sentencepiece as sp
        spp = sp.SentencePieceProcessor()
        if spp.load(path):
            print('>>> LOAD SENTENCEPIECE MODEL')
        else:
            print('>>> FAIL TO LOAD SENTENCE MODEL, EXIT')
            exit()
        return spp

    def convertWords2Ids(self, words, unkIdx=None):
        if unkIdx is None: unkIdx = self.word2id[self.unkToken]
        ids = [self.word2id[word] if word in self.vocab else unkIdx for word in words]
        return ids
    
    def convertIds2Words(self, ids):
        words = [self.id2word[i] for i in ids]
        return words

    def makeLogProbTable(self, line, unkProb=1e-7, idTable=None, lam=1.0):
        if idTable is None:
            idTable = self.makeIdTable(line, paddingIdx=-1, unkCharIdx=self.word2id[self.unkToken])

        # calc theta
        if lam==1.0:
            theta = self.theta
        else:
            theta = self.theta ** lam
            theta = theta / theta.sum()

        I, J = idTable.shape
        probTable = np.zeros((I, J))
        for t in range(I):
            for l in range(min(t+1, J)):
                i = idTable[t,l]
                if i != -1:
                    probTable[t,l] = theta[i]
        logProbTable = np.log(probTable)
        return logProbTable

    def makeIdTable(self, line, paddingIdx=-1, unkCharIdx=None, vocab=None):
        if unkCharIdx is None: unkCharIdx = self.word2id[self.unkToken]
        # specify vocab if you want to limit the vocab for some reasons
        if vocab is None: vocab = self.vocab
       
        if not hasattr(self, 'wordPiecePrefix'):
            self.wordPiecePrefix = None

        if self.wordPiecePrefix is not None:
            # if wordPiecePrefix is '', just skipping whitespace as word boundary.
            heads = {0}
            c = 0
            for a in line:
                if a == ' ':
                    heads.add(c)
                    continue
                c += 1
            # the size of table's column is gained by removing space when using word piece mode
            line = ''.join(line.split())
        idTable = np.full((len(line), self.maxLength), paddingIdx)
        if unkCharIdx is not None:
            idTable[:,0] = unkCharIdx

        for t in range(len(line)):
            for l in range(min(t+1, self.maxLength)):
                w = line[t-l:t+1]
                if self.wordPiecePrefix and t-l in heads:
                    if 0<sum([i in heads for i in range(t-l+1,t+1)]):
                        continue 
                    w = self.wordPiecePrefix + w
                if w in vocab:
                    if self.wordPiecePrefix and 1<=len(set(range(t-l+1,t+1))&heads):
                        continue
                    idTable[t,l] = self.word2id[w]

        return idTable

    def getWordIdsInLine(self, line):
        wordIds = []
        for t in range(len(line)):
            for l in range(min(t+1, self.maxLength)):
                w = line[t-l:t+1]
                if w in self.vocab:
                    wordIds.append(self.word2id[w])
        return wordIds

    def proneVocab(self, thre=None):
        if thre is None:
            # 1/2 * 1/sum(all 1-gram freq)
            thre = 1/2 * 1/self.unigramFreq
        print('prone thre:', thre)

        # escape unk
        self.theta[self.word2id[self.unkToken]] = thre

        dropCount = 0
        for i in range(self.theta.shape[0]):
            if self.theta[i] < thre:
                if len(self.id2word[i])==1:
                    self.theta[i] = thre
                else:
                    self.theta[i] = 0
                    dropCount += 1
        print('drop %d tokens'%dropCount)

        # smooth unk
        self.theta = self.theta/sum(self.theta)

    def shrinkVocab(self, size):
        # get a size, then shrink self.vocab into the size
        print('>>> SHRINK VOCAB')
        size -= len(self.char2id)
        print('char size:', len(self.char2id))
        print('word size:', size)

        sortedTheta = sorted(self.theta, reverse=True)
        thre = sortedTheta[size]
        if thre==sortedTheta[size+1]:
            while thre==sortedTheta[size]:
                size -= 1
            thre = sortedTheta[size]
        print('actrual word size:', size)
        self.proneVocab(thre)

    def reIndex(self):
        nonZeroIdx = np.where(0<self.theta)[0]
        self.theta = self.theta[nonZeroIdx]
        neow2i = {}
        neoi2w = {}
        for i in nonZeroIdx:
            w = self.id2word[i]
            neow2i[w] = len(neow2i)
            neoi2w[neow2i[w]] = w
        self.word2id = neow2i
        self.id2word = neoi2w
        self.vocab = set(self.word2id.keys())

    def makeLengthWiseIdDict(self):
        lengthIdDict = {l+1:set() for l in range(self.maxLength)}
        for w,i in self.word2id.items():
            lengthIdDict[len(w)].add(i)
        return lengthIdDict

    def makeLengthWiseManyHotVector(self):
        lengthIdDict = self.makeLengthWiseIdDict()
        manyhot = [[1 if i in lengthIdDict[l+1] else 0 for i in range(len(self.id2word))] for l in range(self.maxLength)]
        return np.array(manyhot)

    def makeCharIdOfVocab(self, paddingIdx=-1):
        ids = [[self.char2id[self.id2word[i][j]] if j<len(self.id2word[i]) else paddingIdx 
                for j in range(self.maxLength)] 
               for i in range(len(self.word2id))]
        ids = np.array(ids)
        return ids

    def getCharIdSet(self):
        # return ids of chars(=single length tokens) as its set.
        if not hasattr(self, 'charIdSet'):
            self.charIdSet = {i for i,w in self.id2word.items() if len(w)==1}
        return self.charIdSet

    def save(self, path):
        pickle.dump(self.__dict__, open(path, 'wb'))

    def load(self, path):
        self.__dict__ = pickle.load(open(path, 'rb'))

    def loadSentencePieceModel(self, path):
        spp = self.__loadSentencePieceModel(path)
        self.wordPiecePrefix = '▁'
        
        self.word2id = {}
        self.id2word = {}
        maxLength = 0
        theta = []

        # 0,1,2: unk, bos, eos
        size = spp.get_piece_size()
        for i in range(size):
            w = spp.id_to_piece(i)
            s = spp.get_score(i)
            
            #if w==self.unkToken:
            if s==0.0:
                print('set %s as small value (-30)'%w)
                # set p(unk) as small value
                # set specialtoken as small value
                s = -30
            
            self.word2id[w] = i
            self.id2word[i] = w
            theta.append(s)
        
            length = len(w)
            maxLength = max(maxLength, length)

        self.maxLength = maxLength
        
        theta = np.exp(np.array(theta))
        theta = theta / np.sum(theta)

        self.theta = theta
        self.vocab = set(self.word2id.keys())
        self.replaceSpaceMode = True

