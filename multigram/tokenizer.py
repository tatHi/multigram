from . import lm
from . import mdp

class Tokenizer:
    def __init__(self, lm):
        self.lm = lm
        self.id_to_piece = self.lm.id_to_piece
    
    def encode_as_pieces(self, line):
        logProbTable = self.lm.makeLogProbTable(line)
        return mdp.tokenize(line, logProbTable, sampling=False)
    
    def encode_as_ids(self, line):
        return [self.lm.word2id[w] for w in self.encode_as_pieces(line)]

    def sample_encode_as_pieces(self, line, _=None, __=None):
        # _ and __ are dummy arguments to mock the sentencepiece module.
        # _ : number of search width, __ is the number of sampled tokenization.
        # This method is equivalent to `sample_encode_as_pieces(line, -1, 1)` of SentencePiece
        logProbTable = self.lm.makeLogProbTable(line)
        return mdp.tokenize(line, logProbTable, sampling=True)
    
    def sample_encode_as_ids(self, line, _=None, __=None):
        return [self.lm.word2id[w] for w in self.sample_encode_as_pieces(line)]

