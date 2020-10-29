from . import lm
from . import mdp

# each method name is correlated to sentencepiece
class Tokenizer:
    def __init__(self, mlm):
        self.mlm = mlm
        self.id_to_piece = self.mlm.id_to_piece
    
    def encode_as_pieces(self, line):
        logProbTable = self.mlm.makeLogProbTable(line)
        return mdp.tokenize(line, logProbTable, sampling=False)
    
    def encode_as_ids(self, line):
        return [self.mlm.piece_to_id(w) for w in self.encode_as_pieces(line)]

    def sample_encode_as_pieces(self, line, _=None, __=None):
        # _ and __ are dummy arguments to mock the sentencepiece module.
        # _ : number of search width, __ is the number of sampled tokenization.
        # This method is equivalent to `sample_encode_as_pieces(line, -1, 1)` of SentencePiece
        logProbTable = self.mlm.makeLogProbTable(line)
        return mdp.tokenize(line, logProbTable, sampling=True)
    
    def sample_encode_as_ids(self, line, _=None, __=None):
        return [self.mlm.piece_to_id(w) for w in self.sample_encode_as_pieces(line)]

    def nbest_encode_as_pieces(self, line, n):
        logProbTable = self.mlm.makeLogProbTable(line)
        return mdp.nbestSegmentation(line, logProbTable, n)
    
    def nbest_encode_as_ids(self, line, n):
        idTable = self.mlm.makeIdTable(line)
        logProbTable = self.mlm.makeLogProbTable(line, idTable=idTable)
        return mdp.nbestIdSegmentation(idTable, logProbTable, n)

    def get_piece_size(self):
        return len(self.mlm.vocab) 
