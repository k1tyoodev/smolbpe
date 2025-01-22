class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all utf-8 bytes), no merges, no patterns
        self.merges = {}  # store the merge process (int, int) -> int
        self.pattern = ""  # regex pattern
        self.special_tokens = {}  # str -> int
        self.vocab = self._build_vocab()  # int -> bytes

    def _build_vocab(self):
        # update vocab from merges dict
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def train(self, text, vocab_size, verbose=False):
        # train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # string -> token id
        raise NotImplementedError

    def decode(self, ids):
        # token id -> string
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass
