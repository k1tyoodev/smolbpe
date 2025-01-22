from .base import Tokenizer
from .utils import get_stats, get_merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size > 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # find the pair with highest count based on value
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)

            # set new token id
            idx = 256 + i

            ids = get_merge(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        # concat ids byte
        text_bytes = b"".join(self.vocab[idx] for idx in ids)

        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def encode(self, text):
        # text -> raw byte -> token id (interger) -> merged token id (bpe)

        # ------------------ #
        # 流程概述为：
        # 1. 有了原始的 token id 之后，我们需要开始实施 bpe 算法
        # 2. 首先找到出现次数最多的 token id pair，然后用一个新的 token id 替换他
        # 3. 在该项目中，merges 字典 {pair: idx ..} 按顺序存储了所有的替换过程
        # 4. 因此我们需要在该字典中找到第一个 merge 的 pair，该 pair 就是原始 token id 序列当中出现次数最多的 pair
        # 5. 那么如何找呢？通过 idx 去找，因为该 pair 是第一个 merge 的，因此其 idx 肯定是最小的
        # 6. 此时我们有了出现次数最多的 pair 和其对应的 idx，就可以调用 get_merge 函数来得到 merge 后的序列了
        # ------------------ #

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) > 2:
            # find the pair with lowest merge index (256 -> vocab_size)
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            # merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = get_merge(ids, pair, idx)

        return ids
