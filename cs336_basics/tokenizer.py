import json
import regex as re
from collections.abc import Iterable, Iterator
import time
import random
import numpy as np



class tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
    ):
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens
        self.vocab = vocab
        self.cache = {}

        self.token_to_id = {
            token: token_id for token_id, token in vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.special_tokens_bytes = {
            token: token.encode("utf-8") for token in special_tokens
        }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, "r", encoding = "utf-8") as f:
            vocab_json = json.load(f)

        vocab = {}

        for token_id, token in vocab_json.items():
            vocab[int(token_id)] = token.encode("latin-1")

        merges = []

        with open(merges_filepath, "r", encoding = "latin-1") as f:
            for line in f:
                line = line.rstrip("\n")

                if not line:
                    continue

                token1, token2 = line.split(" ", 1)

                merges.append((token1.encode("latin-1"), token2.encode("latin-1")))

        return cls(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens,
        )


    def encode(self, text: str)-> list[int]:
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key = len, reverse= True)
            delimiter = "(" + "|".join(map(re.escape, special_tokens)) + ")"
            parts = re.split(delimiter, text)
        else:
            parts = [text]


        ids = []
        pre_token = []

        for part in parts:
            
            if part in self.special_tokens:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue

            else:
                for m in re.finditer(PAT, part):
                    pre_token = m.group()

                    if pre_token in self.cache:
                        ids.extend(self.cache[pre_token])
                        continue

                    tokens = [bytes([b]) for b in pre_token.encode("utf-8")]

                    while True:
                        best_pair = None
                        best_rank = 10000000

                        for i in range(len(tokens) - 1):
                            pair = (tokens[i], tokens[i+1])
                            if pair in self.merge_ranks:
                                rank = self.merge_ranks[pair]

                                if rank < best_rank:
                                    best_rank = rank
                                    best_pair = pair

                        if best_pair == None:
                            break

                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                        

                            if i < len(tokens) - 1 and best_pair == (tokens[i], tokens[i+1]):
                                new_tokens.append(tokens[i] + tokens[i+1])
                                i += 2

                            else :
                                new_tokens.append(tokens[i])
                                i += 1

                        tokens = new_tokens

                    tokens_id = [self.token_to_id[token] for token in tokens]

                    if len(self.cache) < 100_000:
                        self.cache[pre_token] = tokens_id

                    ids.extend(tokens_id)

                        

        return ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)

            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        decode_list = []
        for token_id in ids:
            decode_list.append(self.vocab[token_id])
        
        return b"".join(decode_list).decode("utf-8", errors="replace")


"""    //benchmark test

def compression_ratio(tokenizer, documents):
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in documents)

    return total_bytes / total_tokens

def sample_doc(file_path, num):
    with open(file_path, "r", encoding = "utf-8") as f:
        text = f.read()

    documents = text.split("<|endoftext|>")

    documents = [doc for doc in documents if doc.strip()]

    random.seed(42)

    sample_documents = random.sample(documents, num)

    return sample_documents

def measure_throughput(tokenizer, documents):

    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)

    start = time.perf_counter()

    for doc in documents:
        tokenizer.encode(doc)

    elapsed = time.perf_counter() - start

    return total_bytes / elapsed


def main():
    tiny_tokenizer = tokenizer.from_files("outputs/tinystories_vocab.json", "outputs/tinystories_merges.txt", ["<|endoftext|>"])
    owt_tokenizer = tokenizer.from_files("outputs/owt_vocab.json", "outputs/owt_merges.txt", ["<|endoftext|>"])

    tiny_doc = sample_doc("data/TinyStoriesV2-GPT4-valid.txt", 10)
    owt_doc = sample_doc("data/owt_valid.txt", 10)

    tiny_ratio = compression_ratio(tiny_tokenizer, tiny_doc)
    owt_ratio = compression_ratio(owt_tokenizer, owt_doc)
    owt_with_tiny_ratio = compression_ratio(tiny_tokenizer, owt_doc)

    throughput = measure_throughput(owt_tokenizer, owt_doc)


    print("tiny_ratio = ", tiny_ratio)
    print("owt_ratio = ", owt_ratio)
    print("owt_with_tiny_ratio = ", owt_with_tiny_ratio)
    print("bytes/sec: ", throughput)
    print("MB/sec: ", throughput / 1e6)


if __name__ == "__main__":
    main() 
    
"""

def encode_dataset(
        tokenizer,
        input_path,
        output_path
):
    buffer_size = 1_000_000
    buffer = []

    with open(input_path, "r", encoding = "utf-8") as f, open(output_path, "wb") as out:
        for token_id in tokenizer.encode_iterable(f):
            buffer.append(token_id)

            if len(buffer) >= buffer_size:
                np.asarray(buffer, dtype = np.uint16).tofile(out)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(out)

    
def main():
    tiny_tokenizer = tokenizer.from_files("outputs/tinystories_vocab.json", "outputs/tinystories_merges.txt", ["<|endoftext|>"])
    owt_tokenizer = tokenizer.from_files("outputs/owt_vocab.json", "outputs/owt_merges.txt", ["<|endoftext|>"])

    # encode_dataset(tiny_tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "outputs/tiny_train.bin")
    
    # encode_dataset(tiny_tokenizer, "data/TinyStoriesV2-GPT4-valid.txt", "outputs/tiny_valid.bin")

    encode_dataset(owt_tokenizer, "data/owt_train.txt", "outputs/owt_train.bin")
    encode_dataset(owt_tokenizer, "data/owt_valid.txt", "outputs/owt_valid.bin")

if __name__ == "__main__":
    main()