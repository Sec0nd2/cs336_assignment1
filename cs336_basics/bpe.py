import regex as re
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter


# parellel child process for pre_tokenization
def parallel_tokenization(
    task: tuple,
):
    input_path, delimiter, start, end = task
    pre_token = dict()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        corpus = chunk.decode("utf-8", errors="ignore")

        parts = re.split(delimiter, corpus)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:
            for m in re.finditer(PAT, part):
                pre_token[m.group()] = pre_token.get(m.group(), 0) + 1
    return pre_token


#pre tokenization 
def pre_tokenization(
    input_path: str,
    special_tokens: list[str],
)-> dict[tuple[bytes, ...], int]:
    
    with open(input_path, "rb") as f:
        num_chunks = 100
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    delimiter = "|".join(map(re.escape, special_tokens))
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, delimiter, start, end))

    num_processes = 8
    with Pool(num_processes) as pool:
        results = pool.map(parallel_tokenization, tasks)

    merged_dict = Counter()
    for d in results:
        merged_dict.update(d)
    pre_tokenization_dict = { tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in merged_dict.items()}
    return pre_tokenization_dict


def pre_token_merges(
    pre_tokenization_dict: dict[tuple[bytes, ...], int],
)->tuple[
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[int, int]]],
    dict[int, int],
    dict[int, list[tuple[bytes, ...]]],
]:
    pairs_counts = Counter()
    pairs_to_position = dict()
    #position_to_pairs = dict()
    pre_token_counts = dict()
    token_sequence = {}
    tokens_id = 0
    for tokens, count in pre_tokenization_dict.items():
        token_sequence[tokens_id] = list(tokens)
        pre_token_counts[tokens_id] = count
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])

            pairs_counts[pair] += count

            pairs_to_position.setdefault(pair, set()).add((tokens_id, i))

            #position_to_pairs[(tokens_id, i)] = pair
        tokens_id += 1

    return pairs_counts, pairs_to_position, pre_token_counts, token_sequence


    # if not pairs_counts:
    #     return pre_tokenization_dict, None
    # max_count = pairs_counts.most_common(1)[0][1]
    
    # all_tops = [j[0] for j in pairs_counts.items() if j[1] == max_count]
    # max_top = max(all_tops)
    

    # tokenization_dict = {}

    # for tokens, count in pre_tokenization_dict.items():
    #     new_tokens = []
    #     i = 0
    #     while i < len(tokens):
    #         if i < len(tokens) - 1 and max_top == (tokens[i], tokens[i+1]):
    #             new_tokens.append(tokens[i] + tokens[i+1])
    #             i += 2
    #         else:
    #             new_tokens.append(tokens[i])
    #             i += 1
    #     tokenization_dict[tuple(new_tokens)] = tokenization_dict.get(tuple(new_tokens), 0) + count

        
    # return tokenization_dict, max_top


def token_merge(
    pairs_counts: dict[tuple[bytes, bytes], int], 
    pairs_to_position: dict[tuple[bytes, bytes], set[tuple[int, int]]], 
    #position_to_pairs: dict[tuple[tuple[bytes, ...], int], tuple],
    pre_token_counts:dict[int, int],
    token_sequence: dict[int, list[tuple[bytes, ...]]],

)->tuple[bytes, bytes]:
    max_count = pairs_counts.most_common(1)[0][1]
     
    all_tops = [j[0] for j in pairs_counts.items() if j[1] == max_count]
    max_top = max(all_tops)    

    position = pairs_to_position[max_top]
    affected_tokens = {}
    for tokens_id, i in position:
        affected_tokens.setdefault(tokens_id,[]).append(i)


    for tokens_id in affected_tokens:
        tokens = token_sequence[tokens_id]

        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pairs_counts[pair] -= pre_token_counts[tokens_id]
            pairs_to_position[pair].discard((tokens_id, j))
            
            
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and max_top == (tokens[i], tokens[i+1]):
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        token_sequence[tokens_id] = new_tokens
        tokens = new_tokens

        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pairs_counts[pair] += pre_token_counts[tokens_id]
            pairs_to_position.setdefault(pair, set()).add((tokens_id, j))

    pairs_counts.pop(max_top, None)
    pairs_to_position.pop(max_top, None)

    return pairs_counts, pairs_to_position, pre_token_counts, token_sequence, max_top

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    #vocab: dict[int, bytes],
    #merges: list[tuple[bytes, bytes]],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]]
]:
    pre_tokenization_dict = pre_tokenization(input_path, special_tokens)

    merge_count = vocab_size - 256 - len(special_tokens)
    merges = []
    vocab = {
        i: bytes([i]) for i in range(256)
    }
    new_tokens = 256

    pairs_counts, pairs_to_position, pre_token_counts, token_sequence = pre_token_merges(pre_tokenization_dict)

    for i in range(merge_count):
        pairs_counts, pairs_to_position, pre_token_counts, token_sequence, pair = token_merge(pairs_counts, pairs_to_position, pre_token_counts, token_sequence)

        if pair is None:
            break

        left, right = pair

        vocab[new_tokens] = left + right
        merges.append(pair)
        new_tokens += 1

    
    # for i in range(merge_count):
    #     pre_tokenization_dict, pair = token_merges(pre_tokenization_dict, new_tokens)

    #     if pair is None:
    #         break

    #     left, right = pair
        
    #     vocab[new_tokens] = left + right
    #     merges.append(pair)
    #     new_tokens += 1

    for special_token in special_tokens:
        vocab[new_tokens] = special_token.encode("utf-8")
        new_tokens += 1
    

    return vocab, merges


# if __name__ == "__main__":
#     vocab, merges = train_bpe(
#         "data/TinyStoriesV2-GPT4-valid.txt",
#         10000,
#         ["<|endoftext|>"],
#     )

#     print("vocab size:", len(vocab))
#     print("merge count:", len(merges))
