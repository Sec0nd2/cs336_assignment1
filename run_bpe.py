
# import time
# import tracemalloc

# from cs336_basics.bpe import train_bpe


# def main():
#     input_path = "data/TinyStoriesV2-GPT4-train.txt"

#     tracemalloc.start()
#     start = time.perf_counter()

#     vocab, merges = train_bpe(
#         input_path=input_path,
#         vocab_size=10_000,
#         special_tokens=["<|endoftext|>"],
#     )

#     elapsed = time.perf_counter() - start

#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()

#     print(f"Training time: {elapsed:.2f} seconds")
#     print(f"Peak Python memory: {peak / 1024 / 1024:.2f} MB")
#     print(f"Vocabulary size: {len(vocab)}")
#     print(f"Number of merges: {len(merges)}")

#     # Find the longest token
#     longest_id, longest_token = max(
#         vocab.items(),
#         key=lambda x: len(x[1]),
#     )

#     print(f"Longest token ID: {longest_id}")
#     print(f"Longest token bytes: {longest_token}")
#     print(f"Longest token length: {len(longest_token)}")
#     print(
#         "Longest token text:",
#         longest_token.decode("utf-8", errors="replace"),
#     )


# if __name__ == "__main__":
#     main()



import json
import time
import tracemalloc
from pathlib import Path

from cs336_basics.bpe import train_bpe


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("data/owt_train.txt")
OUTPUT_DIR = Path("outputs")

VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]

VOCAB_PATH = OUTPUT_DIR / "owt_vocab.json"
MERGES_PATH = OUTPUT_DIR / "owt_merges.txt"


# ============================================================
# Serialization
# ============================================================

def save_vocab(vocab: dict[int, bytes], path: Path) -> None:
    """
    Save vocab as JSON.

    vocab:
        token_id -> token bytes

    JSON cannot directly store bytes, so bytes are decoded using
    latin-1, which provides a one-to-one mapping between byte values
    and Unicode code points.
    """
    serializable_vocab = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            serializable_vocab,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_merges(
    merges: list[tuple[bytes, bytes]],
    path: Path,
) -> None:
    """
    Save BPE merges.

    Each line represents:

        token1 token2
    """
    with open(path, "w", encoding="latin-1") as f:
        for left, right in merges:
            f.write(
                f"{left.decode('latin-1')} "
                f"{right.decode('latin-1')}\n"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    print("=" * 60)
    print("Tinystories BPE Training")
    print("=" * 60)

    print(f"Input:       {INPUT_PATH}")
    print(f"Vocab size:  {VOCAB_SIZE}")
    print(f"Special:     {SPECIAL_TOKENS}")
    print()

    # --------------------------------------------------------
    # Start memory/time measurement
    # --------------------------------------------------------

    tracemalloc.start()
    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=str(INPUT_PATH),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed_time = time.perf_counter() - start_time

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --------------------------------------------------------
    # Save vocabulary and merges
    # --------------------------------------------------------

    save_vocab(vocab, VOCAB_PATH)
    save_merges(merges, MERGES_PATH)

    # --------------------------------------------------------
    # Find longest token
    # --------------------------------------------------------

    longest_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Training Results")
    print("=" * 60)

    print(f"Training time:       {elapsed_time:.2f} seconds")
    print(f"Training time:       {elapsed_time / 60:.2f} minutes")

    print(
        f"Peak Python memory:  "
        f"{peak_memory / 1024 / 1024:.2f} MB"
    )

    print(f"Vocab size:          {len(vocab)}")
    print(f"Number of merges:    {len(merges)}")

    print()
    print("Longest token:")
    print(f"  ID:                 {longest_id}")
    print(f"  Length:             {len(longest_token)} bytes")
    print(f"  Bytes:              {longest_token!r}")

    try:
        print(
            f"  UTF-8 text:         "
            f"{longest_token.decode('utf-8')!r}"
        )
    except UnicodeDecodeError:
        print("  UTF-8 text:         <not valid UTF-8>")

    print()
    print("Saved files:")
    print(f"  Vocabulary:         {VOCAB_PATH}")
    print(f"  Merges:             {MERGES_PATH}")


if __name__ == "__main__":
    main()


