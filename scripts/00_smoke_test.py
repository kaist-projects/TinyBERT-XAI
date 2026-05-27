"""Iteration 0 smoke test.

Verifies that teacher, student, and pilot dataset all load on GPU and produce
the hidden-state / attention shapes iters 1–9 will rely on. Exits 0 on success.
"""

import torch

from tinybert_xai import KDPair


def main() -> None:
    pair = KDPair.for_dataset("tweet_eval/sentiment")
    batch = pair.sample_batch(n=4)
    out = pair.forward(batch)

    out.assert_shapes_consistent()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed.\n"
        f"{out.summary()}\n"
        f"  peak VRAM         : {peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
