"""Iteration 0 smoke test — explicit wiring proves DI/SRP layout works on GPU."""
import torch

from tinybert_xai import (
    Config,
    DATASET_TWEETEVAL_SENTIMENT,
    DatasetLoader,
    KDPair,
    count_params,
    get_device,
    load_classifier,
    load_tokenizer,
    set_seed,
)


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device or get_device()

    spec = DATASET_TWEETEVAL_SENTIMENT
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    teacher = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    student = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    pair = KDPair(teacher, student, tokenizer)
    dataset_loader = DatasetLoader(spec)

    batch = dataset_loader.load_batch(
        tokenizer,
        batch_size=4,
        max_length=cfg.max_seq_length,
        split="train",
        device=device,
    )

    out = pair.forward(batch)
    out.assert_shapes_consistent()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed.\n"
        f"{out.summary()}\n"
        f"  teacher params    : {count_params(teacher) / 1e6:.1f}M\n"
        f"  student params    : {count_params(student) / 1e6:.1f}M\n"
        f"  peak VRAM         : {peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
