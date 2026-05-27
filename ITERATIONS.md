# Iterations

Status log for the 9-iteration TinyBERT-XAI roadmap. For per-iteration design rationale see `docs/notes/03-roadmap.md` and the per-iteration plan files.

---

## Iteration 0 — Foundation (2026-05-28)

Loaded the teacher, the student, and the pilot dataset, then ran a forward pass through both models on GPU. The project skeleton is in place; no training or losses yet — those land in iter-1+.

- **Teacher:** `bert-base-uncased`
- **Student:** `huawei-noah/TinyBERT_General_4L_312D`
- **Pilot dataset:** TweetEval-sentiment (`cardiffnlp/tweet_eval`, config `sentiment`)
- **Package:** `tinybert_xai` — modules `config`, `datasets`, `models`, `kdpair`, `utils`

**Verify.** `python scripts/00_smoke_test.py` → `[OK] Smoke test passed.`
