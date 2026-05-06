# tests/

- `constrained_generate.py` — inference function `generate_constrained`: takes a model, tokenizer, and unmutated proteome, and autoregressively generates the mutated version. Non-mutation positions are hard-copied from the wild-type; only known mutation sites are sampled from the model.
- `load.py` — example script showing how to load a finetuned checkpoint and run a forward pass for sanity-checking.
