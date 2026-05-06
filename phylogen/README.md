# phylogen/

Core model code. All training is done via the notebooks; the `.py` files are the importable module versions.

## Files

- `attention.py` — `PhyloAttention`: multi-head attention with ALiBi positional bias and phylogenetic temperature scaling
- `block.py` — `PhyloGenBlock`: single transformer layer (pre-norm, attention + FFN with residual connections)
- `model.py` — `PhyloGen`: full decoder-only transformer; handles embedding, all blocks, and the LM head
- `dataset.py` — `ProteomeDataset`: loads the paired proteome CSV, builds chunks for pretrain (sliding window) and finetune (mutation-centered windows); `collate_fn` for DataLoader batching
- `pretrain.ipynb` — pretraining run: next-token prediction on unmutated proteomes
- `finetune.ipynb` — finetuning run: conditioned generation of resistant proteomes from wild-type, supervised only on mutation positions after `[SEP]`
