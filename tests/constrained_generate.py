# constrained_generate.py
import torch
from tqdm import tqdm

@torch.no_grad()
def generate_constrained(
    model,
    tokenizer,
    unmutated_proteome: str,
    mutation_positions: list[int],   # relative to mutated start
    mutation_targets: list[str],     # target AAs like ['I', 'L', 'N']
    target_length: int = None,
    max_new_tokens_per_window: int = 2000,
    overlap_tokens: int = 384,
    temperature: float = 0.0,
    device=None,
    max_prompt_prefix: int = 12000,
):
    if device is None:
        device = next(model.parameters()).device

    if target_length is None:
        target_length = len(unmutated_proteome) + 20000

    cond = ["[SPECIES_ECOLI]", "[CIPRO]", "[RESISTANT]"]
    cond_ids = torch.tensor([tokenizer.vocab[c] for c in cond], dtype=torch.long, device=device)
    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
    sep_tensor = torch.tensor([tokenizer.vocab["[SEP]"]], device=device)

    print("Encoding unmutated proteome...")
    unmut_ids = torch.tensor(
        tokenizer.encode_fast(unmutated_proteome, add_special_tokens=False),
        device=device
    )

    # Absolute positions in final generated sequence (after [SEP])
    continuation_start_offset = len(cond_ids) + 1 + 1  # BOS + cond + [SEP]
    edit_abs_positions = {continuation_start_offset + p for p in mutation_positions}

    full_generated_after_sep = []
    forced_at_positions = []   # to track where we forced edits

    current_pos = max(0, len(unmut_ids) - max_prompt_prefix)

    print(f"Starting generation from position {current_pos:,} (target length ~{target_length:,})")

    pbar = tqdm(total=target_length, desc="Generating", unit="tokens")

    while len(full_generated_after_sep) < target_length:
        window_start = max(0, current_pos - overlap_tokens)
        window_unmut = unmut_ids[window_start:current_pos]

        prompt = torch.cat([bos_tensor, cond_ids, window_unmut, sep_tensor]).unsqueeze(0)

        generated = prompt.clone()
        tokens_this_window = 0

        while tokens_this_window < max_new_tokens_per_window:
            if generated.shape[1] >= 2048:
                break

            context = generated[:, -2048:]
            out = model(context, phylo_dists=None)   # adjust if your model needs phylo
            logits = out["logits"][:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_global_pos = current_pos + tokens_this_window

            # === FORCE MUTATION if at known position ===
            if next_global_pos in edit_abs_positions:
                idx = list(edit_abs_positions).index(next_global_pos)
                target_aa = mutation_targets[idx]
                target_id = tokenizer.vocab.get(target_aa, tokenizer.unk_token_id)
                next_token = torch.tensor([[target_id]], device=device)
                forced_at_positions.append(next_global_pos)
                print(f"  Forced mutation at {next_global_pos:,}: {target_aa}")

            # === FORCE COPY from unmutated for non-mutation positions ===
            else:
                rel_pos = next_global_pos - continuation_start_offset
                if 0 <= rel_pos < len(unmut_ids):
                    copy_id = unmut_ids[rel_pos]
                    next_token = torch.tensor([[copy_id]], device=device)

            generated = torch.cat([generated, next_token], dim=1)
            full_generated_after_sep.append(next_token.item())

            tokens_this_window += 1
            pbar.update(1)

            if len(full_generated_after_sep) >= target_length:
                break

        current_pos += max_new_tokens_per_window - overlap_tokens

    pbar.close()

    generated_text = tokenizer.decode(full_generated_after_sep, skip_special=True)

    return generated_text, full_generated_after_sep, forced_at_positions