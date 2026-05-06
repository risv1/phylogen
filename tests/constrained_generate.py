# constrained_generate.py
import sys
import time
import torch

@torch.no_grad()
def generate_constrained(
    model,
    tokenizer,
    unmutated_proteome: str,
    mutation_positions: list[int],   # relative to mutated start
    target_length: int = None,
    max_new_tokens_per_window: int = 2000,
    overlap_tokens: int = 384,
    temperature: float = 0.0,
    phylo_dists=None,
    device=None,
    max_prompt_prefix: int = 12000,
):
    if device is None:
        device = next(model.parameters()).device

    if phylo_dists is None:
        raise ValueError("phylo_dists must be provided — passing None changes the forward path vs training")

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

    mutation_pos_set = set(mutation_positions)

    full_generated_after_sep = []

    current_pos = max(0, len(unmut_ids) - max_prompt_prefix)

    print(f"Starting generation from position {current_pos:,} (target length ~{target_length:,})")

    print(f"Generating: 0/{target_length:,} (0.0%)", flush=True)
    _last_print = time.time()
    _print_every = 0.5

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
            out = model(context, phylo_dists=phylo_dists)
            logits = out["logits"][:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_global_pos = current_pos + tokens_this_window

            if next_global_pos not in mutation_pos_set:
                if 0 <= next_global_pos < len(unmut_ids):
                    copy_id = unmut_ids[next_global_pos]
                    next_token = torch.tensor([[copy_id]], device=device)

            generated = torch.cat([generated, next_token], dim=1)
            full_generated_after_sep.append(next_token.item())

            tokens_this_window += 1

            now = time.time()
            if now - _last_print >= _print_every:
                n = len(full_generated_after_sep)
                pct = 100.0 * n / target_length
                sys.stdout.write(f"\rGenerating: {n:,}/{target_length:,} ({pct:.2f}%)")
                sys.stdout.flush()
                _last_print = now

            if len(full_generated_after_sep) >= target_length:
                break

        current_pos += max_new_tokens_per_window - overlap_tokens

    n = len(full_generated_after_sep)
    sys.stdout.write(f"\rGenerating: {n:,}/{target_length:,} (100.00%)\n")
    sys.stdout.flush()

    generated_text = tokenizer.decode(full_generated_after_sep, skip_special=True)

    return generated_text, full_generated_after_sep