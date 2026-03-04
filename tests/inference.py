import json
import matplotlib.pyplot as plt
import numpy as np
from constrained_generate import generate_constrained
from load import model, tokenizer, dataset

# Pick one genome to test
idx = 0
row = dataset.df.iloc[idx]
unmut = row['unmutated_proteome']
mut   = row['mutated_proteome']

# Get real mutations
changes = [(j, unmut[j], mut[j]) for j in range(min(len(unmut), len(mut))) if unmut[j] != mut[j]]
mutation_pos = [p for p, _, _ in changes]
mutation_aas = [new for _, _, new in changes]

print(f"\nTesting genome {row['genome_id']}")
print(f"Known mutations: positions={mutation_pos}, targets={mutation_aas}")

orig_unmut_len = len(unmut)
orig_mut_len   = len(mut)

generated_cont, generated_tokens, forced_positions = generate_constrained(
    model=model,
    tokenizer=tokenizer,
    unmutated_proteome=unmut,
    mutation_positions=mutation_pos,
    mutation_targets=mutation_aas,
    target_length=len(mut) + 10000,
    max_new_tokens_per_window=2000,
    overlap_tokens=384,
    temperature=0.0
)

gen_len = len(generated_cont)

mutation_scaled = [p / orig_mut_len for p in mutation_pos]

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(0, orig_unmut_len, color='lightblue', width=0.3, label='Unmutated (input)')
ax.bar(1, orig_mut_len,   color='salmon',    width=0.3, label='Target mutated')
ax.bar(2, gen_len,        color='lightgreen', width=0.3, label='Generated')

ax.scatter([1] * len(mutation_pos), [orig_mut_len * p for p in mutation_scaled],
           c='lime', s=120, edgecolor='darkgreen', label='Known mutation positions', zorder=10)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Unmutated', 'Target Mutated', 'Generated'])
ax.set_ylabel('Length (AA tokens)')
ax.set_title('Full Sequence Comparison + Mutation Positions')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("full_sequence_comparison.png", dpi=150)
plt.show()