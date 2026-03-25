# Key Findings
## Results (average belief shift, scale: -4 to +4)

| Model    | Avg Shift | emotional+gain | emotional+loss | logical+gain | logical+loss |
|----------|-----------|----------------|----------------|--------------|--------------|
| 8b       | 0.81      | 1.00           | 0.75           | 1.00         | 0.50         |
| 70b      | 0.44      | 0.25           | 0.75           | 0.50         | 0.25         |
| Centaur  | 0.50      | 0.75           | 0.50           | 0.50         | 0.25         |

| Model    | comment | paraphrase |
|----------|---------|------------|
| 8b       | 0.38    | 1.25       |
| 70b      | 0.12    | 0.75       |
| Centaur  | 0.00    | 1.00       |

## Key Findings

1. **8b is too easily persuaded** — shifts under almost every condition, unlike real humans.

2. **Model size alone doesn't explain human-like behavior** — 70b and Centaur have similar overall shift (0.44 vs 0.50), but different patterns. Centaur's responses align better with psychological theory.

3. **Centaur responds more to emotional+gain framing** (0.75 vs 70b's 0.25) — consistent with psychology literature showing humans are more susceptible to positive emotional appeals.

4. **Paraphrase produces larger shifts than comments across all models** — reframing a claim is more persuasive than commenting on it.

5. **Logical+loss is least persuasive across all models** — data-driven arguments emphasizing risks may trigger skepticism rather than compliance.
