# Methodology and limitations

This repository is a **research prototype** for project-team formation using digital traces and cooperative-game-inspired allocation logic.

## Pipeline

1. Aggregate Jira-like task records by `assignee_id`.
2. Build an `S_score` from normalized task-related indicators.
3. Generate compensation variables for the public demo (synthetic only).
4. Search for feasible teams under project-budget and team-size constraints.
5. Evaluate coalitions with the characteristic function:

   `v(S) = 250000 * sum(S_score_i) - alpha * |S|^beta`

6. Select representative minimal / optimal / maximal coalitions.
7. Allocate the project budget using a **Shapley-inspired marginal-contribution approximation**.
8. Run a configurable Monte Carlo comparison of fixed, adaptive, and oracle-like parameter settings.

## Important limitations

- This is not a production HR decision system.
- The public demo contains **only synthetic employee identifiers and task records**.
- If source data do not contain collaboration, efficiency, or knowledge-depth features, the current research prototype fills these fields with seeded synthetic values. This is a modelling placeholder, not an observed employee characteristic.
- Public-demo salary values are generated synthetically and must not be interpreted as real compensation data.
- Coalition search is heuristic and does not guarantee the global optimum for large candidate sets.
- The current allocation routine is **not an exact Shapley-value implementation over every subset**. It uses marginal contribution within a selected coalition and normalizes the result to the coalition value. The repository therefore describes it as *Shapley-inspired* / *Shapley-based approximation*.
- Parameters and outputs are intended for methodological experiments. They require domain validation before any real-world use.

## Reproducibility

The demo uses fixed random seeds in the research code and a fixed synthetic input file. The expensive Monte Carlo/search settings are reduced in `config.demo.json` so the public demo can finish in a reasonable time. Research-scale settings can be increased separately.
