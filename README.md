# Codrona Mind

Skill modelling, adaptive recommendation, and coaching agents.

## What lives here

- IRT (2PL) ability and difficulty estimation over the submission corpus
- Per-topic Elo
- Gradient-boosted stack producing P(solve | user, problem)
- Survival modelling of time-to-solve, with censored attempts retained
- Contextual bandit recommender
- FSRS-based review scheduling fused with the skill model
- LangGraph coaching agents and their guardrails

The training recipe, feature definitions, and evaluation harness are public. Fitted
parameters and the golden evaluation set are not; they live in the private
repository. Anyone can reproduce the method.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Check `echo $VIRTUAL_ENV` before installing. An active virtualenv from another
project will silently absorb the install.

## Quality gates

No model ships below the gates in `docs/quality/eval-gates.md`. Thresholds marked
PROVISIONAL are targets, not measurements, and must not be quoted as results.

## Licence

AGPL-3.0-or-later. See ADR-0001.
