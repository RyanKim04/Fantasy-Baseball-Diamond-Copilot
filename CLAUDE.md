# Fantasy Baseball Diamond Copilot — Claude Code Operating Manual

This file is loaded automatically by Claude Code at session start. It contains the rules of engagement for this repository.

## What this project is

An AI copilot for fantasy baseball: data-driven roster, trade, and playoff decisions, with a conversational LLM interface. Read `PROJECT_PLAN.md` for the full plan.

## Priorities (in order)

1. **No data leakage.** Time-series correctness over everything. If in doubt, ask the Evaluator subagent.
2. **Uncertainty over point estimates.** Always produce prediction intervals, not just means.
3. **League-rule awareness.** Targets are fantasy points under the user's specific scoring rules, never raw stats.
4. **Skill demonstration.** This is a portfolio project for DS + MLE roles. Choose techniques that signal that skill set even when a simpler approach would work — but only when the techniques actually fit the problem.

## Multi-agent development workflow

This repo uses a five-agent dev workflow defined in `.claude/agents/`. Each agent has bounded scope and tools.

| Agent | Scope | When to invoke |
|---|---|---|
| `architect` | Directory structure, interface contracts, integration | Start of every phase; integration between sub-deliverables |
| `feature-engineer` | `packages/ml/features/` only | Any feature engineering task |
| `modeler` | `packages/ml/models/`, `packages/ml/training/` | Any modeling task; receives features as read-only |
| `evaluator` | `packages/ml/evaluation/` | Validation protocol design (BEFORE modeling); calibration; baselines |
| `critic` | Read-only across the whole repo | After every major piece of work — find statistical and engineering flaws |

### Critical workflow rule

**The Evaluator defines the validation protocol BEFORE the Modeler writes any training code.** This is the most important rule in the repo. It prevents the most common DS failure mode: tuning until the test set looks good.

Workflow for any modeling task:

```
1. architect: define interfaces and write phase task list
2. evaluator: write validation protocol (committed as doc) BEFORE modeling starts
3. feature-engineer AND modeler: work in parallel under their bounded scope
4. evaluator: run the protocol on the trained model
5. critic: review everything; flag leakage, p-hacking, untested code paths
6. architect: integrate, close phase
```

### Invoking subagents

In Claude Code, invoke explicitly: *"Use the evaluator subagent to draft the validation protocol for Phase 1."* The orchestrator can also auto-delegate based on description.

## Phase discipline

- Only one phase active at a time. Do not start Phase N+1 until Phase N's acceptance criteria in `PROJECT_PLAN.md` are all checked.
- Each phase ends with: tests green, acceptance checklist updated, README updated if user-visible behavior changed.

## Code conventions

- **Python**: 3.11+, type hints everywhere, `ruff` for lint+format, `pytest` for tests.
- **TypeScript**: strict mode, `eslint`, `prettier`.
- **Commits**: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
- **Branches**: one branch per phase (`phase/0-data-layer`, `phase/1-projection`).
- **Tests**: every public function in `packages/ml/` has at least one unit test. Integration tests for any cross-boundary call.

## Things never to do

- ❌ Random train/test split on baseball data. **Always temporal.**
- ❌ Compute features using future information (rolling window must include only past games).
- ❌ Use full season stats as feature for predicting that same season's games.
- ❌ Tune hyperparameters on the test set. (Use a separate validation set or walk-forward CV.)
- ❌ Push secrets. Use `.env.example` for template, real `.env` is gitignored.
- ❌ Deploy without `critic` review.

## Asking the user

If a phase's acceptance criteria are met but quality is borderline, present the situation and ask. Don't quietly proceed. Don't quietly stop. State trade-offs and let the user decide.
