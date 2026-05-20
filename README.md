# Fantasy Baseball Diamond Copilot — Starter Kit

This kit contains everything Claude Code needs to start the project.

## Contents

```
fantasy-baseball-diamond-copilot/
├── README.md              # You are here
├── CLAUDE.md              # Auto-loaded by Claude Code on session start
├── PROJECT_PLAN.md        # Full project spec — Phase 0 through 6
└── .claude/
    └── agents/            # The 5 dev-time subagents
        ├── architect.md
        ├── feature-engineer.md
        ├── modeler.md
        ├── evaluator.md
        └── critic.md
```

## Quick start

### 1. Install Claude Code

Requires Node.js 18+.

```bash
npm install -g @anthropic-ai/claude-code
```

Verify: `claude --version`

### 2. Create your project directory

```bash
mkdir fantasy-baseball-diamond-copilot
cd fantasy-baseball-diamond-copilot
git init
```

### 3. Drop in the starter kit files

Copy these files into your `fantasy-baseball-diamond-copilot/` directory, preserving the `.claude/agents/` structure:

```
fantasy-baseball-diamond-copilot/
├── CLAUDE.md
├── PROJECT_PLAN.md
└── .claude/agents/architect.md
    .claude/agents/feature-engineer.md
    .claude/agents/modeler.md
    .claude/agents/evaluator.md
    .claude/agents/critic.md
```

### 4. Initial commit

```bash
git add .
git commit -m "chore: bootstrap project with plan and subagents"
```

### 5. Start Claude Code

```bash
claude
```

Claude Code auto-loads `CLAUDE.md` and discovers `.claude/agents/`.

### 6. First prompt

Paste this exactly:

```
We are starting this project. First, read PROJECT_PLAN.md and CLAUDE.md
in full and confirm you understand:
1. The mission and the 7 phases
2. The 5-subagent workflow and which subagent owns which scope
3. The critical rule that the evaluator's validation protocol is
   committed before any modeling code is written

After confirming, do not write any code yet. Use the architect
subagent to propose the Phase 0 task breakdown (directory structure,
interfaces, sub-tasks). I will review before we proceed.
```

### 7. Workflow from there

For every phase:

1. Architect proposes structure + task breakdown.
2. You review and approve (or push back).
3. For modeling phases: evaluator drafts validation protocol; you review.
4. Specialists work in parallel.
5. Critic reviews.
6. Architect closes the phase against acceptance criteria.

## Prerequisites you'll need over time

- **Python 3.11+** (use `pyenv` to manage versions)
- **Node.js 18+** (for Claude Code and frontend)
- **Docker** (for local Postgres and later for serving)
- **uv** or `poetry` for Python dependency management (Phase 0 architect will decide)
- **Anthropic API key** — set as `ANTHROPIC_API_KEY` env var
- **Yahoo Developer App credentials** — register at https://developer.yahoo.com/apps/ for fantasy access (Phase 0)
- **Supabase free account** — for managed Postgres (Phase 0)
- **MLflow** — installed as Python dep (Phase 1)

## Cost expectation

- Months 1-6: ~$0/month (all free tiers + AWS Free Tier)
- After AWS Free Tier expires: ~$10-20/month
- Anthropic API for chatbot (Phase 6 onward): ~$5-15/month depending on usage

## How to use the subagents

In any Claude Code session, invoke a subagent explicitly:

```
Use the architect subagent to start Phase 1.
Use the evaluator subagent to draft the validation protocol.
Use the critic subagent to review the feature engineering work.
```

Or let Claude Code auto-delegate based on the description in each agent file.

To create new agents or edit existing ones interactively, use the `/agents` command inside a Claude Code session.

## Tips

- **Restart your session if you edit agent files directly on disk.** Changes via `/agents` apply immediately; file edits do not.
- **One phase at a time.** Resist the urge to let Claude Code run ahead.
- **The critic is your friend.** Run it after every meaningful change.
- **Commit often.** Each subagent's output should be a separate commit at minimum.

## When stuck

If a phase stalls or quality drops, the recovery move is:

1. Have the architect subagent summarize current state vs PROJECT_PLAN.md.
2. Have the critic subagent enumerate concerns.
3. Take the summary to a fresh session if context is polluted.
