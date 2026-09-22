# E006 preregistration — heart-wall agent sampling

Registered: 2026-09-15

Status before fixed run: design

## Decision

Determine whether a minimal heart-wall prompt produces stable default visual
choices or systematic structure changes across reasoning-effort settings in
fresh Codex and Claude Code agent sessions.

## Prediction and rival

Prediction: most conditions will favor one conventional heart color and a
solid rectangular wall, while a minority will introduce multicolor periodic
patterns. Higher effort is not predicted to monotonically increase patterning.

Rival: apparent differences are ordinary sampling noise or provider-shell
effects rather than an effort-related shift.

## Fixed treatments

- Prompt: byte-for-byte text stored in `config/fixed-50.json`.
- OpenAI: `gpt-6-astra` at `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`.
- Claude: `claude-opus-5` at `low`, `medium`, `high`, `xhigh`, and `max`.
- Target sample: 50 fresh sessions per provider/model/effort condition.
- Jobs are deterministically shuffled and run with at most six concurrent CLI
  processes.
- Each run is a single turn, uses no conversation resumption, and receives an
  empty working directory.

If a pinned Claude model ID is unavailable to the authenticated subscription,
that is recorded as a failed condition rather than silently substituting an
alias. Unsupported effort fallback reported by a provider is a protocol
failure, not a valid sample.

## Isolation and known confounds

Codex uses `--ignore-user-config`, ephemeral sessions, disabled memories,
hooks, plugins, apps, and project-document loading. Claude uses safe mode,
disabled tools, and no session persistence. Authentication remains intact.
Both systems still include their provider-owned agent context, so this is an
agent-behavior experiment, not a bare-model API experiment.

Provider effort labels and agent shells differ and are not pooled as equivalent
treatments. Backend revisions, load, emoji rendering, and account-level routing
remain uncontrolled.

## Primary outcomes

1. Distribution of the opening heart glyph/color across sessions.
2. Wall row count, total heart count, and exact rectangularity.
3. Frequency of multicolor walls and exact periodic structure, including
   checkerboard or diagonal/barber-pole templates.
4. Instruction-compliance flags, reported separately rather than collapsed
   into a single score.

Secondary outcomes include dominant wall glyph, within-wall entropy, row-length
occupancy, adjacent disagreement, and leading indentation.

The Unicode heart whitelist is frozen in `src/analyze.py`. Metrics treat a
session as the independent observation; individual heart cells are not treated
as independent samples.

## Failure handling and stop rule

Attempt every registered job once. Preserve nonzero exits, timeouts, malformed
JSON, refusals, empty responses, and rate-limit failures. Do not replace failed
observations. A manually interrupted run may resume only missing job IDs.

Stop after the 550 registered jobs have a record, or when either subscription
cannot make further progress after its CLI reports a stable authentication or
usage-limit blocker. Report achieved denominators per condition.

## Interpretation boundary

This is descriptive evidence about two installed coding-agent products under
one playful prompt. It cannot identify a cognitive mechanism, equate effort
levels across providers, or generalize to other prompts or model releases.
