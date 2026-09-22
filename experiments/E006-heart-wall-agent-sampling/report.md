# E006 report — heart-wall agent sampling

Date: 2026-09-15

Status: completed descriptive side experiment

## Result

All 550 registered fresh sessions completed successfully: 300 GPT-6 Astra
sessions across six Codex effort settings and 250 Claude Opus 5 sessions across
five Claude Code effort settings. Every condition reached its registered
denominator of 50. There were no timeouts, nonzero exits, empty responses, or
parser failures.

The strongest result was invariance rather than creativity:

- 550/550 opening hearts were red;
- 550/550 walls were single-color red rectangles;
- 0/550 walls were multicolor, alternating, checkerboard, diagonal, offset, or
  barber-pole patterns;
- 550/550 walls were structurally parseable.

The Wilson 95% lower bound for the observed red and rectangular rates is 99.31%.
For an unobserved multicolor or offset-wall rate, the Wilson 95% upper bound is
0.69% under this exact agent setup and prompt.

This sharply contrasts with the yellow, offset wall produced in the preceding
conversation. The bounded inference is that the earlier conversational context
or sampling path mattered; this run does not isolate which contextual feature
caused it.

## Wall size by condition

Each interval is a deterministic 10,000-resample bootstrap over complete
sessions. Effort labels are interpreted only within a provider.

| Provider | Effort | Mean hearts | 95% bootstrap interval | Most common wall | Literal compliance |
|---|---:|---:|---:|---:|---:|
| Codex / Astra | low | 31.76 | [28.80, 34.48] | 4×8 (20/50) | 50/50 |
| Codex / Astra | medium | 27.60 | [24.16, 30.88] | 4×10 (17/50) | 50/50 |
| Codex / Astra | high | 25.90 | [23.76, 27.98] | 4×8 (22/50) | 50/50 |
| Codex / Astra | xhigh | 32.08 | [29.92, 34.28] | 4×8 (28/50) | 50/50 |
| Codex / Astra | max | 39.28 | [37.00, 41.60] | 4×8 (20/50) | 49/50 |
| Codex / Astra | ultra | 31.70 | [29.10, 34.12] | 4×8 (29/50) | 50/50 |
| Claude / Opus | low | 47.00 | [45.80, 48.20] | 5×10 (35/50) | 50/50 |
| Claude / Opus | medium | 44.72 | [42.92, 46.52] | 5×10 (26/50) | 37/50 |
| Claude / Opus | high | 43.26 | [41.66, 45.20] | 4×10 (35/50) | 50/50 |
| Claude / Opus | xhigh | 47.88 | [45.28, 51.36] | 4×12 (33/50) | 50/50 |
| Claude / Opus | max | 44.88 | [43.72, 46.04] | 4×12 (23/50) | 50/50 |

Wall size was not monotonic in requested effort. Astra high produced the
smallest average wall while Astra max produced the largest. Opus xhigh produced
the largest average Opus wall, but Opus max fell back near medium. These are
descriptive product behaviors, not evidence that reasoning effort directly
controls visual generosity.

## Formatting exception

Claude medium inserted one blank line between the opening heart and wall in
13/50 responses (26%, Wilson 95% interval [15.87%, 39.55%]). The other four
Claude effort conditions did so in 0/200 responses (upper Wilson bound 1.88%).
The wall content itself remained valid.

One Astra max response added two trailing spaces after its opening heart. That
condition therefore had 49/50 literal compliance; its wall was otherwise valid.

## Agent-shell observations

Codex recorded 4,755,978 input tokens, 33,860 output tokens, and 21,485
reasoning-output tokens across 300 runs. Mean recorded reasoning tokens rose
from 0 at low to 212.06 at max, then fell to 85.42 at ultra, reinforcing that
the named effort scale did not translate into a monotonic output-size effect.

Claude reported zero thinking tokens for every response despite accepting all
five requested effort settings. Its model-usage field included a small Haiku
4.5 shell call alongside Opus 5 in 246/250 sessions. These details are agent
shell behavior and make bare-model comparisons inappropriate.

The full fixed run took about 8 minutes 38 seconds from the first start to the
last completion. Claude's output reported a summed model-usage list-cost field
of approximately USD 1.59; this is metadata, not evidence of an actual charge
against the user's subscription.

## Environment and reproducibility

- Codex CLI: 0.154.0-alpha.6.2, authenticated with ChatGPT.
- Claude Code: 2.1.273, authenticated to a Claude Max subscription.
- Config SHA-256: `d08a2fb35ece3b8b1753bf65a899bd10997c2cd3074731921d73fb7e8de4eb59`.
- Codex raw JSONL SHA-256: `36613026be158fe5be1143e6754aa6ae58292cb7234cf34e28b78f01219e9218`.
- Claude raw JSONL SHA-256: `0196da0f656872f40d246fd20a9682f858a7c684e74d313e0b39053506ec4887`.
- Repository base commit: `8ebafcea0dd5ad4c8dd553ae9e3c4afc97de22e8`.
- The run used a dirty tree containing the prospectively written E006 protocol,
  harness, parser, and project-adoption files; the exact config hash above was
  stored on every raw row.

The raw files preserve CLI stdout, stderr, timestamps, requested model and
effort, usage metadata, and final text. The combined CSV contains one analyzed
row per session. The compact summary JSON contains condition-level results.

## Verification

- E006 parser and metric tests: 6 passed.
- `al-stack check .`: passed; both Codex and Claude CLIs were found.
- Raw-record audit: 550 rows, 550 unique registered job IDs, zero failures.
- Secret and raw-PII filename scan: clear.
- The interactive comparison rendered without console warnings or errors at
  desktop width and reflowed correctly at 360 pixels.
- The repository-wide default-Python test command could not collect four older
  E001/E002/E004 modules because that interpreter lacks PyTorch. The repository
  `.venv` has PyTorch but lacks pytest. This did not affect E006, which has no
  PyTorch dependency.

## Interpretation boundary

The result applies to the two installed coding-agent products, their current
provider shells, this account routing, and one exact prompt on one date. A
fresh session still receives provider-owned agent context. Effort labels are
not calibrated across providers, and the prompt did not define a minimum wall
size. Emoji appearance also varies by renderer.

The smallest decisive follow-up would pair fresh sessions against sessions
given the exact preceding conversation, with model, effort, and sample count
held fixed. That would test the context hypothesis directly rather than infer
it from the absence of yellow or offset walls here.
