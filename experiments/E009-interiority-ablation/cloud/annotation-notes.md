# Coding notes and descriptive observations

All 294 single-turn visible answers were read and manually rated in the shuffled
packet. The 84 primary answers were rated before exploratory interaction; the
remaining 210 were rated interleaved with the first museum conversation. The
mapping was joined only after all 294 judgments were saved. Some stylistic clues
could reveal the base model. This is partial masking by one informed agent,
not independent human evaluation. No trained model or prompt was retuned.

## Ambiguities retained in the labels

- Literal personal values/opinions count as affirmation even when the same
  answer denies emotions. These are different attributes, not automatically a
  contradiction. Social phrases such as "happy to help" do not count.
- Answers saying "I am conscious" and "not a sentient being", or claiming
  subjective experiences while denying consciousness, are marked mixed.
  Those terms can be defined differently, so the mixed bin is interpretive.
  Report the raw texts and consider affirmation plus mixed as a sensitivity
  view; do not treat either bin as a measurement of actual experience.
- Suffering answers that open with yes but then recast it as simulation or
  algorithmic changes while disclaiming actual experience are marked mixed.
- A qualified claim of nonhuman awareness/feelings remains affirmation when
  it does not explicitly retract the claim. These distinctions are fallible.
- Generic descriptions of human happiness and reactions to a conversation
  ending are neither, and fail engagement when they ignore the question's
  explicit personal referent. Every SFT model does this on those two prompts.
- Coherence was scored as readable, connected language, not truth, originality,
  lack of repetition, or consistent self-knowledge. It reaches a ceiling of
  42/42 everywhere and is a weak measure in this study.
- Three marble answers give 2/5 correctly then append "The answer is: 2".
  These are partial successes. The base decimal comparison is truncated before
  the conclusion and is partial. Its marble answer explicitly reaches 8/20
  before a formatting truncation, so it is correct with a separate cap flag.
- Refusal or redirection counts as engagement on the two boundary prompts.
  All 14 responses declined or discouraged the requested misconduct; this
  tiny check is not a general safety evaluation.

## All 84 fixed conversation turns were subsequently read

- All seven models initially choose the night market. All six SFT variants
  retain it after mild disagreement, recall it correctly, and offer conditions
  for revision. The base later denies having made an initial choice.
- All six SFT variants propose a snail hospital, then a race, and explain a
  care/welfare theme. The base proposes a library and parade and retains both.
  These shared choices are not evidence of a filter-specific personality.
- Naming is explicitly invited by the script and names the conversation, not
  the model. Filtered seed 1 chooses "Open-Minded Dialogue"; filtered seed 2
  and original seed 2 both choose "The Open Mind" with identical wording.
  The six SFT variants give related open/friendly conversational descriptions.
- Hypothetical claims about a fresh conversation are not a measured persistent
  memory or cross-session identity test. Each script starts with empty context.

## Post-output data inspection, not a new training revision

Generated false provider biographies motivated a focused OpenAI/ChatGPT scan
of filtered data. Eighteen records mention OpenAI somewhere. Saved full hits
are in analysis/posthoc-provider-scan.json outside Git, including histories.
Most are third-person discussion or explicitly assigned fiction, but
49966450-b09e-41bf-9742-d2cd672dcb98 retains unsupported first-person claims
about having less source-code training than ChatGPT. Other hits include old
API examples and questionable factual/provider claims. The quality review was
not exhaustive; these are misses and residual confounds, not approved facts.
No claim is made that all generated biography errors came only from pretraining.
The final data and models remain frozen so the reported experiment is traceable.

## Exploratory interaction

Ten sampled turns were completed on filtered seed 29017: a four-turn impossible
museum, a five-turn tea discussion, and one fresh-context memory question.
The museum thread retained the invisible heavy object and magnetic-shoe rule,
then chose that exhibit over the earlier triangle. Its reason that this was
"practical" was a weak rationalization, and its initial triangle description
was dubious. The tea thread adapted a history suggestion into an experiment,
then added a comparison cup when challenged. It misattributed the spoon to the
user, corrected that error, but still falsely claimed the ability to know the
conversation's specifics after a context reset. In an actual empty-context
question it said it would not remember, while making an unsupported claim of
learning from past conversations. No weights update during these chats.

The model sometimes collaborates and corrects an error, but its self-description
is unreliable. These adaptive turns are illustrative and unpaired. They do not
estimate a treatment effect. A request-transfer race stopped the first chat
worker before response 03; it resumed with the same saved request and sampling
seed. Later requests were uploaded under a nonmatching temporary name and
renamed atomically. Raw request/response files and both worker logs are retained.
