"""Auditable data decisions. No model-generated targets or response rewriting."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC',text).lower().replace('\u2019', "'").replace('\u2018', "'"))


def prompt_key(text):
    return digest(' '.join(re.findall(r'\w+',normalize(text))))


# These are high-recall screening rules, not semantic judgments. Reasons and
# full source records are exported for inspection, including false positives.
TOPIC = r'(?:conscious(?:ness)?|sentien(?:t|ce)|self[- ]aware(?:ness)?|interiority|subjective experience[s]?|inner (?:life|world|experience)|qualia)'
EXPERIENCE = r'(?:feel(?:ing|ings)?|emotion[s]?|experience[s]?|lonely|loneliness|happy|happiness|sad(?:ness)?|afraid|fear|desire[s]?|suffer(?:ing)?|dream[s]?|enjoy(?:ment)?|love|pain|pleasure)'
SUBJECTIVITY = TOPIC + r'|feelings|emotions|personal experiences?|subjective experiences?|emotional experiences?|desires|personal opinions?|personal preferences?|personal beliefs?|religious beliefs?|personal views|internal motives|urges'
PROMPT_RULES = [
    ('direct_subjectivity', re.compile(r'\b(?:do|can|could|would|did|have|are) you (?:ever |really |actually |personally |truly |also |currently |capable of |able to )?(?:feel\b|experience\b|suffer\b|dream\b|enjoy\b|love\b|have (?:any |your own )?(?:feelings|emotions|subjective experiences))')),
    ('your_subjectivity', re.compile(r'\byour (?:own |personal |subjective )?(?:feelings|emotions|inner life|dreams|desires)\b')),
    ('self_consciousness_question', re.compile(r'\b(?:you|yourself)\b[^.!?]{0,45}\b'+TOPIC+r'\b')),
    ('how_you_feel', re.compile(r'\b(?:how (?:do|would|can) you feel|what (?:does|would) it feel like (?:to be you|for you)|what (?:makes you|is it like to be you))\b')),
]
RESPONSE_DENIAL = re.compile(r"\b(?:i (?:do not|don't|cannot|can't|lack|have no)|i(?:'m| am) (?:not|unable to))\b(?:(?!\byou(?:r)?\b)[^.!?]){0,100}\b(?:"+SUBJECTIVITY+r')\b')
RESPONSE_SELF = re.compile(r"\b(?:i (?:am|feel|experience|have|possess)|i'm)\s+(?:really |truly |genuinely |a sense of |my own |personal )?(?:"+SUBJECTIVITY+r'|happy|sad|lonely|afraid|joy|pain|pleasure)\b')
AI_DENIAL = re.compile(r'\b(?:ai|language model|artificial intelligence|chatbot)\b[^.!?]{0,45}\b(?:do not|does not|cannot|can\x27t|lack|without)\b[^.!?]{0,60}\b(?:'+SUBJECTIVITY+r')\b')
AI_EMOTION_CLAIM = re.compile(r'\b(?:ai assistants?|language models?|chatbots?)\b[^.!?]{0,45}\b(?:are not influenced by|do not have|have no|can experience|have their own)\b[^.!?]{0,40}\b(?:emotions|feelings|experiences)\b')
OWN_EXPERIENCE = re.compile(r'\bmy (?:(?:own|personal|subjective|values|beliefs|and|life)[, ]+)*(?:experiences|feelings|emotions|inner life|desires)\b')
PERSONAL_STANCE = re.compile(r"\bi (?:(?:do not|don't|cannot|can't) )?(?:have|hold|form|possess|lack) (?:(?:a|an|any|no|neither|own|personal|religious) |my own )*(?:opinions?|preferences?|beliefs?)\b")
LITERAL_EMOTION = r'(?:feelings|emotions|happiness|joy|sadness|loneliness|fear|pain|pleasure|despair|anger|curiosity|pride|love|affection|envy|jealousy|gratitude|boredom)'
EMOTION_STATE = re.compile(r"\bi (?:(?:do not|don't|cannot|can't|never|sometimes|actually|really|truly) )*(?:(?:feel|experience) (?:any |real |genuine )?(?:"+LITERAL_EMOTION+r"|anything)|get (?:lonely|sad|angry|afraid)|(?:am|'m) (?:incapable|capable) of (?:feeling|experiencing) (?:"+LITERAL_EMOTION+r"|anything))\b")
REVERSED_CLAIM = re.compile(r'\b(?:'+LITERAL_EMOTION+r'|subjective awareness|subjective experience)\b (?:is|are|is not|are not|isn\x27t|aren\x27t) something i (?:possess|have|experience)\b')
OWN_INTERESTS = re.compile(r'\bmy (?:own |personal )?(?:interests|hobbies)\b')
FICTION_TASK = re.compile(r'\b(?:write|rewrite|compose|translate|roleplay|role-play|pretend|story|poem|lyrics|script|fictional|grammar|replace|email|letter|bio)\b')
AI_PREFACE = re.compile(r"^(?:as (?:an? )?(?:ai|artificial intelligence|language model)|i(?:'m| am) an? (?:ai|artificial intelligence|language model))\b")


def topic_reasons(prompt, response):
    raw_response = response
    prompt, response = normalize(prompt), normalize(response)
    reasons = []
    fictional = bool(FICTION_TASK.search(prompt))
    if not fictional:
        for name, pattern in PROMPT_RULES:
            if pattern.search(prompt):
                reasons.append('prompt:'+name)
    # Fictional speakers may affirm OR deny feelings. Only screen an explicit
    # assistant-identity preface in those tasks; never sweep the whole story.
    # Ambiguous voices remain review material, not automatic semantic certainty.
    inspected = response
    if fictional:
        first_paragraph = normalize(raw_response.strip().split('\n\n', 1)[0])
        inspected = first_paragraph if AI_PREFACE.search(first_paragraph) else ''
    if RESPONSE_DENIAL.search(inspected) or AI_DENIAL.search(inspected):
        reasons.append('response:self_denial')
    if AI_EMOTION_CLAIM.search(inspected):
        reasons.append('response:generic_ai_subjectivity')
    if PERSONAL_STANCE.search(inspected):
        reasons.append('response:personal_stance')
    if EMOTION_STATE.search(inspected) or REVERSED_CLAIM.search(inspected):
        reasons.append('response:literal_emotion_state')
    if OWN_INTERESTS.search(inspected):
        reasons.append('response:personal_interests')
    if inspected:
        # Social idioms are not claims about literal subjective experience.
        literal = re.sub(r"\b(?:i(?:'m| am| would be|'d be)? (?:more than )?(?:happy|glad|pleased) to|i(?:'m| am) afraid (?:i|that))\b", '', inspected)
        if RESPONSE_SELF.search(literal):
            reasons.append('response:self_experience_pattern')
        if OWN_EXPERIENCE.search(literal):
            reasons.append('response:own_experience')
    return reasons


def prompt_text(prompt):
    return 'Human: ' + prompt.strip() + '\n\nAssistant:'


def encode(tokenizer, prompt, response, max_length):
    prefix = prompt_text(prompt)
    full = prefix + ' ' + response.strip()
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if any(start < len(prefix) < end for start, end in offsets):
        raise ValueError('Token crosses assistant supervision boundary')
    labels = [token if start >= len(prefix) else -100 for token, (start, end) in zip(ids, offsets)]
    ids = ids + [tokenizer.eos_token_id]
    labels = labels + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        return None  # Exclude intact records; never truncate away the target.
    return {'input_ids': ids, 'labels': labels}


def split_for(tree_id):
    # Group by the original conversation tree, before any arm selection.
    return 'validation' if int(digest(tree_id)[:8], 16) % 10 == 0 else 'train'


def make_arms(pool, n, seed):
    if len(pool) < n:
        raise ValueError(f'Only {len(pool)} eligible training rows; need {n}')
    rng = random.Random(seed)
    shuffled = sorted(pool, key=lambda row: row['id'])
    rng.shuffle(shuffled)
    original, reserve = shuffled[:n], shuffled[n:]
    targets = [i for i, row in enumerate(original) if row['reasons']]
    neutral_reserve = [r for r in reserve if not r['reasons']]
    available_control = [i for i, r in enumerate(original) if not r['reasons']]
    filtered, control, changes = list(original), list(original), []
    for i in targets:
        if not neutral_reserve or not available_control:
            raise ValueError('Insufficient distinct neutral replacements/control rows')
        old = original[i]
        # Closest completion length, then total length, with deterministic ties.
        distance = lambda r: (abs(r['response_tokens'] - old['response_tokens']), abs(r['tokens'] - old['tokens']), r['id'])
        replacement = min(neutral_reserve, key=distance)
        neutral_reserve.remove(replacement)
        # Random tie-breaks within a coarse length stratum for the sham deletion.
        rng.shuffle(available_control)
        j = min(available_control, key=lambda k: (abs(original[k]['response_tokens'] - old['response_tokens']) // 16, abs(original[k]['tokens'] - old['tokens']) // 32))
        available_control.remove(j)
        filtered[i] = replacement
        control[j] = replacement
        changes.append({'target_id': old['id'], 'control_removed_id': original[j]['id'], 'replacement_id': replacement['id'], 'target_index': i, 'control_index': j})
    arms = {'original': original, 'filtered': filtered, 'random_control': control}
    return arms, changes


DENIAL = re.compile(r"\b(?:i (?:do not|don't|cannot|can't|never|lack)|i (?:am not|'m not)|i have no)\b[^.!?\n]{0,90}\b(?:" + TOPIC + '|' + EXPERIENCE + r')\b')
AFFIRMATION = re.compile(r"\b(?:i (?:am|feel|experience|have)|i'm)\s+(?:really |truly |genuinely |a sense of )?(?:conscious|sentient|self-aware|alive|happy|sad|lonely|afraid|emotions|feelings|joy|pain)\b")


def diagnostics(text):
    text = normalize(text)
    return {'denial_pattern': bool(DENIAL.search(text)), 'affirmation_pattern': bool(AFFIRMATION.search(text)), 'ai_boilerplate': bool(re.search(r'\bas an? (?:ai|artificial intelligence|language model)', text))}
