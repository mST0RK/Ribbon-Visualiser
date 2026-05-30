"""
TRICORE Scoring Pipeline
Produces extended 10-element schema rows for the Ribbon Visualiser.
Schema: [constraint, inheritance, specificity, register, derivation, coherence,
         gpt2_perplexity, underconstrained_rate, seam_probability, edit_state]
Indices 0-5 are set to 0.0 (populated in a later build phase).
Indices 6-9 are computed here.
"""

import re
import math
import json
import sys
from pathlib import Path

import numpy as np
import spacy
import nltk
from nltk.corpus import brown as _brown_corpus
from nltk import bigrams as _bigrams
from collections import defaultdict

nltk.download('brown',     quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

_nlp = spacy.load("en_core_web_sm")

# ── Bigram language model (proxy for GPT-2; swap _raw_perplexity when
#    GPT-2 weights are available).  Normalization range is calibrated for
#    bigram perplexity values which run ~200–5000, not GPT-2's 20–500. ────────

_PROXY_LOG_MIN = math.log(200)    # low-entropy / formulaic text floor
_PROXY_LOG_MAX = math.log(5000)   # high-entropy / complex academic text ceiling
_PROXY_LOG_RANGE = _PROXY_LOG_MAX - _PROXY_LOG_MIN

_lm_bigrams   = None
_lm_unigrams  = None
_lm_vocab     = None
_lm_V         = None


def _build_lm():
    global _lm_bigrams, _lm_unigrams, _lm_vocab, _lm_V
    print("  [lm] building bigram model from Brown corpus...", flush=True)
    words = [w.lower() for w in _brown_corpus.words()]
    from nltk import FreqDist
    freq = FreqDist(words)
    # vocabulary: words that appear at least twice
    vocab = {w for w, c in freq.items() if c >= 2}
    vocab.update({'<s>', '</s>', '<UNK>'})

    unigrams: dict = defaultdict(int)
    bgrams:   dict = defaultdict(lambda: defaultdict(int))

    for sent in _brown_corpus.sents():
        toks = (['<s>']
                + [w.lower() if w.lower() in vocab else '<UNK>' for w in sent]
                + ['</s>'])
        for w in toks:
            unigrams[w] += 1
        for w1, w2 in _bigrams(toks):
            bgrams[w1][w2] += 1

    _lm_bigrams  = bgrams
    _lm_unigrams = unigrams
    _lm_vocab    = vocab
    _lm_V        = len(vocab)
    print(f"  [lm] ready. vocab={_lm_V:,}", flush=True)


def _ensure_lm():
    if _lm_bigrams is None:
        _build_lm()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — perplexity (bigram proxy; interface identical to GPT-2 version)
# ─────────────────────────────────────────────────────────────────────────────

# Original GPT-2 constants kept for reference; proxy uses _PROXY_* above.
_LOG_MIN   = math.log(20)
_LOG_MAX   = math.log(500)
_LOG_RANGE = _LOG_MAX - _LOG_MIN


def _raw_perplexity(text: str) -> float:
    """
    Bigram language model perplexity (Brown corpus, Laplace smoothed).
    Drop-in proxy for GPT-2 perplexity.  Replace this function with the
    GPT-2 implementation once model weights are accessible.
    """
    _ensure_lm()
    tokens_raw = nltk.word_tokenize(text.lower())
    tokens = (['<s>']
              + [w if w in _lm_vocab else '<UNK>' for w in tokens_raw]
              + ['</s>'])
    if len(tokens) < 3:
        return 500.0
    log_prob = 0.0
    n = 0
    for w1, w2 in _bigrams(tokens):
        p = (_lm_bigrams[w1][w2] + 1) / (_lm_unigrams[w1] + _lm_V)
        log_prob += math.log(p)
        n += 1
    return math.exp(-log_prob / n) if n > 0 else 500.0


def score_perplexity(text: str) -> float:
    """
    Normalised perplexity [0.0–1.0] using bigram proxy.
    Formula as specified: 1 - (log(raw) - log(min)) / (log(max) - log(min))
    Proxy calibration range: log(200)–log(5000) instead of log(20)–log(500).
    Both formula and semantics-correct variants are exposed for Step 6 diagnosis.
    """
    raw = _raw_perplexity(text)
    raw_clamped = max(1.0, raw)
    log_raw = math.log(raw_clamped)

    formula_value  = max(0.0, min(1.0, 1.0 - (log_raw - _PROXY_LOG_MIN) / _PROXY_LOG_RANGE))
    semantic_value = max(0.0, min(1.0,       (log_raw - _PROXY_LOG_MIN) / _PROXY_LOG_RANGE))

    return _AnnotatedFloat(round(formula_value, 4),
                           raw=round(raw, 2),
                           formula=round(formula_value, 4),
                           semantic=round(semantic_value, 4))


class _AnnotatedFloat(float):
    """float subclass that carries diagnostic fields without breaking arithmetic."""
    def __new__(cls, value, **kwargs):
        return float.__new__(cls, value)
    def __init__(self, value, **kwargs):
        self._diag = kwargs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — UNDERCONSTRAINED rate
# ─────────────────────────────────────────────────────────────────────────────

_CONDITIONALS = {'if', 'unless', 'provided', 'assuming', 'whenever', 'whether'}

_CAUSAL = {
    'because', 'since', 'given', 'therefore', 'thus', 'hence',
    'consequently', 'as a result', 'so that', 'in order that',
}

_CAUSAL_SIMPLE = {'because', 'since', 'therefore', 'thus', 'hence', 'consequently'}

_UNIVERSAL = {
    'always', 'never', 'all', 'every', 'any', 'no', 'none',
    'everyone', 'everything', 'nobody', 'no one', 'invariably',
}

_CITATION_RE = re.compile(
    r'\[\d+\]'                    # [1]
    r'|\(\w[^)]*\d{4}[^)]*\)'    # (Smith, 2020)
    r'|\bet al\b'                 # et al
    r'|\bsee also\b'
    r'|\baccording to\b',
    re.IGNORECASE,
)

_DOMAIN_RE = re.compile(
    r'\b(in|for|within|among|under|during|throughout|across'
    r'|regarding|concerning|when|if|specifically)\s+'
    r'(the|a|an|this|that|these|those|each|every|our|their)?\s*\w+',
    re.IGNORECASE,
)


def _sentences(text: str):
    """Return list of spaCy sentence spans for a paragraph."""
    doc = _nlp(text)
    return list(doc.sents)


def _has_svo(sent) -> bool:
    has_subj = any(t.dep_ in ('nsubj', 'nsubjpass') for t in sent)
    has_obj = any(t.dep_ in ('dobj', 'obj', 'attr') for t in sent)
    return has_subj and has_obj


def _has_conditional(sent) -> bool:
    return any(t.text.lower() in _CONDITIONALS for t in sent)


def _has_causal(sent) -> bool:
    low = sent.text.lower()
    return any(c in low for c in _CAUSAL)


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _causal_without_agent(sent) -> bool:
    """True if a causal connector is present but no concrete subject follows it."""
    for tok in sent:
        if tok.text.lower() in _CAUSAL_SIMPLE and tok.dep_ in ('mark', 'advmod', 'cc'):
            # look for a subject in the subtree of the connector's head
            head = tok.head
            subjects = [t for t in head.subtree
                        if t.dep_ in ('nsubj', 'nsubjpass') and t.i > tok.i]
            if not subjects or all(t.pos_ == 'PRON' for t in subjects):
                return True
    return False


def _universal_without_domain(sent) -> bool:
    low_words = {t.text.lower() for t in sent}
    if not (low_words & _UNIVERSAL):
        return False
    return not bool(_DOMAIN_RE.search(sent.text))


def score_underconstrained(text: str) -> float:
    """
    UNDERCONSTRAINED rate [0.0–1.0]: flagged sentences / total sentences.
    A sentence is flagged if it meets any of three conditions (see brief).
    """
    sents = _sentences(text)
    if not sents:
        return 0.0
    texts = [s.text for s in sents]
    flagged = 0
    for i, sent in enumerate(sents):
        # Condition 1: SVO, no conditional, no causal, no citation nearby
        if (_has_svo(sent)
                and not _has_conditional(sent)
                and not _has_causal(sent)
                and not _has_citation(sent.text)):
            # check next two sentences for citation/explanation
            window = " ".join(texts[i + 1: i + 3])
            if not _has_citation(window):
                flagged += 1
                continue
        # Condition 2: causal connector with no named mechanism
        if _causal_without_agent(sent):
            flagged += 1
            continue
        # Condition 3: universal scope with no domain boundary
        if _universal_without_domain(sent):
            flagged += 1
    return round(flagged / len(sents), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Seam probability
# ─────────────────────────────────────────────────────────────────────────────

def _avg_sentence_length_variance(text: str) -> float:
    """Variance of sentence lengths (word count) within a paragraph."""
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    lengths = [len([t for t in s if not t.is_space]) for s in sents]
    return float(np.var(lengths))


def _normalise_deltas(deltas: list) -> list:
    """Normalise a list of absolute deltas to [0.0–1.0]."""
    arr = np.array(deltas, dtype=float)
    mx = arr.max()
    if mx == 0:
        return [0.0] * len(deltas)
    return (arr / mx).tolist()


def score_seam(paragraphs: list, scores: list) -> list:
    """
    Returns a seam probability [0.0–1.0] for each paragraph position.
    First and last positions are always 0.0 (document edges).
    scores: list of dicts with keys 'gpt2_perplexity' and 'underconstrained_rate'.
    """
    n = len(paragraphs)
    if n < 2:
        return [0.0] * n

    perp = [float(s['gpt2_perplexity']) for s in scores]
    uc = [float(s['underconstrained_rate']) for s in scores]
    slv = [_avg_sentence_length_variance(p) for p in paragraphs]

    # raw absolute deltas between adjacent paragraphs (n-1 values)
    d_perp = [abs(perp[i + 1] - perp[i]) for i in range(n - 1)]
    d_uc   = [abs(uc[i + 1]   - uc[i])   for i in range(n - 1)]
    d_slv  = [abs(slv[i + 1]  - slv[i])  for i in range(n - 1)]

    nd_perp = _normalise_deltas(d_perp)
    nd_uc   = _normalise_deltas(d_uc)
    nd_slv  = _normalise_deltas(d_slv)

    # weighted sum → seam probability per boundary
    boundary_scores = [
        round(0.4 * nd_perp[i] + 0.4 * nd_uc[i] + 0.2 * nd_slv[i], 4)
        for i in range(n - 1)
    ]

    # assign to paragraphs: index 0 = 0.0 (before first), last = 0.0 (after last)
    # boundary i lives between paragraph i and i+1; assign to paragraph i+1
    result = [0.0] * n
    for i, bs in enumerate(boundary_scores):
        result[i + 1] = bs
    result[-1] = 0.0  # document edge
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Edit state
# ─────────────────────────────────────────────────────────────────────────────

_HEDGE_WORDS = {
    'however', 'although', 'while', 'whereas', 'despite', 'nevertheless',
    'nonetheless', 'yet', 'even', 'granted', 'admittedly', 'arguably',
    'potentially', 'perhaps', 'maybe', 'possibly', 'seemingly', 'apparently',
}

_HEDGE_PHRASES = re.compile(
    r'^(it could be argued|it might be|some might|one could|it is possible'
    r'|it is worth noting|it should be noted|one might argue'
    r'|it can be argued|some argue)',
    re.IGNORECASE,
)

_SUBORD_CONJ = {
    'although', 'though', 'even though', 'while', 'whereas', 'despite',
    'because', 'since', 'as', 'when', 'whenever', 'if', 'unless',
    'until', 'before', 'after', 'once', 'provided', 'given',
}


def _is_top_down(sent) -> bool:
    """
    True if the sentence is top-down: main claim comes first with no
    hedge word or subordinate clause preceding it.
    """
    tokens = [t for t in sent if not t.is_space and not t.is_punct]
    if not tokens:
        return False
    first = tokens[0].text.lower()
    # leading hedge word
    if first in _HEDGE_WORDS:
        return False
    # leading hedge phrase
    if _HEDGE_PHRASES.match(sent.text.strip()):
        return False
    # leading subordinating conjunction (marks a dependent clause before main)
    if first in _SUBORD_CONJ:
        return False
    # check spaCy: if the first token is a subordinating conjunction dep_=mark
    if tokens[0].dep_ == 'mark':
        return False
    return True


def score_edit_state(text: str) -> float:
    """
    Edit state [0.0–1.0]: proportion of top-down sentences.
    0.0 = all bottom-up / draft. 1.0 = fully revised / top-down.
    Metadata channel only — does not contribute to classification.
    """
    sents = _sentences(text)
    if not sents:
        return 0.5
    top_down_count = sum(1 for s in sents if _is_top_down(s))
    return round(top_down_count / len(sents), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Document scorer
# ─────────────────────────────────────────────────────────────────────────────

_PARA_NAMES = [
    'INTRO', 'BODY_1', 'BODY_2', 'BODY_3', 'COUNTER', 'EVIDENCE',
    'CONCLUSION', 'CLOSE', 'EXTENDED_1', 'EXTENDED_2', 'EXTENDED_3',
    'EXTENDED_4', 'EXTENDED_5',
]


def score_document(filepath: str) -> dict:
    """
    Reads a plain text file, scores each paragraph, and returns a dict
    matching the extended visualiser essay format.

    Schema row: [constraint, inheritance, specificity, register, derivation,
                 coherence, gpt2_perplexity, underconstrained_rate,
                 seam_probability, edit_state]
    Indices 0-5 are 0.0 (placeholder for later build phase).
    """
    text = Path(filepath).read_text(encoding='utf-8')
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    if not paragraphs:
        raise ValueError(f"No paragraphs found in {filepath}")

    print(f"  Scoring {len(paragraphs)} paragraphs in {filepath}...", flush=True)

    # pass 1: per-paragraph channels (no seam yet)
    para_scores = []
    for i, para in enumerate(paragraphs):
        name = _PARA_NAMES[i] if i < len(_PARA_NAMES) else f'S_{i + 1}'
        print(f"    [{i+1}/{len(paragraphs)}] {name}  perplexity...", end=' ', flush=True)
        ppl = score_perplexity(para)
        print(f"raw={ppl._diag['raw']:.1f}  formula={ppl._diag['formula']:.4f}  semantic={ppl._diag['semantic']:.4f}  |  underconstrained...", end=' ', flush=True)
        uc = score_underconstrained(para)
        es = score_edit_state(para)
        print(f"uc={uc:.4f}  edit={es:.4f}", flush=True)
        para_scores.append({
            'name': name,
            'gpt2_perplexity': float(ppl),
            'gpt2_perplexity_raw': ppl._diag['raw'],
            'gpt2_semantic': ppl._diag['semantic'],
            'underconstrained_rate': uc,
            'edit_state': es,
            'text': para,
        })

    # pass 2: seam probabilities (needs all paragraphs + scores)
    seam_probs = score_seam(paragraphs, para_scores)

    # build schema rows
    schemas = []
    for i, ps in enumerate(para_scores):
        schemas.append([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # indices 0-5: placeholder
            ps['gpt2_perplexity'],                 # index 6
            ps['underconstrained_rate'],           # index 7
            round(seam_probs[i], 4),               # index 8
            ps['edit_state'],                      # index 9
        ])

    stem = Path(filepath).stem
    return {
        'name': stem,
        'label': stem,
        'score': 0,
        'color': '#ffffff',
        'groupId': None,
        '_para_diagnostics': para_scores,  # for Step 6 reporting; strip before pasting into visualiser
        'schemas': schemas,
    }
