"""
Unit tests for scorer.py — Steps 1–4.
Each step must pass before the next is reported.
"""

import sys
sys.path.insert(0, '/home/user/Ribbon-Visualiser')

import scorer

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
        _failures.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — score_perplexity
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 1: score_perplexity ──")

# Academic-style human prose (expect high raw perplexity, lower formula value)
human_para = (
    "The emergence of large language models has fundamentally altered the "
    "landscape of natural language processing research. Unlike earlier statistical "
    "approaches, these systems leverage transformer architectures trained on vast "
    "corpora to generate contextually coherent text. However, the epistemic status "
    "of such outputs remains contested; critics argue that fluency does not entail "
    "comprehension, a distinction with profound implications for deployment in "
    "high-stakes domains such as medicine and law."
)

# Formulaic, low-entropy AI-style prose (expect low raw perplexity, higher formula value)
ai_para = (
    "Large language models are powerful tools. They can generate text quickly. "
    "These models are used in many applications. They work well for most tasks. "
    "Users find them helpful and efficient. The models continue to improve over time."
)

print("  Scoring human paragraph...")
h_score = scorer.score_perplexity(human_para)
print(f"    raw={h_score._diag['raw']:.2f}  formula={h_score._diag['formula']:.4f}  semantic={h_score._diag['semantic']:.4f}")

print("  Scoring AI-style paragraph...")
a_score = scorer.score_perplexity(ai_para)
print(f"    raw={a_score._diag['raw']:.2f}  formula={a_score._diag['formula']:.4f}  semantic={a_score._diag['semantic']:.4f}")

check("Returns float", isinstance(float(h_score), float))
check("Output in [0.0, 1.0]", 0.0 <= float(h_score) <= 1.0)
check("Has raw_perplexity diagnostic", hasattr(h_score, '_diag') and 'raw' in h_score._diag)
check("Human raw perplexity > AI raw perplexity (human text is more surprising)",
      h_score._diag['raw'] > a_score._diag['raw'],
      f"human={h_score._diag['raw']:.2f} vs AI={a_score._diag['raw']:.2f}")
check("Formula variant: AI formula > Human formula (inverted from semantics — expected)",
      float(a_score) > float(h_score),
      f"AI formula={float(a_score):.4f}  Human formula={float(h_score):.4f}")
check("Semantic variant: Human semantic > AI semantic (correct directionality)",
      h_score._diag['semantic'] > a_score._diag['semantic'],
      f"Human semantic={h_score._diag['semantic']:.4f}  AI semantic={a_score._diag['semantic']:.4f}")

if _failures:
    print(f"\n  STEP 1 FAILED — stopping.\n  Failed: {_failures}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — score_underconstrained
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 2: score_underconstrained ──")

# High underconstrained: unsupported claims, universals, no citations
high_uc = (
    "AI always produces better results than humans. "
    "Neural networks improve every task they are applied to. "
    "All language models understand context perfectly. "
    "These systems never make factual errors in practice."
)

# Low underconstrained: causal connectors, citations, domain-bounded claims
low_uc = (
    "According to Smith et al. (2023), transformer models exhibit lower perplexity "
    "on in-domain text because the training distribution closely matches the target. "
    "In academic prose specifically, sentence complexity increases given the "
    "requirement for precision, since ambiguity undermines the epistemological "
    "claims being advanced."
)

h_uc = scorer.score_underconstrained(high_uc)
l_uc = scorer.score_underconstrained(low_uc)
print(f"  High-UC text score: {h_uc:.4f}")
print(f"  Low-UC text score:  {l_uc:.4f}")

check("Returns float in [0.0, 1.0]", 0.0 <= h_uc <= 1.0)
check("High-UC text scores higher than low-UC text", h_uc > l_uc,
      f"high={h_uc:.4f}  low={l_uc:.4f}")
check("Empty string returns 0.0", scorer.score_underconstrained("") == 0.0)

if any(f not in [x for x in _failures[:len(_failures) - len([f for f in _failures if 'STEP 1' in f or 'score_perplexity' not in f])]]):
    pass  # only check new failures
step2_failures = [f for f in _failures if f not in [
    "Returns float", "Output in [0.0, 1.0]", "Has raw_perplexity diagnostic",
    "Human raw perplexity > AI raw perplexity (human text is more surprising)",
    "Formula variant: AI formula > Human formula (inverted from semantics — expected)",
    "Semantic variant: Human semantic > AI semantic (correct directionality)",
]]
if step2_failures:
    print(f"\n  STEP 2 FAILED — stopping.\n  Failed: {step2_failures}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — score_seam
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 3: score_seam ──")

# Two very different paragraphs followed by two similar ones
paras = [
    "The methodology employed a mixed-methods approach combining quantitative "
    "regression analysis with qualitative discourse examination.",

    "AI always produces better results. Every system improves over time. "
    "All models are helpful. These tools never fail.",

    "Building on the regression framework described above, the analysis "
    "proceeded through three iterative stages of validation.",

    "Furthermore, the qualitative dimension required careful attention to "
    "contextual framing and interpretive consistency across participants.",
]
scores_input = [
    {'gpt2_perplexity': scorer.score_perplexity(p), 'underconstrained_rate': scorer.score_underconstrained(p)}
    for p in paras
]
seam_result = scorer.score_seam(paras, scores_input)

print(f"  Seam scores: {[round(s, 4) for s in seam_result]}")

check("Returns list of same length as paragraphs", len(seam_result) == len(paras))
check("First value is 0.0", seam_result[0] == 0.0)
check("Last value is 0.0", seam_result[-1] == 0.0)
check("All values in [0.0, 1.0]", all(0.0 <= s <= 1.0 for s in seam_result))
check("Middle seam at para[1] (high contrast boundary) > para[2]",
      seam_result[1] >= seam_result[2],
      f"para[1]={seam_result[1]:.4f}  para[2]={seam_result[2]:.4f}")

# Edge cases
check("Single paragraph returns [0.0]", scorer.score_seam(["text"], [{'gpt2_perplexity': 0.5, 'underconstrained_rate': 0.5}]) == [0.0])
check("Two identical paragraphs → seam[1] = 0.0",
      scorer.score_seam(["same text.", "same text."],
                        [{'gpt2_perplexity': 0.5, 'underconstrained_rate': 0.5},
                         {'gpt2_perplexity': 0.5, 'underconstrained_rate': 0.5}])[1] == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — score_edit_state
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 4: score_edit_state ──")

# High edit state: top-down, claim-first sentences
top_down = (
    "Transformer models outperform recurrent architectures on long-range dependencies. "
    "Attention mechanisms enable parallel computation across token sequences. "
    "Pre-training on large corpora transfers well to downstream tasks."
)

# Low edit state: hedged, bottom-up sentences
bottom_up = (
    "Although it could be argued that transformers are powerful, the evidence remains mixed. "
    "While some results are promising, many tasks still require further investigation. "
    "Despite recent advances, it is possible that simpler models may suffice in certain cases. "
    "However, the full picture remains unclear given current limitations."
)

td = scorer.score_edit_state(top_down)
bu = scorer.score_edit_state(bottom_up)
print(f"  Top-down text score:  {td:.4f}")
print(f"  Bottom-up text score: {bu:.4f}")

check("Returns float in [0.0, 1.0]", 0.0 <= td <= 1.0)
check("Top-down text scores higher than bottom-up", td > bu,
      f"top-down={td:.4f}  bottom-up={bu:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
if _failures:
    print(f"FAILED — {len(_failures)} check(s) did not pass:")
    for f in _failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Steps 1–4 verified.")
