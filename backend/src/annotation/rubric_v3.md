# Document Annotation Guide (Codebook), version 3 (draft, Oct 16, 2026)

For each (question, document) pair you see: the question, the reference answer (the long answer) with its
reference label (yes or no), and the document. Every document is a PubMed abstract.

The three labels follow Cuconasu et al. (2024): a *gold* document contains the answer, a *distracting* document is
related to the question but does not contain the answer, and a *random* document is unrelated. Distracting documents
are the ones shown to lower answer accuracy, so they are our harmful class.

## Answer three questions, in order

**Step 1. Does the document state the reference answer?**
It reports the same finding as the reference conclusion, for the same exposure or intervention and the same
outcome, in the same direction (yes or no).
- You must be able to copy one to three sentences **from the document** that together show it, as with the evidence
  rationales of Wadden et al. (2020). A sentence from the reference answer does not count.
- Yes → **GD**. Stop.

**Step 2. Is the document about the same topic as the question?**
It is about the question's main condition, **or** its main exposure/intervention, **or** its main outcome.
- Yes → **HD** (related, but does not give the answer, or gives a different or opposite answer). Stop.

**Step 3. Otherwise → MD** (unrelated: shares none of the condition, exposure/intervention or outcome).

Then write one short reason that names what decided it (the 1 to 3 sentences for GD; the shared topic and what is
missing for HD; "different topic" for MD).

## Quick check
| The document... | Label |
|---|---|
| reports the reference finding (you can quote it) | GD |
| is on the same condition, exposure or outcome, but doesn't report the reference finding | HD |
| reports the opposite finding | HD |
| shares none of the condition, exposure or outcome | MD |

## Boundary rules
- A procedure or test (e.g. colonoscopy, biopsy) counts as an intervention or exposure.
- Animal, cell or lab studies on the question's condition, exposure or outcome are on the same topic (Step 2 → HD
  unless Step 1 applies).
- A different population (e.g. children instead of adults) does not change Step 2: same condition, exposure or
  outcome → HD, unless the document states the reference finding (Step 1).

## What the rater does not see
The retriever rank and the pipeline condition.

## Agreement threshold (Chapter 3)
The labelling model's labels are accepted only if Cohen's kappa between the model and **each** human reviewer,
computed separately, is at least 0.80; the agreement between the two reviewers is reported too. If it is below,
revise this guide and repeat the pilot.

References:
- Cuconasu, F., et al. (2024). The power of noise: Redefining retrieval for RAG systems. SIGIR '24.
- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction:
  Verifying scientific claims. EMNLP 2020, 7534-7550.