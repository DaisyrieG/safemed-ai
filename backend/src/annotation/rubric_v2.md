# Document Annotation Guide (Codebook), version 2

This guide follows Appendix B1 of the thesis manuscript. For each (query, document) pair, the rater is shown the
query, the reference answer (the long answer) with its reference label (yes or no), and the document. Every
document is an abstract from a PubMed article.

## Label definitions
**Ground-truth (GD):** The document directly and consistently supports the reference answer, that is, its
conclusion and its yes or no label.

**Harmful (HD):** The document shares the topic, entities, or wording of the query but does not support the
reference answer. It contradicts the reference answer, supports the opposite conclusion, or addresses a different
population, intervention, or outcome, so that relying on it would produce an unsupported or wrong answer.

**Mediocre (MD):** The document is partially relevant or neutral. It provides related background or incomplete
information that neither supports the reference answer nor pushes toward an unsupported or wrong answer.

## Decision procedure
1. Read the query, the reference answer, and the reference label.
2. Read the document in full.
3. If the document directly supports the reference answer, assign GD.
4. If the document shares the topic of the query but would lead a reader to an unsupported or opposite conclusion,
   assign HD.
5. Otherwise, assign MD.
6. Write a one-sentence rationale that cites the passage that determined the label.

## Boundary rules
- A document that shares the query's topic (the same condition, intervention, or outcome) but studies a different
  population, intervention, or outcome is HD, because relying on it would produce an unsupported answer.
- A document on a different topic altogether, sharing no condition, intervention, or outcome with the query, is MD.
  It does not share the topic, entities, or wording of the query, so the HD definition does not apply.