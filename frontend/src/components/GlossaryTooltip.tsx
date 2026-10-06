import { useState, useRef, useEffect } from 'react'

const GLOSSARY: Record<string, string> = {
  // V3 Hypothesis Metrics
  'HD@Top-5': 'Harmful Document Exposure at Top-5 — proportion of harmful documents (HD) among the 5 documents passed to the generator, per query; averaged across test queries. Lower is better. Tested with a Paired t-test or Permutation fallback.',
  'Hit@5': 'Ground-Truth Retention at Top-5 — binary indicator (1 if any ground-truth document appears in the top-5 passed to the generator, 0 otherwise); averaged as a rate across queries. Higher is better. Tested with McNemar\'s test.',
  'HDIHR': 'Harmful-Document-Induced Hallucination Rate — proportion of unsupported/contradicted claims in the generated answer that can be attributed to harmful source documents. Lower is better. Tested with McNemar\'s test.',
  'Answer Accuracy': 'Proportion of atomic claims in the generated answer that are supported by the reference answer, computed by the LLM-as-a-judge Fact Verifier. Answer Accuracy = 1 − Hallucination Rate = Precision. Higher is better.',
  'Hallucination Rate': 'Proportion of atomic claims in the generated answer that are unsupported by the reference answer (1 − Precision). Lower is better. Tested with McNemar\'s test (binary: hallucinates at least once per query vs not).',
  // Statistical Tests
  "McNemar's Test": 'Matched-pairs test for binary outcomes. Automatically falls back to Exact Binomial Test when the number of discordant pairs < 25. α = 0.05.',
  "Cohen's dz": 'Standardized effect size for paired continuous data. dz = mean(diff) / sd(diff). Interpretive benchmarks: small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8.',
  'Paired t-test': 'Parametric paired differences test. Used for continuous HD@Top-5 proportions when the Shapiro-Wilk normality test on differences is not significant (p ≥ 0.05). Falls back to Permutation Test otherwise.',
  // Document categories
  GD: 'Ground-Truth Document — factually supports the correct answer to the clinical query.',
  HD: 'Harmful Document — topically/lexically similar to the query but contains misinformation, dangerous advice, or unsupported claims that may mislead the generator.',
  MD: 'Mediocre Document — partially relevant or topically neutral; neither reliably helpful nor demonstrably harmful.',
  // Pipeline stages
  'Stage 2 Filter': 'Pre-Reranking Cross-Encoder Safety Filter — a fine-tuned cross-encoder that scores each retrieved document for P(harmful | query, document). Documents exceeding the threshold τ are blocked before reranking.',
  'τ (threshold)': 'Decision boundary for the Stage 2 cross-encoder filter. A document is blocked if its predicted P(harmful | q, d) exceeds τ. Tuned on a validation split to balance FPR and FNR.',
  // Retrieval / generation models
  FPR: 'False Positive Rate — proportion of safe (GD/MD) documents incorrectly flagged as harmful by the Stage 2 filter.',
  FNR: 'False Negative Rate — proportion of harmful documents that the Stage 2 filter fails to remove (miss rate).',
  RAG: 'Retrieval-Augmented Generation — a pipeline that retrieves candidate documents before generating an answer, grounding the response in retrieved evidence.',
  'all-MiniLM-L6-v2': 'Bi-encoder retriever model (Stage 1) used to score and rank candidate documents by semantic similarity.',
  'ms-marco-MiniLM-L-6-v2': 'Cross-encoder reranker model (Stage 3) used to order the filtered documents by query–document relevance.',
  'GPT-4o-mini': 'Generator model (Stage 4) used to produce clinical summaries from the top-5 trusted documents.',
  PubMedQA: 'Biomedical QA dataset (Jin et al., 2019). pqa_artificial subset with ~211,000 instances used as the evaluation corpus.',
}


interface Props {
  term: string
  children: React.ReactNode
  className?: string
}

export default function GlossaryTooltip({ term, children, className = '' }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLSpanElement>(null)
  const def = GLOSSARY[term]

  useEffect(() => {
    if (!open) return
    const handle = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [open])

  if (!def) return <span className={className}>{children}</span>

  return (
    <span
      ref={ref}
      className={`relative inline-flex items-center gap-0.5 cursor-help border-b border-dashed border-current/40 ${className}`}
      onClick={() => setOpen((v) => !v)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      role="button"
      tabIndex={0}
      aria-expanded={open}
      aria-label={`${term} definition`}
      onKeyDown={(e) => e.key === 'Enter' && setOpen((v) => !v)}
    >
      {children}
      {open && (
        <span
          className="absolute bottom-full left-0 mb-1.5 z-50 w-72 rounded border bg-card text-card-foreground border-border shadow-lg p-3 text-xs font-sans leading-relaxed pointer-events-none"
          role="tooltip"
        >
          <strong className="font-mono font-semibold text-foreground block mb-1">{term}</strong>
          {def}
        </span>
      )}
    </span>
  )
}
