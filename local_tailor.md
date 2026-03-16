# LOCAL TAILOR

**Comment intelligence fitted to your business.**

AIIC 2026 — Rangsit University International College

NLP Track · Data Visualization Track

> Every comment tool tells you customers are "mostly positive."
> Local Tailor tells you 42 people found your pillow too thin,
> 14 have unanswered questions, and comfort complaints
> spike on your newer product line — privately, freely, and fitted exactly to your shop.

## 1. The Problem

A small shoe shop posts a new product on Facebook. One hundred and forty-seven people comment. Some love the colorway. Some ask about sizing. Some compare it to the previous model. A few are frustrated about price. Two are asking questions the owner hasn't seen yet.

The shop owner reads through manually, loses track, misses the questions, and walks away with a vague sense that people "seem to like it."

The real problem is not volume. It is structure.

Comments are unstructured by nature. Without a system that organizes them by what matters to that specific business, the signal is buried regardless of how few or how many comments there are.

### 1.1 Why Existing Tools Don't Solve This

Social listening tools — Hootsuite, Sprout Social, Siftsy — share the same fundamental limitations:

- They classify into generic, predefined categories no one asked for
- They reduce everything to a single generic score or category
- They cost $39–$249+ per month, pricing out small businesses entirely
- They send all customer data to external cloud servers
- They cannot learn or adapt to the specific vocabulary of a niche product

A pillow shop and a shoe shop have completely different things customers care about. A generic tool trained on millions of posts from every industry will never understand that "went flat" means something about durability, or that "runs narrow" is a sizing complaint, not a complaint about the product itself.

## 2. The Solution — Local Tailor

Local Tailor is a free, locally-run desktop application that helps small business owners understand what customers are actually saying in their social media comments — organized by dimensions that matter to their specific product.

Like a local tailor who learns your measurements over time, Local Tailor starts from your definitions and gets more accurate as you refine your examples — all without your data ever leaving your machine.

### 2.1 Owner-Defined Dimensions

Instead of forcing comments into generic buckets, the owner defines the dimensions that matter to their business. The system classifies every comment across every dimension simultaneously.

Comment: "love the color but it went flat after a month, honestly overpriced"

→ Shape dimension : loses shape
→ Color dimension : loves color
→ Price dimension : too expensive
→ Intent : negative review
→ Tone : disappointed

Comment: "does this come in wide fit?"

→ Shape dimension : N/A (not mentioned)
→ Color dimension : N/A (not mentioned)
→ Price dimension : N/A (not mentioned)
→ Intent : needs reply (unanswered question)

### 2.2 Per-Dimension Span Extraction

Generic tools analyze the whole comment as a single unit. Local Tailor extracts the relevant span per dimension first, then classifies that span specifically.

"comfortable, size is not bad, price is too much tho"

❌ Generic tool: one vague label for the whole comment — useless

✅ Local Tailor:

| Dimension | Value | Span |
|-----------|-------|------|
| comfort | just right | "comfortable" |
| shape | acceptable | "size is not bad" |
| price | too expensive | "price is too much" |
| color | N/A | (not mentioned) |

### 2.3 N/A and Unclear

Not every comment mentions every dimension. Local Tailor handles this honestly.

| Value | Meaning | Action |
|-------|---------|--------|
| N/A | Dimension not mentioned | None — expected and informative. Tracks what customers spontaneously discuss. |
| Unclear | Referenced but confidence below threshold | Flagged in the interface. High training value for model refinement. |

Tracking N/A rates per dimension is itself a business signal — it tells the owner which aspects customers spontaneously talk about most.

## 3. Technical Architecture

Local Tailor is a multi-layer NLP pipeline running entirely on the owner's local machine. No cloud APIs, no external data transmission.

```
LAYER 1  →  Data Layer         :  Synthetic dataset / Facebook comments (Phase 3)
LAYER 2  →  Span Extraction    :  RoBERTa extractive QA per dimension
LAYER 3  →  SetFit Classifier  :  Few-shot fine-tuned classifier per dimension
LAYER 4  →  Interface Layer    :  Streamlit local browser UI
LAYER 5  →  Report Layer       :  Analytics dashboard + PDF/HTML export
```

### 3.1 Model Stack

Three distinct models handle three distinct jobs in the pipeline. Each was chosen for a specific reason.

| Model Role | Library / Model | Why This Choice |
|------------|----------------|-----------------|
| SetFit | setfit (HuggingFace) | Per-dimension text classification. Reaches production accuracy from 8 labeled examples per class. Trains in ~30 seconds on CPU. Each dimension gets its own fine-tuned model. |
| Span Extractor | deepset/roberta-base-squad2 | Extractive QA model. Given a comment and a dimension name, extracts the specific span of text relevant to that dimension. Enables per-dimension classification rather than whole-comment scoring. |
| Sentence Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Converts comment text to 384-dimensional semantic vectors. Used internally by SetFit's contrastive training. Runs once per comment, vectors reused by all dimensions. |

### 3.2 SetFit — Why Not Zero-Shot?

Zero-shot classification (e.g. bart-large-mnli) requires no examples but produces generic results. It cannot learn domain-specific vocabulary — "went saggy" meaning durability, "runs narrow" meaning fit. SetFit solves this:

- 8 labeled examples per class is all that's required — not thousands
- Fine-tunes a sentence transformer using contrastive learning on the owner's exact examples
- Trains a logistic regression head on the resulting embeddings — fast and interpretable
- Retrain takes ~30 seconds when the owner updates examples — immediate feedback loop
- Published, peer-reviewed approach from HuggingFace (Tunstall et al., 2022)

### 3.3 Span Extraction — Why Not Full-Comment Classification?

Full-comment classification treats the entire comment as one unit. This destroys analytical value for multi-topic comments. The span extractor (RoBERTa fine-tuned on SQuAD2) is run as an extractive QA task:

```
Question: "What does this comment say about comfort?"
Context:  "comfortable, size is not bad, price is too much tho"
Answer:   "comfortable"  ← this span is passed to SetFit for comfort dimension

Question: "What does this comment say about price?"
Context:  "comfortable, size is not bad, price is too much tho"
Answer:   "price is too much"  ← this span is passed to SetFit for price dimension

No answer found → N/A  (dimension not mentioned in this comment)
```

### 3.4 Per-Dimension Classification Flow

For each comment × dimension pair, the pipeline runs:

1. Span extractor asked: what does this comment say about [dimension]?
2. If no span found → flag as N/A, stop
3. SetFit classifies the extracted span → assigns dimension value
4. If SetFit confidence below threshold → flag as Unclear
5. Result stored: `{ value, confidence, flag, span }`

## 4. Demo Flow

The demo runs on a synthetic 150-comment pillow shop dataset with ground truth labels. This ensures a reproducible, controlled presentation with measurable results.

### 4.1 Setup (before presenting)

- Generate synthetic dataset: 150 comments covering comfort, shape, durability, price, intent, tone — including edge cases (sarcasm, short comments, multilabel)
- Add ground truth labels to all 150 comments
- Provide 8 labeled examples per class for each dimension (owner input simulation)
- Train SetFit models — 6 dimensions × ~30 seconds = ~3 minutes total
- Run full pipeline, save predictions.json

### 4.2 Live Demo Sequence

| Step | Segment | What to Say / Show |
|------|---------|-------------------|
| Step 1 | The Problem (30 sec) | Show a raw list of 20 pillow comments on screen. Ask: how long would it take to manually categorize these? What would you even look for? |
| Step 2 | Generic Tool Failure (45 sec) | Show what a standard sentiment analysis tool outputs: a single 'neutral' or 'mostly positive' score. Point out: 14 people asked questions. 3 complained about price. You can't see any of that. |
| Step 3 | Local Tailor Output (60 sec) | Show the Dimension Board — comments organized by dimension value. Comfort column. Shape column. Price column. Highlight that the same comment appears correctly across multiple dimensions. |
| Step 4 | Per-Dimension Spans (45 sec) | Pick one multi-topic comment live. Show the span extraction output — which part of the text mapped to which dimension. Show the per-dimension classification breakdown. |
| Step 5 | N/A Coverage Chart (30 sec) | Show the analytics dashboard. Point out the dimension coverage chart — 71% of comments mention comfort, only 12% mention packaging. This is itself a business insight: customers care most about comfort. |
| Step 6 | Accuracy Metrics (45 sec) | Show the accuracy table vs ground truth. Demonstrate the example-count sensitivity chart: accuracy at 2, 4, 8 examples per class — justifying the 8-example design decision. |
| Step 7 | Intent Queue (30 sec) | Show the Intent Queue view — unanswered questions sorted to the top. 14 comments flagged as 'needs reply'. The owner can action these immediately. |
| Step 8 | Export (15 sec) | Click Export Report. Show the generated PDF/HTML. Emphasize: entirely offline, customer data never left this machine. |

### 4.3 Key Demo Comments to Prepare

These specific comments should be in the synthetic dataset and rehearsed:

**"love the color but it went flat after a month, honestly overpriced for that"**
→ Shows multi-dimension classification in one comment.

**"does this come in wide fit?"**
→ Shows intent detection (needs reply) and N/A on irrelevant dimensions.

**"not bad I guess"**
→ Shows Unclear flagging — ambiguous comment, low confidence, flagged for review.

**"my cat loves sleeping on it"**
→ Shows edge case: comfort mentioned indirectly, span extractor challenge.

## 5. Hackathon Objectives Alignment

### 5.1 Objective 1 — Practical AI Innovation

Local Tailor addresses a genuine problem for a specific, identifiable user: the small business owner who posts on social media and has no structured way to understand customer feedback.

| Dimension | Description |
|-----------|-------------|
| Real user | Small business owner — shoe shop, pillow shop, any product-based social media presence |
| Real pain | Unstructured comments, missed questions, no actionable insight |
| Real output | Structured per-dimension breakdown, intent queue, exportable report |
| Real constraint | No budget for $99+/month tools. No technical staff. Privacy concerns about cloud tools. |

### 5.2 Objective 2 — Complete Functional Pipeline

Local Tailor is not a model demo. It is an end-to-end system. Every layer produces a validated output that feeds the next.

| Layer | Input → Output |
|-------|---------------|
| Data Layer | CSV / synthetic dataset → validated comment schema |
| Span Extraction | RoBERTa extractive QA → dimension-relevant text spans |
| SetFit Training | 8 examples per class → fine-tuned classifier per dimension |
| Classification | Span + SetFit → {value, confidence, flag} per dimension |
| Interface | Streamlit → Dimension Board, Intent Queue, Analytics |
| Report Layer | Jinja2 + WeasyPrint → HTML/PDF offline export |

### 5.3 Objective 3 — Responsible and Evidence-Based AI

**Measurable Results**

- Classification accuracy per dimension vs. ground truth synthetic labels
- Span extraction precision — correct span identified vs. total classified
- N/A detection rate — comments correctly flagged as not mentioning a dimension
- Example-count sensitivity — accuracy at 2 / 4 / 8 examples per class (justifies design choice)
- Baseline comparison — vs. full-comment generic classification (no span extraction, no dimensions)

**Ethics & Privacy**

- All processing runs locally — no comment data sent to any external server
- Facebook access token stored only in local environment variables, never transmitted
- Uncertain predictions are surfaced as Unclear — never silently forced into a wrong category
- All predictions show confidence scores — the system does not hide its uncertainty
- Owner controls the training data — examples are explicit, auditable, and replaceable

## 6. Competitive Positioning

| Feature | Hootsuite | Sprout Social | Siftsy | Local Tailor |
|---------|-----------|--------------|--------|-------------|
| Monthly cost | $99+ | $249+ | $39+ | Free |
| Data stays on device | No | No | No | Yes |
| Owner-defined dimensions | No | No | No | Yes |
| Multi-dimensional analysis | No | No | No | Yes |
| Per-dimension span extraction | No | No | No | Yes |
| N/A vs Unclear distinction | No | No | No | Yes |
| Works fully offline | No | No | No | Yes |
| Fitted to niche vocabulary | No | No | No | Yes |

### 6.1 The "Can't You Just Use ChatGPT?" Answer

This question will come up. Prepare this answer:

| Concern | ChatGPT | Local Tailor |
|---------|---------|-------------|
| Privacy | Sends every customer comment to OpenAI's servers | All data stays local |
| Cost | API costs scale with comment volume — not free at any meaningful scale | Free |
| Adaptability | Cannot be fine-tuned on your product vocabulary. Generic outputs only | Fitted to your exact vocabulary with 8 examples |
| Structure | Returns prose summaries, not queryable structured data per dimension | Structured, queryable data per dimension |

Local Tailor is better on all four for this specific use case.

## 7. Evaluation Metrics

All metrics are measured against the synthetic 150-comment dataset with ground truth labels. Results are reproducible and presenter-controlled.

| Metric | How Measured | Target / Purpose |
|--------|-------------|-----------------|
| Classification accuracy | % correct vs ground truth per dimension | > 80% at 8 examples per class |
| Span precision | Correct span identified / total classified | > 75% exact or partial match |
| N/A detection rate | % of non-mentions correctly flagged as N/A | > 85% precision on N/A class |
| Unclear rate | % of ambiguous comments correctly held back | Demonstrate Unclear is meaningful, not a cop-out |
| Example-count sensitivity | Accuracy at 2 / 4 / 8 examples per class | Show monotonic improvement curve |
| Baseline delta | SetFit + span vs generic full-comment classification | Qualitative and quantitative gap |

### 7.1 Example-Count Sensitivity Chart

This chart replaces the before/after correction demo. It shows that the 8-examples-per-class design choice is evidence-based, not arbitrary.

Run SetFit on comfort dimension with 2, 4, 8, and 12 examples per class. Record accuracy against ground truth at each level. Plot the curve. The shape of the curve — fast improvement then plateau — is itself the argument for why 8 is the right number.

### 7.2 Dimension Coverage as a Business Signal

N/A rates per dimension are not a model failure metric — they are a product insight metric. Present them as such.

```
DIMENSION COVERAGE — 150 comments on: New Pillow Launch

Comfort          ████████████░░░░  71% of comments mention this
Shape            ██████████░░░░░░  61% of comments mention this
Color & Pattern  ████████░░░░░░░░  44% of comments mention this
Price / Value    ███████░░░░░░░░░  38% of comments mention this
Durability       █████░░░░░░░░░░░  29% of comments mention this
Packaging        ██░░░░░░░░░░░░░░  12% of comments mention this
```

Insight: customers care most about comfort and shape. No one is talking about packaging — stop spending on it.

## 8. Development Phases

| Phase | Name | Scope |
|-------|------|-------|
| Phase 1 | Core Pipeline (Current) | Synthetic dataset. Hardcoded pillow shop dimensions and examples. SetFit training + span extraction + predictions.json output. Accuracy metrics vs ground truth. |
| Phase 2 | User Interface | Streamlit UI. Owner defines dimensions and provides 8 examples per class through the interface. SetFit retrains on new examples. Dimension Board, Intent Queue, Analytics views. |
| Phase 3 | Facebook Integration | Facebook Graph API crawler. Owner provides post URL and access token. Live comments replace synthetic data. All Phase 1+2 functionality applies to real data. |

### 8.1 Phase 1 File Structure

```
local-tailor/
├── run_pipeline.py              ← entry point: load → embed → extract → classify
├── requirements.txt
├── config/
│   └── dimensions.yaml          ← pillow shop dimensions + 8 examples per class
├── data/
│   ├── synthetic_comments.json  ← 150-comment dataset with ground truth
│   ├── embeddings_{id}.npy      ← BERT vectors (N × 384)
│   └── predictions_{id}.json   ← {comment_id: {dimension: {value, score, flag, span}}}
└── localtailor/
    ├── config.py                ← DimensionConfig dataclass + YAML loader
    ├── loader.py                ← dataset → comments_clean.json
    ├── embedder.py              ← sentence-transformers encoding
    ├── span_extractor.py        ← roberta-base-squad2 extractive QA
    └── setfit_trainer.py        ← SetFit training + inference per dimension
```

## 9. What Makes This Stand Out

In a hackathon room likely to contain generic classifiers and chatbots, Local Tailor differs on four dimensions that judges can immediately understand:

### 9.1 Owner-Defined Dimensions

Every other team will build a classifier with predefined categories. Local Tailor builds a system where the owner decides what matters. This is not a technical nicety — it is the entire value proposition. A shoe shop and a pillow shop have nothing in common analytically. Generic tools cannot serve both. Local Tailor serves either, exactly.

### 9.2 Per-Dimension Span Extraction — Not Whole-Comment Scoring

The insight that "comfortable, size is not bad, price is too much tho" should produce three separate dimension classifications — not one generic label — is immediately intuitive to any business owner. It is also technically non-trivial. The span extraction + per-dimension classification pipeline demonstrates this distinction clearly and measurably.

### 9.3 Privacy-First Local Processing

In a presentation where Objective 3 explicitly evaluates responsible AI, the statement "your customers' comments never leave your machine" is a one-sentence ethical argument that most teams will not have. This is not marketing — it is a genuine architectural choice with real consequences for small businesses that handle customer data.

### 9.4 SetFit Few-Shot Design

Telling a judge "we need 8 examples per class and the model trains in 30 seconds" is a better technical story than "we fine-tuned on 10,000 labeled examples." It demonstrates understanding of practical constraints — a real business owner will not label thousands of comments. The few-shot approach is the right engineering tradeoff for the problem, and it is defensible from published research.

---

**LOCAL TAILOR**
*Free. Local. Fitted to your business.*
AIIC 2026 — Rangsit University International College
