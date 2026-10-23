# Claude API: Request for Testing Budget

**Project:** GNN PRUNER, a system that answers questions about very large documents (contracts,
books) using only quotes it has checked against the document, with page-level citations.

**Request:** **US $50** of prepaid Claude API (Anthropic) credit, to test the system with a
paid AI model. A hard spending limit of $50 is set in the Anthropic Console, so the amount
cannot be exceeded.

**Prices:** Anthropic's published API rates as of September 2026. Re-check them at
claude.com/pricing before submitting. Amounts are in US dollars and exclude taxes.

---

## 1. What we need

| Item | Cost |
|---|---|
| Anthropic Console account (platform.claude.com) | Free |
| API key for the project | Free |
| Prepaid API credit (pay per use, no subscription) | **$50** |
| Spending limit set to $50 | Free |

No subscription, servers or other software are needed.

---

## 2. Why we need it

- **The system needs an AI model** to read the relevant parts of a document and write the
  final, cited answer. We have built and tested it with free options. Testing now needs a
  stronger model.
- **Free options are not enough:**
  - Google Gemini's free tier allows about 20 requests per day for its main model.
  - Running AI models on the laptop (8 GB memory) froze it.
- **Why Claude:**
  - strong accuracy on long documents
  - it follows the "cite only the given evidence" rule closely
  - it can read up to ~1 million tokens at once, which is needed for one comparison test
  - it is already integrated in our code

---

## 3. What the $50 covers

| Test | Details | Estimate |
|---|---|---|
| Current test set | 80 questions × 2 versions of the system, answers by Claude Opus 5.5 | ~$2 |
| Preparing test documents | summaries for 3 long documents (~2,000 pages), at the 50% batch discount | ~$3 |
| Long-document test | 100 questions on 100- to 1,000-page documents | ~$11 |
| Comparison with "AI reads the whole document" | the same documents sent whole to Claude, reused via caching | ~$15 |
| Automatic grading of answers | ~300 gradings | ~$3 |
| Small checks during testing | | ~$5 |
| **Subtotal** | | **~$39** |
| Buffer | re-runs, small price changes | ~$11 |
| **Total** | | **$50** |

Approximate value in Indian rupees: about ₹4,400 at ₹87 per dollar, plus applicable taxes.
Use the exchange rate on the billing date.

**Smaller option: $25.** This drops the whole-document comparison and halves the
long-document test.

---

## 4. Pricing used (per 1 million tokens; 1 million tokens ≈ 1,500 contract pages)

| Model | Used for | Input | Output |
|---|---|---|---|
| Claude Opus 5.5 | final answers, whole-document comparison | $4.00 | $20.00 |
| Claude Sonnet 5.5 | searching chapters, grading answers | $2.00 | $10.00 |
| Claude Haiku 4.5 | document summaries | $1.00 | $5.00 |

Cost savers used:
- **Batch processing:** 50% off.
- **Caching:** a document already sent is re-read at $0.20 per million tokens instead of
  $4.00.

---

## 5. Cost controls

- **Spending limit:** a hard $50 limit in the Anthropic Console.
- **Cost estimate before every run:** each run is approved before it starts.
- **Every call logged with its cost.** Interrupted runs resume without paying twice.
- **Development stays on the free tier.** Paid calls are used only for the tests above.

---

## 6. Data

- **Test data only:** generated contracts and public datasets. No confidential documents
  are sent.
- **Training:** under Anthropic's commercial terms, API data is not used to train models
  by default.
- **The API key** stays on the developer's machine and is never stored in project files.
