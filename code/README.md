# Buy or Wait? — AI-Powered Financial Decision Agent
**HackerRank Orchestrate (September 2026)**

Autonomous financial agent that determines whether a user can safely afford a requested purchase or expense, evaluating full payments, seller installments, 2-step partial payment schedules, or recommending waiting.

---

## 1. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `numpy`

### Installation
```bash
pip install pandas numpy
```

### Run Full Predictions (250 Requests)
To execute the end-to-end pipeline across all evaluation requests in `dataset/requests.csv` and generate `output.csv`:
```bash
python code/main.py
```
This produces `output.csv` in the repository root and syncs `dataset/output.csv` conforming strictly to the evaluation schema.

### Run Official Evaluation Benchmark (25 Solved Samples)
To validate the model accuracy and decision consistency against `dataset/sample_requests.csv`:
```bash
python code/evaluation/main.py
```

---

## 2. Solution Architecture

The system utilizes a hybrid neuro-symbolic approach combining multimodal extraction, message NLP parsing, deterministic cash-flow simulation, and multi-objective Pareto optimization:

```
                  ┌───────────────────────────────┐
                  │ dataset/ (CSVs, Media Images) │
                  └───────────────┬───────────────┘
                                  │
                       [DataLoader & Vision OCR]
                                  │
                                  ▼
                     [Financial State Adjuster]
                 (Salary shifts, amendments, leases)
                                  │
                                  ▼
                  [90-Day Daily Simulator Engine]
               (Currency isolation, debt reservations,
               cycle-aware multi-stream salary tracking)
                                  │
                                  ▼
                    [Payment Plan Optimizer]
                 (Full, Installments, Partial, Wait)
                                  │
                       6-Tier Tie-Breaking Hierarchy
                                  │
                                  ▼
                   output.csv & usage_report.md
```

### Key Modules:
1. **`code/data_loader.py`:**
   - Normalizes input whitespace and field padding.
   - Strict `user_id` scoping to eliminate cross-user data leakage.
   - Built-in multimodal image extraction cache linking receipts/bills in `media/images/*.png`.
   - Bidirectional currency conversion helper using dated fixed exchange rates.

2. **`code/message_processor.py`:**
   - Parses employer payroll notices, approved client invoices, lease renewals, and contract endings.
   - Enforces challenge conflict-resolution hierarchy: explicit cancellations/amendments > newer source records > settled events > conservative safe interpretation.
   - Reconciles one-time arrears, temporary cuts, and recurring childcare deductions.

3. **`code/simulator.py`:**
   - **Cash Timing:** Uses `settlement_date` as the primary timing anchor for cash movements.
   - **Currency Isolation:** Converts all cashflows into the user's `home_currency` before arithmetic.
   - **Duplicate Reconciliation:** Prunes pending duplicate card charges before debit reservation.
   - **Multi-Stream Recurrence:** Clusters multi-income payrolls by cycle day, filtering one-off bonuses/arrears.
   - **Solvency Guarantee:** Ensures daily projected balance never dips below `minimum_balance_to_keep`.

4. **`code/optimizer.py`:**
   - Explores full payment today, eligible installments from `request_payment_options.csv`, 2-step partial schedules, and flexible spending reductions (`stop:<id>` and `reduce_to:<id>:<amt>`).
   - Implements the mandatory 6-level tie-breaking hierarchy.
   - Generates natural, grounded explanations matching benchmark styles.

5. **`code/evaluation/`:**
   - `main.py`: Quantitative benchmark test runner.
   - `usage_report.md`: Accounting of model calls, tokens, latency, and computational costs.

---

## 3. Project Structure
```text
code/
├── main.py                     # Main prediction pipeline entry point
├── data_loader.py              # Data ingestion and exchange rate converter
├── message_processor.py        # Untrusted message NLP & financial adjustments
├── simulator.py                # 90-day multi-currency daily cashflow engine
├── optimizer.py                # Payment plan optimizer & decision generator
├── README.md                   # Setup instructions and solution documentation
└── evaluation/
    ├── main.py                 # Benchmark evaluation script
    └── usage_report.md         # Final token usage and cost accounting report
```
