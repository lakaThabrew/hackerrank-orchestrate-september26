# Buy or Wait? — Task Tracker & Implementation Checklist

This document tracks the step-by-step implementation of the AI-powered financial decision agent for HackerRank Orchestrate (September 2026).

---

## Task Overview

- [x] **Task 1: Multimodal Image Extraction & Missing Amounts Filler**
  - [x] Inspect all 16 images in `dataset/media/images/image_01.png` to `image_16.png`.
  - [x] Extract the exact monetary amount, currency, and date from each image.
  - [x] Link each image to its respective event via `dataset/images.csv`.
  - [x] Create `code/data_loader.py` with an integrated image-amount cache.
  - [x] Verify that 100% of the 25,342 events in `financial_events.csv` have valid numeric amounts.

- [x] **Task 2: Message Parser & Financial State Adjuster**
  - [x] Parse all 215 messages in `dataset/messages.csv`.
  - [x] Extract updates to salary (amount + effective dates), rent increases, and payout statuses.
  - [x] Detect and ignore unconfirmed bonuses, pending commissions, and untrusted instructions / prompt-injections.
  - [x] Build `code/message_processor.py` to reconcile events using challenge conflict-resolution rules.

- [x] **Task 3: 90-Day Cash Flow Simulation Engine**
  - [x] Detect recurring spending patterns (weekly, bi-weekly, monthly) and differentiate fixed vs. variable essentials.
  - [x] Build conservative 90-day daily cashflow projection in `code/simulator.py`.
  - [x] Reserve pending debits from current balance on `request_date`.
  - [x] Add confirmed salary on scheduled settlement dates; ignore pending credits and unrealized investments.
  - [x] Implement `amount_safe_to_pay` binary search / analytical solver subject to `minimum_balance_to_keep`.
  - [x] Implement `earliest_date_for_full_payment` search across the 90-day horizon.

- [x] **Task 4: Payment Plan Optimizer & Option Selector**
  - [x] Evaluate `full_payment` feasibility against user preferences.
  - [x] Evaluate seller installment options from `dataset/request_payment_options.csv` against user's `max_installment_months` and `desired_completion_date`.
  - [x] Evaluate 2-step `partial_payment` plans when allowed and beneficial.
  - [x] Evaluate permitted flexible spending changes (`stop:<id>` and `reduce_to:<id>:<amount>`).
  - [x] Apply the 6-level tie-breaking hierarchy:
    1. Complete by `desired_completion_date`.
    2. Require no spending changes.
    3. Minimize total payable amount.
    4. Start payment earlier.
    5. Fewer payments.
    6. Lowest `payment_option_id`.
  - [x] Implement grounded `decision_explanation` generator matching problem style.

- [x] **Task 5: Validation Against `sample_requests.csv`**
  - [x] Implement evaluation script `code/evaluation/main.py`.
  - [x] Test the pipeline on all 25 benchmark cases in `dataset/sample_requests.csv`.
  - [x] Verify exact match on `affordability_status`, `recommended_payment_method`, `payment_plan`, `earliest_date_for_full_payment`, and `spending_changes_needed`.
  - [x] Validate numerical accuracy of `amount_safe_to_pay`.

- [x] **Task 6: Full-Dataset Run & Submission Packaging**
  - [x] Run end-to-end pipeline across all 250 requests in `dataset/requests.csv`.
  - [x] Generate root-level `output.csv` with strict schema validation.
  - [x] Generate `code/evaluation/usage_report.md` tracking model calls, tokens, and cost breakdown.
  - [x] Create submission package `code.zip` (excluding datasets, node_modules, and virtual environments).
  - [x] Validate final deliverables and verify compliance with submission guidelines.
