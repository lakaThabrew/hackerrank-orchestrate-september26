"""
Official Evaluation Workflow for Buy or Wait? Financial Decision Agent
Validates predictions against benchmark sample requests in dataset/sample_requests.csv.
Computes precision, exact match, and error metrics across all required schema fields.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure code directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_loader import DataLoader
from message_processor import MessageProcessor
from simulator import FinancialSimulator
from optimizer import PaymentPlanOptimizer

def run_evaluation(data_dir=None, verbose=True):
    if data_dir is None:
        repo_root = os.path.dirname(CODE_DIR)
        data_dir = os.path.join(repo_root, 'dataset')

    loader = DataLoader(data_dir=data_dir)
    loader.load_all()

    processor = MessageProcessor(loader)
    simulator = FinancialSimulator(loader, processor)
    optimizer = PaymentPlanOptimizer(loader, processor, simulator)

    sample_path = os.path.join(data_dir, 'sample_requests.csv')
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample requests file not found at: {sample_path}")

    samples_df = pd.read_csv(sample_path, skipinitialspace=True)
    samples_df.columns = [c.strip() for c in samples_df.columns]
    for col in samples_df.columns:
        if samples_df[col].dtype == 'object':
            samples_df[col] = samples_df[col].str.strip()

    total_cases = len(samples_df)
    metrics = {
        'status_match': 0,
        'method_match': 0,
        'plan_match': 0,
        'earliest_date_match': 0,
        'spending_changes_match': 0,
        'safe_amt_exact_match': 0,
        'safe_amt_within_5pct': 0,
        'safe_amt_abs_diff': []
    }

    results = []

    print(f"============================================================")
    print(f"Running Buy or Wait? Evaluation Benchmark ({total_cases} requests)")
    print(f"============================================================")

    for idx, row in samples_df.iterrows():
        rid = row['request_id']
        pred = optimizer.optimize_request(row)

        true_status = str(row['affordability_status']).strip()
        true_method = str(row['recommended_payment_method']).strip()
        true_plan = str(row['payment_plan']).strip()
        true_earliest = str(row['earliest_date_for_full_payment']).strip() if pd.notna(row['earliest_date_for_full_payment']) else ""
        true_changes = str(row['spending_changes_needed']).strip()
        true_safe = float(row['amount_safe_to_pay'])

        pred_status = pred['affordability_status']
        pred_method = pred['recommended_payment_method']
        pred_plan = pred['payment_plan']
        pred_earliest = pred['earliest_date_for_full_payment']
        pred_changes = pred['spending_changes_needed']
        pred_safe = float(pred['amount_safe_to_pay'])

        s_match = (pred_status == true_status)
        m_match = (pred_method == true_method)
        p_match = (pred_plan == true_plan)
        e_match = (pred_earliest == true_earliest)
        c_match = (pred_changes == true_changes)
        
        amt_diff = abs(pred_safe - true_safe)
        metrics['safe_amt_abs_diff'].append(amt_diff)
        amt_exact = (amt_diff < 0.01)
        amt_5pct = (amt_diff / max(1.0, true_safe) <= 0.05)

        if s_match: metrics['status_match'] += 1
        if m_match: metrics['method_match'] += 1
        if p_match: metrics['plan_match'] += 1
        if e_match: metrics['earliest_date_match'] += 1
        if c_match: metrics['spending_changes_match'] += 1
        if amt_exact: metrics['safe_amt_exact_match'] += 1
        if amt_5pct: metrics['safe_amt_within_5pct'] += 1

        results.append({
            'request_id': rid,
            'status_match': s_match,
            'method_match': m_match,
            'plan_match': p_match,
            'earliest_match': e_match,
            'changes_match': c_match,
            'pred_status': pred_status,
            'true_status': true_status,
            'pred_method': pred_method,
            'true_method': true_method,
            'pred_plan': pred_plan,
            'true_plan': true_plan,
            'pred_safe': pred_safe,
            'true_safe': true_safe,
            'pred_explanation': pred['decision_explanation'],
            'true_explanation': row['decision_explanation']
        })

        if verbose:
            status_icon = "[MATCH]" if s_match else "[DIFF]"
            method_icon = "[MATCH]" if m_match else "[DIFF]"
            plan_icon = "[MATCH]" if p_match else "[DIFF]"
            print(f"[{rid}] Status: {pred_status} ({true_status}) {status_icon} | Method: {pred_method} ({true_method}) {method_icon} | Plan: {plan_icon}")
            if not m_match or not s_match:
                print(f"   Pred Expl: {pred['decision_explanation']}")
                print(f"   True Expl: {row['decision_explanation']}")

    print(f"\n============================================================")
    print(f"BENCHMARK EVALUATION SUMMARY ({total_cases} requests)")
    print(f"============================================================")
    print(f"  Affordability Status Accuracy:    {metrics['status_match']}/{total_cases} ({metrics['status_match']/total_cases*100:.1f}%)")
    print(f"  Recommended Method Accuracy:      {metrics['method_match']}/{total_cases} ({metrics['method_match']/total_cases*100:.1f}%)")
    print(f"  Payment Plan Exact Match:         {metrics['plan_match']}/{total_cases} ({metrics['plan_match']/total_cases*100:.1f}%)")
    print(f"  Earliest Full Date Match:         {metrics['earliest_date_match']}/{total_cases} ({metrics['earliest_date_match']/total_cases*100:.1f}%)")
    print(f"  Spending Changes Match:           {metrics['spending_changes_match']}/{total_cases} ({metrics['spending_changes_match']/total_cases*100:.1f}%)")
    print(f"  Amount Safe To Pay (Exact):       {metrics['safe_amt_exact_match']}/{total_cases} ({metrics['safe_amt_exact_match']/total_cases*100:.1f}%)")
    print(f"  Amount Safe To Pay (Within 5%):   {metrics['safe_amt_within_5pct']}/{total_cases} ({metrics['safe_amt_within_5pct']/total_cases*100:.1f}%)")
    print(f"  Mean Absolute Error (Safe Amt):   {np.mean(metrics['safe_amt_abs_diff']):.2f}")
    print(f"============================================================\n")

    return metrics, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate financial decision agent")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to dataset directory")
    parser.add_argument('--quiet', action='store_true', help="Suppress per-request output")
    args = parser.parse_args()

    run_evaluation(data_dir=args.data_dir, verbose=not args.quiet)
