"""
HackerRank Orchestrate (September 2026) — Buy or Wait?
Main Entry Point for Autonomous Financial Decision Agent.

Processes dataset/requests.csv, reconstructs financial cash flows,
evaluates payment options and spending adjustments, and generates
the final predictions in output.csv conforming strictly to the evaluation schema.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Ensure module imports work regardless of cwd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_loader import DataLoader
from message_processor import MessageProcessor
from simulator import FinancialSimulator
from optimizer import PaymentPlanOptimizer

REQUIRED_COLUMNS = [
    'request_id',
    'amount_safe_to_pay',
    'affordability_status',
    'recommended_payment_method',
    'payment_plan',
    'earliest_date_for_full_payment',
    'spending_changes_needed',
    'decision_explanation'
]

def run_pipeline(data_dir=None, output_path=None, verbose=True):
    repo_root = os.path.dirname(CURRENT_DIR)
    if data_dir is None:
        data_dir = os.path.join(repo_root, 'dataset')
    
    if output_path is None:
        output_path = os.path.join(repo_root, 'output.csv')

    print(f"============================================================")
    print(f"Buy or Wait? Autonomous Financial Decision Pipeline")
    print(f"Repository Root: {repo_root}")
    print(f"Dataset Path:    {data_dir}")
    print(f"Output Path:     {output_path}")
    print(f"============================================================")

    # 1. Initialize data loaders and core engines
    print("Loading datasets and media assets...")
    loader = DataLoader(data_dir=data_dir)
    loader.load_all()

    processor = MessageProcessor(loader)
    simulator = FinancialSimulator(loader, processor)
    optimizer = PaymentPlanOptimizer(loader, processor, simulator)

    requests_df = loader.requests_df
    total_requests = len(requests_df)
    print(f"Successfully loaded {total_requests} evaluation requests.")

    predictions = []

    # 2. Iterate through each evaluation request
    for idx, row in requests_df.iterrows():
        rid = row['request_id']
        if verbose and (idx % 25 == 0 or idx == total_requests - 1):
            print(f"Processing request {idx + 1}/{total_requests} [{rid}]...")

        pred = optimizer.optimize_request(row)

        # Enforce exact formatting according to project contract
        output_row = {
            'request_id': rid,
            'amount_safe_to_pay': pred['amount_safe_to_pay'],
            'affordability_status': pred['affordability_status'],
            'recommended_payment_method': pred['recommended_payment_method'],
            'payment_plan': pred['payment_plan'],
            'earliest_date_for_full_payment': pred['earliest_date_for_full_payment'],
            'spending_changes_needed': pred['spending_changes_needed'],
            'decision_explanation': pred['decision_explanation']
        }
        predictions.append(output_row)

    # 3. Build DataFrame and ensure column order
    out_df = pd.DataFrame(predictions)[REQUIRED_COLUMNS]

    # 4. Save to target output path and dataset/output.csv
    out_df.to_csv(output_path, index=False)
    dataset_output = os.path.join(data_dir, 'output.csv')
    out_df.to_csv(dataset_output, index=False)

    print(f"\nPipeline execution completed successfully!")
    print(f"Saved {len(out_df)} predictions to: {output_path}")
    print(f"Synced template replica to:     {dataset_output}")
    print(f"Output schema validated: {list(out_df.columns)}")
    print(f"============================================================\n")

    return out_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Buy or Wait? Decision Agent Pipeline")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to dataset directory")
    parser.add_argument('--output', type=str, default=None, help="Path to output CSV file")
    parser.add_argument('--quiet', action='store_true', help="Disable progress logging")
    args = parser.parse_args()

    run_pipeline(data_dir=args.data_dir, output_path=args.output, verbose=not args.quiet)
