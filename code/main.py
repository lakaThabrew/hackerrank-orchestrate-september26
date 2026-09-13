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

def validate_predictions(df, requests_df, options_df=None):
    """
    Strict validation of output schema, row coverage, enum values, bounds,
    chronological plans, partial payment sums, and installment options.
    """
    if len(df) != len(requests_df):
        raise ValueError(f"Expected {len(requests_df)} rows in predictions, got {len(df)}")
    
    if list(df['request_id']) != list(requests_df['request_id']):
        raise ValueError("request_id sequence does not match requests.csv")

    allowed_affordability = {'affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable'}
    allowed_methods = {'full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended'}

    for _, row in df.iterrows():
        rid = row['request_id']
        status = row['affordability_status']
        method = row['recommended_payment_method']
        safe_amt = float(row['amount_safe_to_pay'])
        orig = requests_df[requests_df['request_id'] == rid].iloc[0]
        ramt = float(orig['requested_amount'])
        rdate = str(orig['request_date']).strip()

        if status not in allowed_affordability:
            raise ValueError(f"Row {rid}: invalid affordability_status '{status}'")
        if method not in allowed_methods:
            raise ValueError(f"Row {rid}: invalid recommended_payment_method '{method}'")
        if not (-1e-5 <= safe_amt <= ramt + 1e-5):
            raise ValueError(f"Row {rid}: amount_safe_to_pay {safe_amt} out of bounds [0, {ramt}]")

        earliest = str(row['earliest_date_for_full_payment']).strip()
        if earliest in ['nan', 'None']:
            earliest = ''
        if status == 'affordable_now' and earliest != rdate:
            raise ValueError(f"Row {rid}: for affordable_now, earliest_date_for_full_payment must equal request_date {rdate}, got '{earliest}'")

        changes = str(row['spending_changes_needed']).strip()
        if changes != 'none':
            c_items = changes.split('|')
            if len(c_items) > 3:
                raise ValueError(f"Row {rid}: too many spending changes ({len(c_items)} > 3)")
            for c_it in c_items:
                if not (c_it.startswith('stop:') or c_it.startswith('reduce_to:')):
                    raise ValueError(f"Row {rid}: invalid spending change format '{c_it}'")

        plan = str(row['payment_plan']).strip()
        if plan != 'none':
            items = plan.split('|')
            prev_d = None
            tot_plan = 0.0
            for it in items:
                parts = it.split(':')
                if len(parts) != 2:
                    raise ValueError(f"Row {rid}: invalid payment plan item '{it}'")
                d = datetime.strptime(parts[0], '%Y-%m-%d')
                if prev_d and d < prev_d:
                    raise ValueError(f"Row {rid}: payment plan dates not chronological: {plan}")
                prev_d = d
                tot_plan += float(parts[1].replace(',', ''))

            if method == 'partial_payment':
                if len(items) != 2:
                    raise ValueError(f"Row {rid}: partial_payment must have exactly 2 payments, got {len(items)}")
                p1_amt = float(items[0].split(':')[1].replace(',', ''))
                if abs(p1_amt - safe_amt) > 1.0:
                    raise ValueError(f"Row {rid}: first partial payment {p1_amt} must match safe_amt {safe_amt}")
                if abs(tot_plan - ramt) > 1.0:
                    raise ValueError(f"Row {rid}: partial payments sum {tot_plan} does not match requested amount {ramt}")

        if method == 'installments' and options_df is not None:
            opts = options_df[options_df['request_id'] == rid]
            if len(opts) == 0:
                raise ValueError(f"Row {rid}: recommended installments but no payment options exist")


def run_pipeline(data_dir=None, output_path=None, verbose=True):
    repo_root = os.path.dirname(CURRENT_DIR)
    if data_dir is None:
        data_dir = os.path.join(repo_root, 'dataset')

    if output_path is None:
        output_path = os.path.join(repo_root, 'output.csv')

    if verbose:
        print(f"============================================================")
        print(f"Buy or Wait? Autonomous Financial Decision Pipeline")
        print(f"Repository Root: {repo_root}")
        print(f"Dataset Path:    {data_dir}")
        print(f"Output Path:     {output_path}")
        print(f"============================================================")

    # 1. Initialize data and models
    if verbose:
        print("Loading datasets and media assets...")
    loader = DataLoader(data_dir=data_dir)
    processor = MessageProcessor(loader)
    simulator = FinancialSimulator(loader, processor)
    optimizer = PaymentPlanOptimizer(loader, processor, simulator)

    requests_df = loader.requests_df
    total_requests = len(requests_df)
    if verbose:
        print(f"Successfully loaded {total_requests} evaluation requests.")

    # 2. Process all requests
    predictions = []
    for idx, (_, row) in enumerate(requests_df.iterrows()):
        rid = row['request_id']
        if verbose and (idx % 25 == 0 or idx == total_requests - 1):
            print(f"Processing request {idx+1}/{total_requests} [{rid}]...")

        pred = optimizer.optimize_request(row)

        output_row = {
            'request_id': pred['request_id'],
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

    # Validate output schema and constraints
    validate_predictions(out_df, requests_df, options_df=loader.payment_options_df)

    # 4. Save to target output path
    out_df.to_csv(output_path, index=False)

    print(f"\nPipeline execution completed successfully!")
    print(f"Saved {len(out_df)} predictions to: {output_path}")
    print(f"Output schema and data integrity validated: {list(out_df.columns)}")
    print(f"============================================================\n")

    return out_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Buy or Wait? Decision Agent Pipeline")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to dataset directory")
    parser.add_argument('--output', type=str, default=None, help="Path to output CSV file")
    parser.add_argument('--quiet', action='store_true', help="Disable progress logging")
    args = parser.parse_args()

    run_pipeline(data_dir=args.data_dir, output_path=args.output, verbose=not args.quiet)
