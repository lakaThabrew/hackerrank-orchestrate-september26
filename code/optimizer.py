"""
Payment Plan Optimizer & Decision Generator
Challenge: HackerRank Orchestrate (September 2026) - Buy or Wait?
Author: Antigravity

Evaluates financial requests against user preferences, 90-day cashflow feasibility,
seller payment options, partial payments, and flexible spending changes.
Applies the mandatory 6-level tie-breaking hierarchy to select the optimal plan.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

def format_amount(val, currency):
    """
    Formats a numeric amount according to dataset conventions:
    EUR and USD use 2 decimal places.
    INR, IDR, and ZAR use integer strings when round, or 2 decimal places otherwise.
    """
    if currency in ['EUR', 'USD']:
        return f"{val:.2f}"
    if abs(val - round(val)) < 1e-4:
        return str(int(round(val)))
    return f"{val:.2f}"

def format_currency_text(val, currency):
    """
    Formats numbers for natural language explanations (with commas).
    """
    if currency in ['EUR', 'USD']:
        return f"{currency} {val:,.2f}"
    if abs(val - round(val)) < 1e-4:
        return f"{currency} {int(round(val)):,}"
    return f"{currency} {val:,.2f}"

def format_date_text(date_str):
    """
    Formats 'YYYY-MM-DD' into 'D Month YYYY' (e.g. '8 August 2025').
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return f"{dt.day} {dt.strftime('%B %Y')}"


class PaymentPlanOptimizer:
    def __init__(self, data_loader, message_processor, simulator):
        self.loader = data_loader
        self.processor = message_processor
        self.sim = simulator
        if hasattr(self.loader, 'payment_options_df') and self.loader.payment_options_df is not None:
            self.options_df = self.loader.payment_options_df
        else:
            options_path = os.path.join(getattr(self.loader, 'data_dir', 'dataset'), 'request_payment_options.csv')
            self.options_df = pd.read_csv(options_path)

    def get_payment_options_for_request(self, request_id):
        return self.options_df[self.options_df['request_id'] == request_id].copy()

    def find_flexible_spending_changes(self, user_id, request_date):
        """
        Identifies eligible flexible spending changes for a user.
        Excludes protected categories. Only includes categories user is willing to stop or reduce.
        Returns list of candidate change actions:
        - ('stop', event_id, description, category, monthly_saving)
        - ('reduce_to', event_id, description, category, minimum_allowed_amount, monthly_saving)
        """
        prof = self.loader.get_user_profile(user_id)
        protected = set(prof['expense_categories_to_protect'].split('|')) if pd.notna(prof['expense_categories_to_protect']) else set()
        
        willing_stop = set()
        if pd.notna(prof['expense_categories_user_is_willing_to_stop']) and prof['expense_categories_user_is_willing_to_stop']:
            willing_stop = set(prof['expense_categories_user_is_willing_to_stop'].split('|'))
            
        willing_reduce = set()
        if pd.notna(prof['expense_categories_user_is_willing_to_reduce']) and prof['expense_categories_user_is_willing_to_reduce']:
            willing_reduce = set(prof['expense_categories_user_is_willing_to_reduce'].split('|'))

        events = self.loader.get_user_events(user_id)
        # Derive effective cash date
        def get_cash_date(row):
            s_date = row['settlement_date']
            if pd.notna(s_date) and str(s_date).strip() != '':
                return str(s_date).strip()
            return str(row['event_date']).strip()

        events['cash_date'] = events.apply(get_cash_date, axis=1)
        # Settle before or on request date
        hist = events[
            (events['cash_date'] <= request_date) &
            (events['status'] == 'settled') &
            (events['flexibility'].isin(['stoppable', 'reducible', 'reducible_or_stoppable']))
        ].copy()

        candidates = []
        # Group by description/category to take the most recent occurrence
        for (desc, cat), grp in hist.groupby(['description', 'category']):
            if cat in protected:
                continue
            last_ev = grp.sort_values('cash_date').iloc[-1]
            eid = last_ev['event_id']
            flex = last_ev['flexibility']
            amt = float(last_ev['amount'])
            min_amt = float(last_ev['minimum_allowed_amount']) if pd.notna(last_ev['minimum_allowed_amount']) else None

            if (flex in ['stoppable', 'reducible_or_stoppable']) and (cat in willing_stop):
                candidates.append({
                    'type': 'stop',
                    'event_id': eid,
                    'description': desc,
                    'category': cat,
                    'action_str': f"stop:{eid}",
                    'saving': amt
                })

            if (flex in ['reducible', 'reducible_or_stoppable']) and (cat in willing_reduce) and min_amt is not None and amt > min_amt:
                candidates.append({
                    'type': 'reduce_to',
                    'event_id': eid,
                    'description': desc,
                    'category': cat,
                    'action_str': f"reduce_to:{eid}:{format_amount(min_amt, prof['home_currency'])}",
                    'new_amount': min_amt,
                    'saving': amt - min_amt
                })

        return candidates

    def optimize_request(self, request_row):
        """
        Executes end-to-end plan evaluation for a single request row.
        """
        rid = request_row['request_id']
        uid = request_row['user_id']
        rdate = request_row['request_date']
        ramt = float(request_row['requested_amount'])
        cdate = request_row['desired_completion_date']
        allows_partial = str(request_row['allows_partial_payment']).lower() in ['true', '1', 't']
        
        prof = self.loader.get_user_profile(uid)
        curr = prof['home_currency']
        min_bal = float(prof['minimum_balance_to_keep'])
        user_methods = set(prof['payment_methods_user_will_consider'].split('|'))
        max_inst = prof['max_installment_months']
        max_inst = float(max_inst) if pd.notna(max_inst) and str(max_inst).strip() != '' else None

        # 1. Base financial capacity
        safe_amt = self.sim.get_amount_safe_to_pay(uid, rdate, ramt, request_id=rid)
        earliest_full_date = self.sim.get_earliest_date_for_full_payment(uid, rdate, ramt, request_id=rid)

        if safe_amt >= ramt:
            safe_amt = ramt
            earliest_full_date = rdate

        options = self.get_payment_options_for_request(rid)
        candidates = []

        # --- A. Immediate Full Payment (no changes) ---
        if 'full_payment' in user_methods and safe_amt >= ramt:
            opt_id = 'none'
            full_opts = options[options['payment_method'] == 'full_payment']
            if len(full_opts) > 0:
                opt_id = full_opts.iloc[0]['payment_option_id']
            candidates.append({
                'method': 'full_payment',
                'plan_str': f"{rdate}:{format_amount(ramt, curr)}",
                'total_payable': ramt,
                'first_payment_date': rdate,
                'completion_date': rdate,
                'number_of_payments': 1,
                'option_id': opt_id,
                'spending_changes': 'none',
                'requires_changes': False,
                'changes_desc': ''
            })

        # --- B. Installments (no changes) ---
        if 'installments' in user_methods and max_inst is not None:
            inst_opts = options[options['payment_method'] == 'installments']
            for _, opt in inst_opts.iterrows():
                num_pay = int(opt['number_of_payments'])
                if num_pay <= max_inst:
                    freq = int(opt['payment_frequency_days']) if pd.notna(opt['payment_frequency_days']) else 30
                    f_dt = datetime.strptime(opt['first_payment_date'], '%Y-%m-%d')
                    pdates = [(f_dt + timedelta(days=k * freq)).strftime('%Y-%m-%d') for k in range(num_pay)]
                    
                    # Validate date bounds: first payment >= rdate, completion <= cdate, within 90-day horizon
                    horizon_end_str = (datetime.strptime(rdate, '%Y-%m-%d') + timedelta(days=90)).strftime('%Y-%m-%d')
                    if pdates[0] < rdate or pdates[-1] > cdate or pdates[-1] > horizon_end_str:
                        continue

                    pamt = float(opt['payment_amount'])
                    tot_payable = float(opt['total_payable_amount'])

                    # Check safety of this installment plan through plan completion and user deadline
                    plan_dict = {d: pamt for d in pdates}
                    traj, min_b, _ = self.sim.simulate_trajectory(
                        uid, rdate, request_id=rid, payment_plan=plan_dict
                    )
                    eval_end_date = max(pdates[-1], cdate)
                    plan_traj = [b for d, b in traj if d.strftime('%Y-%m-%d') <= eval_end_date]
                    if min(plan_traj) >= min_bal:
                        plan_str = '|'.join([f"{d}:{format_amount(pamt, curr)}" for d in pdates])
                        candidates.append({
                            'method': 'installments',
                            'plan_str': plan_str,
                            'total_payable': tot_payable,
                            'first_payment_date': pdates[0],
                            'completion_date': pdates[-1],
                            'number_of_payments': num_pay,
                            'option_id': opt['payment_option_id'],
                            'spending_changes': 'none',
                            'requires_changes': False,
                            'changes_desc': '',
                            'installment_amount': pamt,
                            'min_avail_balance': min_b
                        })

        # --- C. Partial Payment (no changes) ---
        if 'partial_payment' in user_methods and allows_partial:
            if 0 < safe_amt < ramt and earliest_full_date != "" and earliest_full_date <= cdate:
                if curr in ['IDR', 'INR', 'ZAR']:
                    pay1 = float(round(safe_amt))
                    pay2 = float(round(ramt) - round(pay1))
                else:
                    pay1 = float(round(safe_amt, 2))
                    pay2 = float(round(ramt - pay1, 2))

                if pay1 > 0 and pay2 > 0:
                    plan_dict = {rdate: pay1, earliest_full_date: pay2}
                    traj, min_b, _ = self.sim.simulate_trajectory(
                        uid, rdate, request_id=rid, payment_plan=plan_dict
                    )
                    if min_b >= min_bal:
                        plan_str = f"{rdate}:{format_amount(pay1, curr)}|{earliest_full_date}:{format_amount(pay2, curr)}"
                        candidates.append({
                            'method': 'partial_payment',
                            'plan_str': plan_str,
                            'total_payable': ramt,
                            'first_payment_date': rdate,
                            'completion_date': earliest_full_date,
                            'number_of_payments': 2,
                            'option_id': 'partial',
                            'spending_changes': 'none',
                            'requires_changes': False,
                            'changes_desc': '',
                            'pay1': pay1,
                            'pay2': pay2
                        })

        # --- D. Wait (no changes) ---
        if 'full_payment' in user_methods and earliest_full_date != "" and rdate < earliest_full_date <= cdate:
            candidates.append({
                'method': 'wait',
                'plan_str': f"{earliest_full_date}:{format_amount(ramt, curr)}",
                'total_payable': ramt,
                'first_payment_date': earliest_full_date,
                'completion_date': earliest_full_date,
                'number_of_payments': 1,
                'option_id': 'wait',
                'spending_changes': 'none',
                'requires_changes': False,
                'changes_desc': ''
            })

        # --- E. Evaluate Spending Changes (if needed) ---
        viable_no_change = [c for c in candidates if c['completion_date'] <= cdate]
        if len(viable_no_change) == 0:
            flex_changes = self.find_flexible_spending_changes(uid, rdate)
            # Try single changes and combinations (up to 3 changes)
            change_combinations = []
            for c1 in flex_changes:
                change_combinations.append([c1])
            for i, c1 in enumerate(flex_changes):
                for j, c2 in enumerate(flex_changes[i+1:], i+1):
                    if c1['event_id'] != c2['event_id']:
                        change_combinations.append([c1, c2])
                        for k, c3 in enumerate(flex_changes[j+1:], j+1):
                            if c1['event_id'] != c3['event_id'] and c2['event_id'] != c3['event_id']:
                                change_combinations.append([c1, c2, c3])

            for comb in change_combinations:
                change_str = '|'.join([c['action_str'] for c in comb])
                # Test full payment today with these spending changes
                plan_dict = {rdate: ramt}
                _, min_b, is_safe = self.sim.simulate_trajectory(
                    uid, rdate, request_id=rid, payment_plan=plan_dict, spending_changes=change_str
                )
                if is_safe and 'full_payment' in user_methods:
                    # Construct description for explanation
                    desc_parts = []
                    for c in comb:
                        desc_clean = c['description'].lower()
                        if c['type'] == 'stop':
                            desc_parts.append(f"Stop the {desc_clean}")
                        elif c['type'] == 'reduce_to':
                            desc_parts.append(f"reduce the {desc_clean} to {format_currency_text(c['new_amount'], curr)}")
                    
                    if len(desc_parts) == 1:
                        changes_desc = desc_parts[0]
                    elif len(desc_parts) == 2:
                        p2 = desc_parts[1][0].lower() + desc_parts[1][1:]
                        changes_desc = f"{desc_parts[0]} and {p2}"
                    else:
                        p2 = desc_parts[1][0].lower() + desc_parts[1][1:]
                        p3 = desc_parts[2][0].lower() + desc_parts[2][1:]
                        changes_desc = f"{desc_parts[0]}, {p2}, and {p3}"

                    candidates.append({
                        'method': 'full_payment',
                        'plan_str': f"{rdate}:{format_amount(ramt, curr)}",
                        'total_payable': ramt,
                        'first_payment_date': rdate,
                        'completion_date': rdate,
                        'number_of_payments': 1,
                        'option_id': 'full_changes',
                        'spending_changes': change_str,
                        'requires_changes': True,
                        'changes_desc': changes_desc,
                        'min_avail_balance': min_b
                    })

        # --- Step 4: Rank Candidates according to 6-level hierarchy ---
        if len(candidates) == 0:
            return self._build_not_affordable_result(
                rid, ramt, safe_amt, curr, cdate, min_bal, earliest_full_date=earliest_full_date
            )

        def ranking_key(cand):
            completes_on_time = (cand['completion_date'] <= cdate)
            no_changes = (not cand['requires_changes'])
            tot = cand['total_payable']
            start = cand['first_payment_date']
            n_pay = cand['number_of_payments']
            opt_id = cand['option_id']
            return (
                0 if completes_on_time else 1,   # 1. Complete full request by desired_completion_date
                0 if no_changes else 1,           # 2. Require no spending changes
                tot,                             # 3. Minimize total amount paid
                start,                           # 4. Start payment earlier
                n_pay,                           # 5. Use fewer payments
                opt_id                           # 6. Lowest payment_option_id
            )

        candidates.sort(key=ranking_key)
        best = candidates[0]

        # If the best plan cannot complete by deadline and requires changes or is not feasible,
        # verify if it should be not_affordable
        if best['completion_date'] > cdate and best['method'] != 'wait':
            return self._build_not_affordable_result(
                rid, ramt, safe_amt, curr, cdate, min_bal, earliest_full_date=earliest_full_date
            )

        # Build output fields
        method = best['method']
        plan_str = best['plan_str']
        changes_str = best['spending_changes']

        # Determine affordability status
        if method == 'full_payment' and not best['requires_changes']:
            status = 'affordable_now'
        elif method in ['partial_payment', 'installments'] or best['requires_changes']:
            status = 'affordable_with_plan'
        elif method == 'wait':
            status = 'affordable_later'
        else:
            status = 'not_affordable'

        # Earliest date for full payment
        # For affordable_now, must equal request_date
        # Otherwise equals earliest_full_date without spending changes
        earliest_out = rdate if status == 'affordable_now' else earliest_full_date

        # Generate Grounded Decision Explanation
        explanation = self._generate_explanation(
            best, rid, uid, rdate, ramt, curr, min_bal, safe_amt, cdate, earliest_full_date
        )

        safe_amt_out = safe_amt
        if method == 'partial_payment' and 'pay1' in best:
            safe_amt_out = best['pay1']

        return {
            'request_id': rid,
            'amount_safe_to_pay': safe_amt_out,
            'affordability_status': status,
            'recommended_payment_method': method,
            'payment_plan': plan_str,
            'earliest_date_for_full_payment': earliest_out,
            'spending_changes_needed': changes_str,
            'decision_explanation': explanation
        }

    def _build_not_affordable_result(self, rid, ramt, safe_amt, curr, cdate, min_bal, earliest_full_date=""):
        cdate_text = format_date_text(cdate)

        if safe_amt >= ramt:
            expl = (
                f"Do not make this payment by {cdate_text}. "
                f"Although the full amount is financially safe today, "
                f"no eligible payment method is accepted by the user."
            )
        elif safe_amt > 0:
            expl = (
                f"Do not proceed with the {format_currency_text(ramt, curr)} request. "
                f"Although {format_currency_text(safe_amt, curr)} is available today, "
                f"the full amount cannot be completed safely within 90 days."
            )
        else:
            expl = (
                f"Do not make this payment by {cdate_text}. "
                f"None of the available options keeps the {format_currency_text(min_bal, curr)} minimum protected."
            )

        return {
            'request_id': rid,
            'amount_safe_to_pay': safe_amt,
            'affordability_status': 'not_affordable',
            'recommended_payment_method': 'not_recommended',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': earliest_full_date,
            'spending_changes_needed': 'none',
            'decision_explanation': expl
        }

    def _generate_explanation(self, best, rid, uid, rdate, ramt, curr, min_bal, safe_amt, cdate, earliest_full_date):
        method = best['method']
        
        if method == 'full_payment':
            if best['requires_changes']:
                return (
                    f"{best['changes_desc']}, then pay {format_currency_text(ramt, curr)} today. "
                    f"This leaves at least {format_currency_text(min_bal, curr)} available."
                )
            else:
                return (
                    f"Pay {format_currency_text(ramt, curr)} today. "
                    f"This leaves at least {format_currency_text(min_bal, curr)} available over the next 90 days."
                )

        elif method == 'installments':
            n_pay = best['number_of_payments']
            pamt = best.get('installment_amount', ramt / n_pay)
            f_date_text = format_date_text(best['first_payment_date'])
            return (
                f"Use {n_pay} installments of {format_currency_text(pamt, curr)}, starting {f_date_text}. "
                f"This leaves at least {format_currency_text(min_bal, curr)} available."
            )

        elif method == 'partial_payment':
            pay1 = best['pay1']
            pay2 = best['pay2']
            e_date_text = format_date_text(best['completion_date'])
            return (
                f"Pay {format_currency_text(pay1, curr)} today and the remaining {format_currency_text(pay2, curr)} on {e_date_text}. "
                f"This completes the full request and keeps the {format_currency_text(min_bal, curr)} minimum protected."
            )

        elif method == 'wait':
            e_date_text = format_date_text(best['completion_date'])
            return (
                f"Pay {format_currency_text(ramt, curr)} in full on {e_date_text}. "
                f"Paying earlier would take the balance below the {format_currency_text(min_bal, curr)} minimum."
            )

        return ""


if __name__ == '__main__':
    import sys
    sys.path.append('code')
    from data_loader import DataLoader
    from message_processor import MessageProcessor
    from simulator import FinancialSimulator

    loader = DataLoader()
    processor = MessageProcessor(loader)
    sim = FinancialSimulator(loader, processor)
    optimizer = PaymentPlanOptimizer(loader, processor, sim)

    print("PaymentPlanOptimizer initialized.")
    samples = pd.read_csv('dataset/sample_requests.csv')
    row0 = samples.iloc[0]
    res = optimizer.optimize_request(row0)
    print("Sample 1 result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
