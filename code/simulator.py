"""
90-Day Cash Flow Simulation Engine
Challenge: HackerRank Orchestrate (September 2026) - Buy or Wait?
Author: Antigravity

Accurately projects a user's daily cashflow over a 90-day horizon:
- Full currency isolation (normalizes foreign cash flows to home_currency)
- Settlement-date cash timing (uses settlement_date for cash movement)
- Reconciles duplicate card charges and unrealized/pending non-cash items
- Multi-stream salary projection with support for temporary reductions and arrears
- Confirmed non-payroll income (approved client invoices)
- Computes amount_safe_to_pay and earliest_date_for_full_payment subject to minimum_balance_to_keep
"""

from message_processor import MessageProcessor
from data_loader import DataLoader
import calendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class FinancialSimulator:
    def __init__(self, data_loader, message_processor):
        self.loader = data_loader
        self.processor = message_processor

    def simulate_trajectory(self, user_id, request_date, request_id=None,
                            payment_plan=None, spending_changes=None, forecast_days=90):
        prof = self.loader.get_user_profile(user_id)
        home_curr = prof['home_currency']
        min_bal = float(prof['minimum_balance_to_keep'])
        avail = float(prof['current_available_balance'])

        events = self.loader.get_user_events(user_id)
        adj_events, adj = self.processor.apply_adjustments_to_events(
            user_id, request_date, events, request_id=request_id
        )

        stopped_events = set()
        reduced_events = {}
        if spending_changes and spending_changes != 'none':
            items = spending_changes.split('|') if isinstance(spending_changes, str) else spending_changes
            for item in items:
                item = item.strip()
                if item.startswith('stop:'):
                    stopped_events.add(item.split(':')[1])
                elif item.startswith('reduce_to:'):
                    parts = item.split(':')
                    reduced_events[parts[1]] = float(parts[2])

        req_dt = datetime.strptime(request_date, '%Y-%m-%d')
        end_dt = req_dt + timedelta(days=forecast_days)
        end_date_str = end_dt.strftime('%Y-%m-%d')

        def get_cash_date(row):
            s_date = row['settlement_date']
            if pd.notna(s_date) and str(s_date).strip() != '':
                return str(s_date).strip()
            return str(row['event_date']).strip()

        adj_events['cash_date'] = adj_events.apply(get_cash_date, axis=1)

        # 1. Starting Balance: Reserve pending debits on or before request_date
        pending_debits = adj_events[
            (adj_events['status'] == 'pending') &
            (adj_events['direction'] == 'debit') &
            (adj_events['cash_date'] <= request_date)
        ]
        
        reserved_pending_sum = 0.0
        for _, pevt in pending_debits.iterrows():
            desc = str(pevt['description']).lower()
            if 'duplicate' in desc:
                continue
            if pd.notna(pevt['linked_event_id']) and str(pevt['linked_event_id']).strip() != '':
                linked_ev = adj_events[adj_events['event_id'] == pevt['linked_event_id']]
                if len(linked_ev) > 0 and linked_ev.iloc[0]['status'] == 'settled':
                    continue
            
            c_amt = self.loader.convert_to_home_currency(
                pevt['amount'], pevt['currency'], home_curr, pevt['cash_date']
            )
            reserved_pending_sum += c_amt

        starting_bal = avail - reserved_pending_sum

        daily_inflows = {req_dt + timedelta(days=i): 0.0 for i in range(forecast_days + 1)}
        daily_outflows = {req_dt + timedelta(days=i): 0.0 for i in range(forecast_days + 1)}

        # 2. Known scheduled / pending future events
        known_future = adj_events[
            (adj_events['cash_date'] >= request_date) &
            (adj_events['cash_date'] <= end_date_str)
        ]

        for _, fevt in known_future.iterrows():
            eid = fevt['event_id']
            if eid in stopped_events:
                continue
            
            desc = str(fevt['description']).lower()
            if 'duplicate' in desc and fevt['status'] == 'pending':
                continue

            raw_amt = reduced_events.get(eid, float(fevt['amount']))
            c_amt = self.loader.convert_to_home_currency(
                raw_amt, fevt['currency'], home_curr, fevt['cash_date']
            )
            f_dt = datetime.strptime(fevt['cash_date'], '%Y-%m-%d')
            
            if fevt['direction'] == 'debit' and fevt['status'] in ['scheduled', 'pending', 'settled']:
                daily_outflows[f_dt] += c_amt
            elif fevt['direction'] == 'credit' and fevt['status'] in ['scheduled', 'settled']:
                if fevt['category'] == 'salary':
                    if not adj['contract_ended']:
                        daily_inflows[f_dt] += c_amt
                elif fevt['category'] in ['refund', 'reimbursement', 'investment_sale', 'transfer', 'income']:
                    daily_inflows[f_dt] += c_amt

        # 3. Confirmed client invoices from messages (non-payroll income)
        for inv in adj.get('confirmed_incomes', []):
            inv_dt = datetime.strptime(inv['settlement_date'], '%Y-%m-%d')
            if req_dt <= inv_dt <= end_dt:
                daily_inflows[inv_dt] += float(inv['amount'])

        # 4. Confirmed Salary Recurrence
        if not adj['contract_ended']:
            past_sal = adj_events[
                (adj_events['category'] == 'salary') &
                (adj_events['status'] == 'settled')
            ].copy()
            
            past_sal = past_sal[
                ~past_sal['description'].str.contains(r'arrears|bonus|komisi|penyesuaian|one-off', case=False, na=False)
            ]

            salary_streams = []
            if len(past_sal) > 0:
                # Iterate from most recent to oldest to find distinct recurring streams (e.g. 15th vs 20th)
                past_sal_sorted = past_sal.sort_values('cash_date', ascending=False)
                for _, row in past_sal_sorted.iterrows():
                    s_curr = row['currency']
                    s_amt = self.loader.convert_to_home_currency(
                        row['amount'], s_curr, home_curr, row['cash_date']
                    )
                    s_day = datetime.strptime(row['cash_date'], '%Y-%m-%d').day
                    # If this day is within 3 days of an already added stream, it's the same stream over time
                    if any(abs(stream['day'] - s_day) <= 3 for stream in salary_streams):
                        continue
                    salary_streams.append({
                        'description': row['description'],
                        'amount': s_amt,
                        'day': s_day,
                        'last_date': row['cash_date']
                    })
            else:
                fut_sal = adj_events[
                    (adj_events['category'] == 'salary') &
                    (adj_events['status'] == 'scheduled')
                ]
                if len(fut_sal) > 0:
                    for s_desc, grp in fut_sal.groupby('description'):
                        row = grp.iloc[0]
                        s_curr = row['currency']
                        s_amt = self.loader.convert_to_home_currency(
                            row['amount'], s_curr, home_curr, row['cash_date']
                        )
                        s_day = datetime.strptime(row['cash_date'], '%Y-%m-%d').day
                        salary_streams.append({
                            'description': s_desc,
                            'amount': s_amt,
                            'day': s_day,
                            'last_date': row['cash_date']
                        })

            if adj['salary_date_shift'] and len(salary_streams) > 0:
                shift_dt = datetime.strptime(adj['salary_date_shift'], '%Y-%m-%d')
                salary_streams[0]['day'] = shift_dt.day

            arrears_remaining = adj.get('arrears_adjustment', 0.0)
            for stream in salary_streams:
                s_amt = stream['amount']
                s_day = stream['day']
                is_first_cycle = True

                past_this_month = past_sal[
                    (past_sal['description'] == stream['description']) &
                    (past_sal['cash_date'].str.startswith(request_date[:7])) &
                    (past_sal['cash_date'] <= request_date)
                ]
                start_m = 1 if len(past_this_month) > 0 else 0

                for m_off in range(start_m, start_m + 4):
                    cur_m = req_dt.month + m_off
                    cur_y = req_dt.year + (cur_m - 1) // 12
                    cur_m = ((cur_m - 1) % 12) + 1
                    max_d = calendar.monthrange(cur_y, cur_m)[1]
                    p_dt = datetime(cur_y, cur_m, min(s_day, max_d))

                    if req_dt < p_dt <= end_dt:
                        already = any(
                            fevt['category'] == 'salary' and abs((datetime.strptime(fevt['cash_date'], '%Y-%m-%d') - p_dt).days) <= 3
                            for _, fevt in known_future.iterrows()
                        )
                        if not already:
                            cycle_amt = s_amt
                            if adj.get('salary_override'):
                                if is_first_cycle or not adj['salary_override'].get('temporary', False):
                                    cycle_amt = adj['salary_override']['amount']

                            if arrears_remaining > 0 and is_first_cycle:
                                cycle_amt += arrears_remaining
                                arrears_remaining = 0.0

                            daily_inflows[p_dt] += cycle_amt
                            is_first_cycle = False

        # 5. Monthly Recurring Commitments
        monthly_cats = ['rent', 'utilities', 'debt_repayment', 'cloud_storage',
                        'streaming', 'music_subscription', 'delivery_membership',
                        'education', 'family_support', 'housing', 'insurance', 'gym', 'entertainment']

        hist_debits = adj_events[
            (adj_events['cash_date'] <= request_date) &
            (adj_events['status'] == 'settled') &
            (adj_events['direction'] == 'debit')
        ].copy()

        for cat in monthly_cats:
            cat_events = hist_debits[hist_debits['category'] == cat].sort_values('cash_date')
            if len(cat_events) == 0:
                continue

            last_ev = cat_events.iloc[-1]
            eid = last_ev['event_id']
            if eid in stopped_events:
                continue

            base_amt = float(last_ev['amount'])
            if cat == 'rent' and adj.get('rent_multiplier', 1.0) != 1.0:
                base_amt *= adj['rent_multiplier']

            amt = reduced_events.get(eid, base_amt)
            c_amt = self.loader.convert_to_home_currency(
                amt, last_ev['currency'], home_curr, last_ev['cash_date']
            )
            ev_dt = datetime.strptime(last_ev['cash_date'], '%Y-%m-%d')
            day_of_month = ev_dt.day

            already_this_month = any(
                datetime.strptime(d, '%Y-%m-%d').month == req_dt.month and
                datetime.strptime(d, '%Y-%m-%d').year == req_dt.year and
                datetime.strptime(d, '%Y-%m-%d') <= req_dt
                for d in cat_events['cash_date']
            )
            start_m_off = 1 if already_this_month else 0

            for m_off in range(start_m_off, start_m_off + 4):
                cur_m = req_dt.month + m_off
                cur_y = req_dt.year + (cur_m - 1) // 12
                cur_m = ((cur_m - 1) % 12) + 1
                max_d = calendar.monthrange(cur_y, cur_m)[1]
                p_dt = datetime(cur_y, cur_m, min(day_of_month, max_d))
                if req_dt < p_dt <= end_dt:
                    already = any(
                        fevt['category'] == cat and abs((datetime.strptime(fevt['cash_date'], '%Y-%m-%d') - p_dt).days) <= 3
                        for _, fevt in known_future.iterrows()
                    )
                    if not already:
                        daily_outflows[p_dt] += c_amt

        # 5b. Extra recurring expense from messages (e.g. childcare)
        if adj.get('extra_recurring_expense'):
            extra_exp = adj['extra_recurring_expense']
            extra_cat = extra_exp['category']
            extra_amt = extra_exp.get('amount')
            if extra_amt is None:
                cat_past = hist_debits[hist_debits['category'] == extra_cat]
                if len(cat_past) > 0:
                    extra_amt = float(cat_past.iloc[-1]['amount'])
                else:
                    currency_defaults = {'EUR': 200.0, 'USD': 175.0, 'INR': 12500.0, 'ZAR': 4000.0, 'IDR': 3000000.0}
                    extra_amt = currency_defaults.get(home_curr, 200.0)
            
            for m_off in range(0, 4):
                cur_m = req_dt.month + m_off
                cur_y = req_dt.year + (cur_m - 1) // 12
                cur_m = ((cur_m - 1) % 12) + 1
                max_d = calendar.monthrange(cur_y, cur_m)[1]
                p_dt = datetime(cur_y, cur_m, min(15, max_d))
                if req_dt < p_dt <= end_dt:
                    daily_outflows[p_dt] += float(extra_amt)

        # 6. Short-cycle essentials: groceries, transport (exclude discretionary one-offs like shopping/dining)
        short_cats = ['groceries', 'transport']
        for cat in short_cats:
            cat_events = hist_debits[hist_debits['category'] == cat].sort_values('cash_date')
            if len(cat_events) < 2:
                continue
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in cat_events['cash_date']]
            diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            median_interval = max(3, int(round(np.median(diffs))))

            last_date = dates[-1]
            last_ev = cat_events.iloc[-1]
            eid = last_ev['event_id']
            if eid in stopped_events:
                continue

            default_amt = float(cat_events['amount'].tail(3).mean())
            amt = reduced_events.get(eid, default_amt)
            c_amt = self.loader.convert_to_home_currency(
                amt, last_ev['currency'], home_curr, last_ev['cash_date']
            )

            p_dt = last_date + timedelta(days=median_interval)
            while p_dt <= end_dt:
                if p_dt > req_dt:
                    daily_outflows[p_dt] += c_amt
                p_dt += timedelta(days=median_interval)

        # 7. Payment Plan deduction
        if payment_plan:
            for pdate_str, pamt in payment_plan.items():
                pdt = datetime.strptime(pdate_str, '%Y-%m-%d')
                if pdt in daily_outflows:
                    daily_outflows[pdt] += float(pamt)

        # 8. Compute Trajectory
        trajectory = []
        b = starting_bal
        for i in range(forecast_days + 1):
            day = req_dt + timedelta(days=i)
            b += daily_inflows[day] - daily_outflows[day]
            trajectory.append((day, b))

        min_balance = min(bal for _, bal in trajectory)
        is_safe = (min_balance >= min_bal)

        return trajectory, min_balance, is_safe

    def get_amount_safe_to_pay(self, user_id, request_date, requested_amount, request_id=None):
        prof = self.loader.get_user_profile(user_id)
        min_bal = float(prof['minimum_balance_to_keep'])
        _, min_b, _ = self.simulate_trajectory(user_id, request_date, request_id=request_id)
        safe = max(0.0, min_b - min_bal)
        return min(safe, float(requested_amount))

    def get_earliest_date_for_full_payment(self, user_id, request_date, requested_amount, request_id=None):
        prof = self.loader.get_user_profile(user_id)
        min_bal = float(prof['minimum_balance_to_keep'])
        req_amt = float(requested_amount)
        
        trajectory, _, _ = self.simulate_trajectory(user_id, request_date, request_id=request_id)
        
        for idx, (day, _) in enumerate(trajectory):
            # Evaluate across pay cycle (next 30 days) to avoid day 90 boundary artifacts
            horizon_end = min(len(trajectory), idx + 31)
            subsequent_bals = [b for _, b in trajectory[idx:horizon_end]]
            if len(subsequent_bals) > 0 and min(subsequent_bals) - req_amt >= min_bal:
                return day.strftime('%Y-%m-%d')

        return ""


if __name__ == '__main__':
    loader = DataLoader()
    processor = MessageProcessor(loader)
    sim = FinancialSimulator(loader, processor)
    print("Simulator ready.")
