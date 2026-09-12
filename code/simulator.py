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
        """
        Simulates the user's cash balance from request_date to request_date + forecast_days.
        
        payment_plan: dict of {date_str: amount} to deduct as expense.
        spending_changes: list of strings like ['stop:event_14', 'reduce_to:event_21:100'].
        
        Returns:
            trajectory: list of (datetime, balance)
            min_balance: float (lowest balance over the period)
            is_safe: bool (min_balance >= minimum_balance_to_keep)
        """
        prof = self.loader.get_user_profile(user_id)
        min_bal = float(prof['minimum_balance_to_keep'])
        avail = float(prof['current_available_balance'])

        events = self.loader.get_user_events(user_id)
        adj_events, adj = self.processor.apply_adjustments_to_events(
            user_id, request_date, events, request_id=request_id
        )

        # Parse spending changes
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

        # 1. Starting Balance
        # Pending debits on or before request_date must be reserved
        pending_debits = adj_events[
            (adj_events['status'] == 'pending') &
            (adj_events['direction'] == 'debit') &
            (adj_events['event_date'] <= request_date)
        ]
        starting_bal = avail - float(pending_debits['amount'].sum())

        daily_inflows = {req_dt + timedelta(days=i): 0.0 for i in range(forecast_days + 1)}
        daily_outflows = {req_dt + timedelta(days=i): 0.0 for i in range(forecast_days + 1)}

        # 2. Known future events in financial_events.csv
        known_future = adj_events[
            (adj_events['event_date'] > request_date) &
            (adj_events['event_date'] <= end_dt.strftime('%Y-%m-%d'))
        ]
        for _, fevt in known_future.iterrows():
            eid = fevt['event_id']
            if eid in stopped_events:
                continue
            amt = reduced_events.get(eid, float(fevt['amount']))
            f_dt = datetime.strptime(fevt['event_date'], '%Y-%m-%d')
            
            if fevt['direction'] == 'debit' and fevt['status'] in ['scheduled', 'pending', 'settled']:
                daily_outflows[f_dt] += amt
            elif fevt['direction'] == 'credit' and fevt['status'] in ['scheduled', 'settled'] and fevt['category'] == 'salary' and not adj['contract_ended']:
                daily_inflows[f_dt] += amt

        # 3. Model Salary Recurrence if employment has not ended
        if not adj['contract_ended']:
            sal_amt = None
            sal_day = 15
            if adj['salary_override']:
                sal_amt = float(adj['salary_override']['amount'])

            future_sal = adj_events[
                (adj_events['category'] == 'salary') &
                (adj_events['event_date'] >= request_date) &
                (adj_events['status'] == 'scheduled')
            ]
            past_sal = adj_events[
                (adj_events['category'] == 'salary') &
                (adj_events['status'] == 'settled')
            ].sort_values('event_date')

            if len(future_sal) > 0:
                fs_dt = datetime.strptime(
                    future_sal.iloc[0]['settlement_date'] if pd.notna(future_sal.iloc[0]['settlement_date']) else future_sal.iloc[0]['event_date'],
                    '%Y-%m-%d'
                )
                sal_day = fs_dt.day
                if sal_amt is None:
                    sal_amt = float(future_sal.iloc[0]['amount'])
            elif len(past_sal) > 0:
                last_sal = past_sal.iloc[-1]
                if sal_amt is None:
                    sal_amt = float(last_sal['amount'])
                s_dt = datetime.strptime(
                    last_sal['settlement_date'] if pd.notna(last_sal['settlement_date']) else last_sal['event_date'],
                    '%Y-%m-%d'
                )
                sal_day = s_dt.day

            if adj['salary_date_shift']:
                shift_dt = datetime.strptime(adj['salary_date_shift'], '%Y-%m-%d')
                sal_day = shift_dt.day

            if sal_amt is not None:
                past_sal_this_month = past_sal[past_sal['event_date'].str.startswith(request_date[:7])]
                start_m_off = 1 if len(past_sal_this_month) > 0 and sal_day <= req_dt.day else 0

                for m_off in range(start_m_off, start_m_off + 4):
                    cur_m = req_dt.month + m_off
                    cur_y = req_dt.year + (cur_m - 1) // 12
                    cur_m = ((cur_m - 1) % 12) + 1
                    max_d = calendar.monthrange(cur_y, cur_m)[1]
                    p_dt = datetime(cur_y, cur_m, min(sal_day, max_d))
                    if req_dt < p_dt <= end_dt:
                        already = any(
                            fevt['category'] == 'salary' and abs((datetime.strptime(fevt['event_date'], '%Y-%m-%d') - p_dt).days) <= 3
                            for _, fevt in known_future.iterrows()
                        )
                        if not already:
                            daily_inflows[p_dt] += sal_amt

        # 4. Monthly Recurring Expenses
        monthly_cats = ['rent', 'utilities', 'debt_repayment', 'cloud_storage',
                        'streaming', 'music_subscription', 'delivery_membership',
                        'education', 'family_support', 'housing', 'insurance']
        hist = adj_events[
            (adj_events['event_date'] <= request_date) &
            (adj_events['status'] == 'settled')
        ].copy()

        for cat in monthly_cats:
            cat_events = hist[hist['category'] == cat].sort_values('event_date')
            if len(cat_events) == 0:
                continue
            last_ev = cat_events.iloc[-1]
            eid = last_ev['event_id']
            if eid in stopped_events:
                continue
            amt = reduced_events.get(eid, float(last_ev['amount']))
            ev_dt = datetime.strptime(last_ev['event_date'], '%Y-%m-%d')
            day_of_month = ev_dt.day

            already_this_month = any(
                datetime.strptime(d, '%Y-%m-%d').month == req_dt.month and
                datetime.strptime(d, '%Y-%m-%d').year == req_dt.year and
                datetime.strptime(d, '%Y-%m-%d') <= req_dt
                for d in cat_events['event_date']
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
                        fevt['category'] == cat and abs((datetime.strptime(fevt['event_date'], '%Y-%m-%d') - p_dt).days) <= 3
                        for _, fevt in known_future.iterrows()
                    )
                    if not already:
                        daily_outflows[p_dt] += amt

        # 5. Short-cycle essentials: groceries, transport, dining, shopping
        short_cats = ['groceries', 'transport', 'dining', 'shopping']
        for cat in short_cats:
            cat_events = hist[hist['category'] == cat].sort_values('event_date')
            if len(cat_events) < 2:
                continue
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in cat_events['event_date']]
            diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            median_interval = max(3, int(round(np.median(diffs))))

            last_date = dates[-1]
            last_ev = cat_events.iloc[-1]
            eid = last_ev['event_id']
            if eid in stopped_events:
                continue
            default_amt = float(cat_events['amount'].tail(3).mean())
            amt = reduced_events.get(eid, default_amt)

            p_dt = last_date + timedelta(days=median_interval)
            while p_dt <= end_dt:
                if p_dt > req_dt:
                    daily_outflows[p_dt] += amt
                p_dt += timedelta(days=median_interval)

        # 6. Apply payment plan if provided
        if payment_plan:
            for pdate_str, pamt in payment_plan.items():
                pdt = datetime.strptime(pdate_str, '%Y-%m-%d')
                if pdt in daily_outflows:
                    daily_outflows[pdt] += float(pamt)

        # 7. Calculate Trajectory
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

    def get_earliest_date_for_full_payment(self, user_id, request_date, requested_amount, request_id=None, desired_completion_date=None):
        prof = self.loader.get_user_profile(user_id)
        min_bal = float(prof['minimum_balance_to_keep'])
        req_amt = float(requested_amount)
        
        trajectory, _, _ = self.simulate_trajectory(user_id, request_date, request_id=request_id)
        
        # Check if full payment is safe on day d
        # On day d, balance decreases by req_amt and stays down for all subsequent days
        for idx, (day, _) in enumerate(trajectory):
            # Check if all subsequent days have balance - req_amt >= min_bal
            subsequent_bals = [b for _, b in trajectory[idx:]]
            if min(subsequent_bals) - req_amt >= min_bal:
                return day.strftime('%Y-%m-%d')

        return ""


if __name__ == '__main__':
    import sys
    sys.path.append('code')
    from data_loader import DataLoader
    from message_processor import MessageProcessor

    loader = DataLoader()
    processor = MessageProcessor(loader)
    sim = FinancialSimulator(loader, processor)

    print("FinancialSimulator initialized and ready.")
    for uid, rdate, ramt, rid in [
        ('user_01', '2024-03-03', 25256, 'request_01'),
        ('user_02', '2025-08-05', 46018000, 'request_02'),
        ('user_07', '2024-09-05', 197400, 'request_07'),
        ('user_09', '2026-07-04', 166.61, 'request_09'),
        ('user_16', '2023-08-12', 122500, 'request_16'),
    ]:
        safe = sim.get_amount_safe_to_pay(uid, rdate, ramt, request_id=rid)
        earliest = sim.get_earliest_date_for_full_payment(uid, rdate, ramt, request_id=rid)
        print(f"[{rid}] SafeToPay: {safe} | EarliestDate: {earliest}")
