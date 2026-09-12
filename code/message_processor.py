import re
import pandas as pd
from datetime import datetime

def clean_amt(s):
    if not s:
        return None
    s = s.replace(',', '').rstrip('.').strip()
    try:
        return float(s)
    except Exception:
        return None


class MessageProcessor:
    def __init__(self, data_loader):
        self.loader = data_loader

    def get_adjustments_for_user(self, user_id, request_date, request_id=None):
        """
        Processes messages relevant to user_id (and request_id) sent on or before request_date.
        Returns a dict of adjustments to apply to financial events and future projections.
        """
        msgs = self.loader.get_user_messages(user_id, request_id=request_id)
        user_prof = self.loader.get_user_profile(user_id)
        home_curr = user_prof['home_currency'] if user_prof else 'USD'

        if len(msgs) == 0:
            return {
                'salary_override': None,
                'salary_date_shift': None,
                'contract_ended': False,
                'rent_multiplier': 1.0,
                'arrears_adjustment': 0.0,
                'extra_recurring_expense': None,
            }

        # Filter messages sent on or before request_date
        req_dt = datetime.strptime(request_date, '%Y-%m-%d')
        valid_msgs = []
        for _, m in msgs.iterrows():
            sent_date_str = str(m['sent_at'])[:10]
            try:
                sent_dt = datetime.strptime(sent_date_str, '%Y-%m-%d')
                if sent_dt <= req_dt:
                    valid_msgs.append(m)
            except Exception:
                valid_msgs.append(m)

        adjustments = {
            'salary_override': None,        # dict: {'amount': float, 'effective_date': str or None, 'temporary': bool}
            'salary_date_shift': None,      # str: 'YYYY-MM-DD'
            'contract_ended': False,        # bool: if True, no future salary after contract end
            'rent_multiplier': 1.0,         # float: e.g. 1.12 for 12% increase
            'arrears_adjustment': 0.0,      # float: one-time additional credit on next salary
            'extra_recurring_expense': None,# dict: {'category': str, 'amount': float, 'start_date': str}
        }

        # Sort messages by sent_at ascending so newer messages take precedence
        valid_msgs = sorted(valid_msgs, key=lambda x: str(x['sent_at']))

        for m in valid_msgs:
            text = m['message_text']
            
            # 1. Contract / Employment Ended
            if re.search(r'contract has ended|employment has ended|kontrak.*berakhir|pendapatan.*telah berakhir', text, re.I):
                adjustments['contract_ended'] = True

            # 2. Rent increase
            rent_pct_match = re.search(r'rent by (\d+)%|sewa.*sebesar (\d+)%', text, re.I)
            if rent_pct_match:
                pct = float(rent_pct_match.group(1) or rent_pct_match.group(2))
                adjustments['rent_multiplier'] = 1.0 + (pct / 100.0)

            # 3. Salary date shift
            date_match = re.search(r'expected on (\d{4}-\d{2}-\d{2})|diharapkan pada (\d{4}-\d{2}-\d{2})', text, re.I)
            if date_match and ('replaces' in text.lower() or 'menggantikan' in text.lower() or 'payroll' in text.lower()):
                new_date = date_match.group(1) or date_match.group(2)
                adjustments['salary_date_shift'] = new_date

            # 4. Salary updates
            # 4a. Temporary reduction / unpaid leave
            m_temp = re.search(r'(?:temporary monthly pay is|salary is reduced to|gaji.*berkurang menjadi)\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            if m_temp:
                amt = clean_amt(m_temp.group(2))
                curr = m_temp.group(1)
                if amt is not None:
                    # check foreign currency conversion if needed
                    if curr and curr != home_curr:
                        eff_date = request_date
                        try:
                            rate = self.loader.get_exchange_rate(eff_date, curr, home_curr)
                            amt = amt * rate
                        except Exception:
                            pass
                    adjustments['salary_override'] = {'amount': amt, 'effective_date': None, 'temporary': True}

            # 4b. Regular salary change / promotion
            m_sal = re.search(r'(?:naik menjadi|gaji pokok.*adalah|first salary (?:will be|of)|regular salary (?:is now|of)|gaji bulanan.*?adalah|gaji bulanan Anda naik menjadi)\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            m_sal_of = re.search(r'salary of\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            
            chosen_sal_match = m_sal or m_sal_of
            if chosen_sal_match and not m_temp:
                amt = clean_amt(chosen_sal_match.group(2))
                curr = chosen_sal_match.group(1)
                if amt is not None:
                    eff_date = None
                    eff_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
                    if eff_match:
                        eff_date = eff_match.group(1)
                    
                    if curr and curr != home_curr:
                        conv_date = eff_date or request_date
                        try:
                            rate = self.loader.get_exchange_rate(conv_date, curr, home_curr)
                            amt = amt * rate
                        except Exception:
                            pass
                    
                    adjustments['salary_override'] = {'amount': amt, 'effective_date': eff_date, 'temporary': False}

            # 4c. Arrears adjustment
            arrears_match = re.search(r'arrears adjustment of\s*([A-Z]{3})?\s*([0-9,.]+)|penyesuaian tunggakan sebesar\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            if arrears_match:
                amt = clean_amt(arrears_match.group(2) or arrears_match.group(4))
                curr = arrears_match.group(1) or arrears_match.group(3)
                if amt is not None:
                    if curr and curr != home_curr:
                        try:
                            rate = self.loader.get_exchange_rate(request_date, curr, home_curr)
                            amt = amt * rate
                        except Exception:
                            pass
                    adjustments['arrears_adjustment'] = amt

        return adjustments

    def apply_adjustments_to_events(self, user_id, request_date, events_df, request_id=None):
        """
        Applies parsed message adjustments to a copy of user's events dataframe.
        """
        adj = self.get_adjustments_for_user(user_id, request_date, request_id=request_id)
        df = events_df.copy()

        # 1. Apply Rent Multiplier
        if adj['rent_multiplier'] != 1.0:
            rent_mask = (df['category'] == 'rent') & (df['event_date'] >= request_date)
            df.loc[rent_mask, 'amount'] = df.loc[rent_mask, 'amount'] * adj['rent_multiplier']

        # 2. Contract Ended -> remove/zero future salary
        if adj['contract_ended']:
            future_sal = (df['category'] == 'salary') & (df['event_date'] >= request_date)
            df.loc[future_sal, 'amount'] = 0.0

        # 3. Salary Date Shift
        if adj['salary_date_shift']:
            future_sal = (df['category'] == 'salary') & (df['event_date'] >= request_date)
            if future_sal.sum() > 0:
                idx = df[future_sal].index[0]
                df.loc[idx, 'settlement_date'] = adj['salary_date_shift']
                df.loc[idx, 'event_date'] = adj['salary_date_shift']

        # 4. Salary Override
        if adj['salary_override']:
            amt = adj['salary_override']['amount']
            eff = adj['salary_override'].get('effective_date')
            temp = adj['salary_override'].get('temporary', False)
            future_sal = (df['category'] == 'salary') & (df['event_date'] >= request_date)
            if future_sal.sum() > 0:
                if temp:
                    # Apply to first next salary
                    idx = df[future_sal].index[0]
                    df.loc[idx, 'amount'] = amt
                else:
                    if eff:
                        eff_mask = future_sal & (df['event_date'] >= eff)
                        df.loc[eff_mask, 'amount'] = amt
                    else:
                        df.loc[future_sal, 'amount'] = amt

        # 5. Arrears
        if adj['arrears_adjustment'] > 0:
            future_sal = (df['category'] == 'salary') & (df['event_date'] >= request_date)
            if future_sal.sum() > 0:
                idx = df[future_sal].index[0]
                df.loc[idx, 'amount'] += adj['arrears_adjustment']

        return df, adj


if __name__ == '__main__':
    import sys
    sys.path.append('code')
    from data_loader import DataLoader
    loader = DataLoader()
    processor = MessageProcessor(loader)

    # Test sample users: user_02, user_06, user_07, user_08, user_12, user_16
    for uid, rdate in [('user_02', '2025-08-05'), ('user_06', '2026-01-03'), ('user_07', '2024-09-05'), ('user_08', '2025-02-07'), ('user_12', '2026-04-05'), ('user_16', '2023-08-12')]:
        events = loader.get_user_events(uid)
        adj_events, adj = processor.apply_adjustments_to_events(uid, rdate, events)
        print(f"=== {uid} (request_date={rdate}) ===")
        print(f"Adjustments: {adj}")
        sal = adj_events[(adj_events['category']=='salary') & (adj_events['event_date'] >= rdate)]
        if len(sal) > 0:
            print(f"Next salary after adjustment: {sal.iloc[0]['event_date']} amt={sal.iloc[0]['amount']}")
        print()
