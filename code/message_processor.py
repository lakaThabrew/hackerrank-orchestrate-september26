"""
Message Parser & Financial State Adjuster
Challenge: HackerRank Orchestrate (September 2026) - Buy or Wait?
Author: Antigravity

Parses messages to extract ground-truth adjustments:
- Confirmed non-payroll income (approved client invoices with settlement dates)
- Salary revisions, date shifts, temporary cuts, arrears, and partial/full household employment endings
- Rent increases with proper settlement date timing
- Strict untrusted data filtering (ignores unconfirmed bonuses, commissions, and prompt injections)
"""

import re
from datetime import datetime
import pandas as pd


def clean_amount(amt_str):
    if not amt_str:
        return None
    cleaned = str(amt_str).replace(',', '').rstrip('.')
    try:
        return float(cleaned)
    except ValueError:
        return None


class MessageProcessor:
    def __init__(self, data_loader):
        self.loader = data_loader

    def get_adjustments_for_user(self, user_id, request_date, request_id=None):
        """
        Parses all trusted messages for the given user sent on or before request_date.
        """
        prof = self.loader.get_user_profile(user_id)
        home_curr = prof['home_currency'] if prof else 'USD'

        messages_df = self.loader.get_user_messages(user_id, request_id=request_id)
        
        # Filter messages sent on or before request_date
        valid_msgs = []
        for _, m in messages_df.iterrows():
            sent_at = str(m['sent_at'])
            msg_date = sent_at.split('T')[0] if 'T' in sent_at else sent_at.split(' ')[0]
            if msg_date <= request_date:
                valid_msgs.append(m)

        adjustments = {
            'salary_override': None,        # dict: {'amount': float, 'effective_date': str or None, 'temporary': bool}
            'salary_date_shift': None,      # str: 'YYYY-MM-DD'
            'contract_ended': False,        # bool: if True, no future salary
            'rent_multiplier': 1.0,         # float: e.g. 1.12 for 12% increase
            'arrears_adjustment': 0.0,      # float: one-time additional credit on next salary
            'confirmed_incomes': [],        # list of dict: [{'amount': float, 'settlement_date': str, 'description': str}]
            'extra_recurring_expense': None # dict: {'category': str, 'amount': float, 'start_date': str}
        }

        # Sort messages by sent_at ascending so newer messages take precedence
        valid_msgs = sorted(valid_msgs, key=lambda x: str(x['sent_at']))

        for m in valid_msgs:
            text = m['message_text']

            # 1. Confirmed Approved Client Invoices (Non-payroll income)
            m_inv = re.search(
                r'(?:client approved an invoice payment of|klien menyetujui pembayaran faktur sebesar)\s*([A-Z]{3})?\s*([0-9,.]+).*?'
                r'(?:settlement is expected on|penyelesaian diperkirakan pada)\s*(\d{4}-\d{2}-\d{2})',
                text, re.I
            )
            if m_inv:
                inv_curr = m_inv.group(1)
                inv_amt = clean_amount(m_inv.group(2))
                inv_settle_date = m_inv.group(3)
                if inv_amt is not None and inv_settle_date:
                    conv_amt = self.loader.convert_to_home_currency(inv_amt, inv_curr or home_curr, home_curr, inv_settle_date)
                    adjustments['confirmed_incomes'].append({
                        'amount': conv_amt,
                        'settlement_date': inv_settle_date,
                        'description': 'Approved client invoice'
                    })

            # 2. Contract / Employment Ended
            # Check if one household record ended with remaining salary specified
            m_part_end = re.search(
                r'(?:one household employment record has ended|employment.*ended).*?'
                r'(?:remaining confirmed monthly salary is|gaji.*tersisa adalah)\s*([A-Z]{3})?\s*([0-9,.]+)',
                text, re.I
            )
            if m_part_end:
                rem_curr = m_part_end.group(1)
                rem_amt = clean_amount(m_part_end.group(2))
                if rem_amt is not None:
                    conv_amt = self.loader.convert_to_home_currency(rem_amt, rem_curr or home_curr, home_curr, request_date)
                    adjustments['salary_override'] = {'amount': conv_amt, 'effective_date': None, 'temporary': False}
                    adjustments['contract_ended'] = False
            elif re.search(r'contract has ended|employment has ended|kontrak.*berakhir|pendapatan.*telah berakhir', text, re.I):
                # Total contract ending
                adjustments['contract_ended'] = True

            # 3. Rent increase
            rent_pct_match = re.search(r'rent by (\d+)%|sewa.*sebesar (\d+)%', text, re.I)
            if rent_pct_match:
                pct = float(rent_pct_match.group(1) or rent_pct_match.group(2))
                adjustments['rent_multiplier'] = 1.0 + (pct / 100.0)

            # 4. Salary date shift
            date_match = re.search(r'expected on (\d{4}-\d{2}-\d{2})|diharapkan pada (\d{4}-\d{2}-\d{2})', text, re.I)
            if date_match and ('replaces' in text.lower() or 'menggantikan' in text.lower() or 'payroll' in text.lower()):
                new_date = date_match.group(1) or date_match.group(2)
                adjustments['salary_date_shift'] = new_date

            # 5. Salary updates
            # 5a. Temporary reduction / unpaid leave
            m_temp = re.search(r'(?:temporary monthly pay is|salary is reduced to|gaji.*berkurang menjadi)\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            if m_temp:
                amt = clean_amount(m_temp.group(2))
                curr = m_temp.group(1)
                if amt is not None:
                    conv_amt = self.loader.convert_to_home_currency(amt, curr or home_curr, home_curr, request_date)
                    adjustments['salary_override'] = {'amount': conv_amt, 'effective_date': None, 'temporary': True}

            # 5b. Regular salary change / promotion
            m_sal = re.search(
                r'(?:naik menjadi|gaji pokok.*adalah|first salary (?:will be|of)|regular salary (?:is now|of)|monthly salary (?:has increased to|is now|is)|confirmed (?:base )?salary is|gaji bulanan.*?adalah|gaji bulanan Anda naik menjadi)\s*([A-Z]{3})?\s*([0-9,.]+)',
                text, re.I
            )
            m_sal_of = re.search(r'salary of\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            
            chosen_sal_match = m_sal or m_sal_of
            if chosen_sal_match and not m_temp and not m_part_end:
                amt = clean_amount(chosen_sal_match.group(2))
                curr = chosen_sal_match.group(1)
                if amt is not None:
                    eff_date = None
                    eff_match = re.search(r'(?:effective|from|mulai|starting)?\s*(\d{4}-\d{2}-\d{2})', text, re.I)
                    if eff_match:
                        eff_date = eff_match.group(1)
                    conv_date = eff_date or request_date
                    conv_amt = self.loader.convert_to_home_currency(amt, curr or home_curr, home_curr, conv_date)
                    adjustments['salary_override'] = {'amount': conv_amt, 'effective_date': eff_date, 'temporary': False}

            # 5c. Arrears adjustment
            arrears_match = re.search(r'arrears adjustment of\s*([A-Z]{3})?\s*([0-9,.]+)|penyesuaian tunggakan sebesar\s*([A-Z]{3})?\s*([0-9,.]+)', text, re.I)
            if arrears_match:
                amt = clean_amount(arrears_match.group(2) or arrears_match.group(4))
                curr = arrears_match.group(1) or arrears_match.group(3)
                if amt is not None:
                    conv_amt = self.loader.convert_to_home_currency(amt, curr or home_curr, home_curr, request_date)
                    adjustments['arrears_adjustment'] = conv_amt

            # 6. Extra recurring deductions (e.g. childcare)
            m_child = re.search(r'recurring\s+(\w+)\s+payment begins in the same month', text, re.I)
            if m_child:
                raw_cat = m_child.group(1).lower()
                cat = 'family_support' if raw_cat == 'childcare' else raw_cat
                adjustments['extra_recurring_expense'] = {
                    'category': cat,
                    'amount': None,
                    'start_date': request_date
                }

        return adjustments

    def apply_adjustments_to_events(self, user_id, request_date, events_df, request_id=None):
        """
        Applies parsed message adjustments to a copy of user's events dataframe.
        """
        adj = self.get_adjustments_for_user(user_id, request_date, request_id=request_id)
        df = events_df.copy()

        # Date column for cash settlement timing (fallback to event_date)
        date_col = df['settlement_date'].fillna(df['event_date'])

        # 1. Apply Rent Multiplier to payments settling on or after request_date
        if adj['rent_multiplier'] != 1.0:
            rent_mask = (df['category'] == 'rent') & (date_col >= request_date)
            df.loc[rent_mask, 'amount'] = df.loc[rent_mask, 'amount'] * adj['rent_multiplier']

        # 2. Contract Ended -> remove/zero future salary
        if adj['contract_ended']:
            future_sal = (df['category'] == 'salary') & (date_col >= request_date)
            df.loc[future_sal, 'amount'] = 0.0

        # 3. Salary Date Shift
        if adj['salary_date_shift']:
            future_sal = (df['category'] == 'salary') & (date_col >= request_date)
            if future_sal.sum() > 0:
                idx = df[future_sal].index[0]
                df.loc[idx, 'settlement_date'] = adj['salary_date_shift']
                df.loc[idx, 'event_date'] = adj['salary_date_shift']

        # 4. Salary Override
        if adj['salary_override']:
            amt = adj['salary_override']['amount']
            eff = adj['salary_override'].get('effective_date')
            temp = adj['salary_override'].get('temporary', False)
            future_sal = (df['category'] == 'salary') & (date_col >= request_date)
            if future_sal.sum() > 0:
                if temp:
                    # Apply to first next salary
                    idx = df[future_sal].index[0]
                    df.loc[idx, 'amount'] = amt
                else:
                    if eff:
                        eff_mask = future_sal & (date_col >= eff)
                        df.loc[eff_mask, 'amount'] = amt
                    else:
                        df.loc[future_sal, 'amount'] = amt

        # 5. Arrears
        if adj['arrears_adjustment'] > 0:
            future_sal = (df['category'] == 'salary') & (date_col >= request_date)
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

    print("MessageProcessor initialized and ready.")
    for uid, rdate in [('user_34', '2024-12-04'), ('user_154', '2024-11-20'), ('user_16', '2023-08-12')]:
        adj = processor.get_adjustments_for_user(uid, rdate)
        print(f"[{uid}] on {rdate} adjustments: {adj}")
