import os
import pandas as pd
import numpy as np

# Exact extracted amounts for the 16 multimodal evidence images
IMAGE_EXTRACTED_AMOUNTS = {
    'event_253': 4365000.0,   # image_01: IDR 4,365,000 net pay on payslip
    'event_1442': 100000.0,   # image_02: INR 100,000 outstanding rent balance on receipt
    'event_1545': 41272.0,    # image_03: INR 41,272 net cash paid on store receipt
    'event_1700': 2854.0,     # image_04: INR 2,854 total item bill (free delivery)
    'event_1786': 704.05,     # image_05: INR 704.05 Airtel business bill due
    'event_3051': 1995.0,     # image_06: INR 1,995 total Blinkit grocery order
    'event_3231': 8528.0,     # image_07: INR 8,528 restaurant tax invoice grand total
    'event_4535': 15339.0,    # image_08: INR 15,339 housing society maintenance receipt
    'event_5170': 723.0,      # image_09: INR 723 water bill payment receipt
    'event_6033': 79679.26,   # image_10: INR 79,679.26 grocery invoice balance due
    'event_6859': 3650.0,     # image_11: INR 3,650 hospital provisional bill amount payable
    'event_7307': 33.5,       # image_12: USD 33.50 taxi cab ride receipt
    'event_7941': 2298.0,     # image_13: INR 2,298 DailyObjects online shopping total
    'event_9421': 4543.0,     # image_14: INR 4,543 pharmacy medicine bill total
    'event_9806': 9968.0,     # image_15: INR 9,968 flight booking invoice grand total
    'event_10521': 393.22,    # image_16: INR 393.22 EV charging tax invoice total
}


def _read_clean_csv(filepath, **kwargs):
    """
    Reads a CSV while stripping whitespace from column names and string values.
    """
    df = pd.read_csv(filepath, skipinitialspace=True, **kwargs)
    df = df.rename(columns=lambda c: str(c).strip())
    return df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)


class DataLoader:
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        self.profiles_df = None
        self.events_df = None
        self.requests_df = None
        self.sample_requests_df = None
        self.payment_options_df = None
        self.exchange_rates_df = None
        self.messages_df = None
        self.images_df = None
        self.load_all()

    def load_all(self):
        self.profiles_df = _read_clean_csv(os.path.join(self.data_dir, 'financial_profiles.csv'))
        self.events_df = _read_clean_csv(os.path.join(self.data_dir, 'financial_events.csv'))
        self.requests_df = _read_clean_csv(os.path.join(self.data_dir, 'requests.csv'))
        self.payment_options_df = _read_clean_csv(os.path.join(self.data_dir, 'request_payment_options.csv'))
        self.exchange_rates_df = _read_clean_csv(os.path.join(self.data_dir, 'exchange_rates.csv'))
        self.messages_df = _read_clean_csv(os.path.join(self.data_dir, 'messages.csv'))
        self.images_df = _read_clean_csv(os.path.join(self.data_dir, 'images.csv'))
        
        sample_path = os.path.join(self.data_dir, 'sample_requests.csv')
        if os.path.exists(sample_path):
            self.sample_requests_df = _read_clean_csv(sample_path)

        # Impute missing amounts from image extraction
        self._impute_image_amounts()

    def _impute_image_amounts(self):
        missing_mask = self.events_df['amount'].isna()
        missing_count = missing_mask.sum()
        if missing_count > 0:
            for event_id, amount in IMAGE_EXTRACTED_AMOUNTS.items():
                self.events_df.loc[self.events_df['event_id'] == event_id, 'amount'] = amount

        remaining_missing = self.events_df['amount'].isna().sum()
        assert remaining_missing == 0, f"Error: {remaining_missing} events still have missing amounts!"

    def get_exchange_rate(self, rate_date, from_curr, to_curr):
        if from_curr == to_curr:
            return 1.0
        
        # Ensure rate_date is YYYY-MM-DD
        rate_date_str = str(rate_date).split('T')[0].split(' ')[0]
        match = self.exchange_rates_df[
            (self.exchange_rates_df['rate_date'] == rate_date_str) &
            (self.exchange_rates_df['from_currency'] == from_curr) &
            (self.exchange_rates_df['to_currency'] == to_curr)
        ]
        if len(match) > 0:
            return float(match.iloc[0]['rate'])
        
        # Check inverse
        inv_match = self.exchange_rates_df[
            (self.exchange_rates_df['rate_date'] == rate_date_str) &
            (self.exchange_rates_df['from_currency'] == to_curr) &
            (self.exchange_rates_df['to_currency'] == from_curr)
        ]
        if len(inv_match) > 0:
            return 1.0 / float(inv_match.iloc[0]['rate'])
        
        # If exact date not found, find nearest available date for this currency pair
        pair_matches = self.exchange_rates_df[
            (self.exchange_rates_df['from_currency'] == from_curr) &
            (self.exchange_rates_df['to_currency'] == to_curr)
        ]
        if len(pair_matches) > 0:
            return float(pair_matches.iloc[-1]['rate'])

        pair_inv = self.exchange_rates_df[
            (self.exchange_rates_df['from_currency'] == to_curr) &
            (self.exchange_rates_df['to_currency'] == from_curr)
        ]
        if len(pair_inv) > 0:
            return 1.0 / float(pair_inv.iloc[-1]['rate'])

        raise ValueError(f"No exchange rate found for {from_curr}->{to_curr} on {rate_date_str}")

    def convert_to_home_currency(self, amount, currency, home_currency, rate_date):
        if currency == home_currency or pd.isna(currency):
            return float(amount)
        rate = self.get_exchange_rate(rate_date, currency, home_currency)
        return float(amount) * rate

    def get_user_profile(self, user_id):
        row = self.profiles_df[self.profiles_df['user_id'] == user_id]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()

    def get_user_events(self, user_id):
        return self.events_df[self.events_df['user_id'] == user_id].copy()

    def get_request(self, request_id):
        req_id = str(request_id).strip()
        row = self.requests_df[self.requests_df['request_id'] == req_id]
        if len(row) == 0 and self.sample_requests_df is not None:
            row = self.sample_requests_df[self.sample_requests_df['request_id'] == req_id]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()

    def get_payment_options(self, request_id):
        req_id = str(request_id).strip()
        return self.payment_options_df[self.payment_options_df['request_id'] == req_id].copy()

    def get_user_messages(self, user_id, request_id=None):
        cond = (self.messages_df['user_id'] == user_id)
        if request_id:
            req_id = str(request_id).strip()
            # Only messages for this user: either general (no request_id) or matching this request_id
            cond = cond & (
                self.messages_df['request_id'].isna() |
                (self.messages_df['request_id'] == '') |
                (self.messages_df['request_id'] == req_id)
            )
        return self.messages_df[cond].copy()


if __name__ == '__main__':
    loader = DataLoader()
    print("DataLoader successfully loaded all datasets.")
    print(f"Total events: {len(loader.events_df)}, missing amounts: {loader.events_df['amount'].isna().sum()}")
    print(f"Requests count: {len(loader.requests_df)}")
    print(f"Sample request_26: {loader.get_request('request_26')}")
