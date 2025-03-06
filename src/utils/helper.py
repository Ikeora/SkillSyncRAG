import datetime
import re
from dateutil.relativedelta import relativedelta

def convert_todate(relative_time_str):
    # Get the current date and time
    now = datetime.date.today()
    relative_time_str = relative_time_str.strip()

    # Define regex patterns for different time units
    patterns = {
        'minutes': r'(\d+)\s*minutes?\s*ago',
        'hours': r'(\d+)\s*hours?\s*ago',
        'days': r'(\d+)\s*days?\s*ago',
        'weeks': r'(\d+)\s*weeks?\s*ago',
        'months': r'(\d+)\s*months?\s*ago',
        'years': r'(\d+)\s*years?\s*ago'
    }

    # Check each pattern and apply the corresponding datetime.timedelta or relativedelta
    for unit, pattern in patterns.items():
        match = re.match(pattern, relative_time_str)
        if match:
            value = int(match.group(1))
            if unit == 'minutes':
                return now - relativedelta(minutes=value)
            elif unit == 'hours':
                return now - relativedelta(hours=value)
            elif unit == 'days':
                return now - relativedelta(days=value)
            elif unit == 'weeks':
                return now - relativedelta(weeks=value)
            elif unit == 'months':
                return now - relativedelta(months=value)
            elif unit == 'years':
                return now - relativedelta(years=value)

    # If no pattern matches, return None or raise an error
    return None

if __name__ == "__main__":
    print(convert_todate("2 days   ago"))  # Output: 2022-03-07
    print(convert_todate("3 weeks   ago"))  # Output: 2022-02-15
    print(convert_todate("1 year ago"))  # Output: 2021-03-10