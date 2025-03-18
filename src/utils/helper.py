import datetime
import re
from dateutil.relativedelta import relativedelta


def convert_to_datetime(relative_time_str):
    """
        Converts a relative time string like '3 days ago' into a corresponding 
        datetime object.

        Args:
            relative_time_str (str): A string representing a relative date, such as
                                    '1 day ago', '3 days ago', etc.

        Returns:
            datetime: The corresponding datetime object for the specified relative time.
            
        Example:
            >>> convert_to_datetime('3 days ago')
            datetime.datetime(2025, 3, 3, 15, 30, 00)  # The exact date will depend on the current time
            
            Should return datetime in string to be compatible with FireStore datatypes in MM/DD/YYYY.
        """
   
    # First check if the datatype == null
    if not relative_time_str:
        return "03/07/2025"

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
    datetime_formatted = re.match(r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b", relative_time_str)
    if datetime_formatted:
        return relative_time_str
    
    for unit, pattern in patterns.items():
        match = re.match(pattern, relative_time_str)
        if match:
            value = int(match.group(1))
            if unit == 'minutes':
                return (now - relativedelta(minutes=value)).strftime("%m/%d/%Y")
            elif unit == 'hours':
                return (now - relativedelta(hours=value)).strftime("%m/%d/%Y")
            elif unit == 'days':
                return (now - relativedelta(days=value)).strftime("%m/%d/%Y")
            elif unit == 'weeks':
                return (now - relativedelta(weeks=value)).strftime("%m/%d/%Y")
            elif unit == 'months':
                return (now - relativedelta(months=value)).strftime("%m/%d/%Y")
            elif unit == 'years':
                return (now - relativedelta(years=value)).strftime("%m/%d/%Y")
            
    
    # If no pattern matches, return None
    return (now).strftime("%m/%d/%Y")
            
    
    # If no pattern matches, return None or raise an error
    raise ValueError(f"Invalid relative time string: {relative_time_str}")

























if __name__ == "__main__":
    print(convert_to_datetime("2 days   ago"))  # Output: 2022-03-07
    print(convert_to_datetime("3 weeks   ago"))  # Output: 2022-02-15
    print(convert_to_datetime("1 year ago"))  # Output: 2021-03-10
    print(convert_to_datetime("03/04/2025")) 