import pandas as pd
from datetime import datetime, timedelta, date

def get_next_euromillions_draw_date(data_file_path: str) -> date:
    """
    Calculates the next Euromillions draw date based on the latest date in the dataset.
    Euromillions draws are typically on Tuesdays and Fridays.

    Args:
        data_file_path: Path to the CSV dataset (e.g., "euromillions_enhanced_dataset.csv").

    Returns:
        The next Euromillions draw date as a datetime.date object.
        Returns today's date if the dataset is empty or date column is problematic,
        though a more robust error or default might be needed for production.
    """
    try:
        df = pd.read_csv(data_file_path)
        if 'Date' not in df.columns or df.empty: # Added df.empty check here
            print(f"Warning: 'Date' column not found or empty in {data_file_path}. Defaulting to calculate from today.")
            latest_date = datetime.now().date()
        else:
            # Ensure 'Date' column is parsed as datetime objects
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True) # Remove rows where date couldn't be parsed

            if df.empty:
                print(f"Warning: No valid dates found in {data_file_path} after parsing. Defaulting to calculate from today.")
                latest_date = datetime.now().date()
            else:
                latest_date = df['Date'].max().date()

    except FileNotFoundError:
        print(f"Warning: Data file {data_file_path} not found. Defaulting to calculate from today.")
        latest_date = datetime.now().date()
    except Exception as e:
        print(f"Error reading or parsing dates from {data_file_path}: {e}. Defaulting to calculate from today.")
        latest_date = datetime.now().date()

    next_date_candidate = latest_date + timedelta(days=1)

    while True:
        weekday = next_date_candidate.weekday()  # Monday is 0 and Sunday is 6
        # Tuesday is 1, Friday is 4
        # We need to ensure that if latest_date itself is a draw date, we find the *next* one.
        # So, if next_date_candidate is a draw day AND it's strictly greater than latest_date, it's a valid candidate.
        # However, our loop structure with `latest_date + timedelta(days=1)` and then incrementing
        # already ensures we are looking for a date *after* latest_date.
        if weekday == 1 or weekday == 4: # Tuesday or Friday
            # If latest_date was a draw date, and next_date_candidate is that same date,
            # this means the latest_date in the file was yesterday, and today is a draw day.
            # This logic correctly finds the *next upcoming* draw date.
            # If the latest_date in the file is *today* and *today* is a draw day,
            # this will find the draw day *after* today.
            if next_date_candidate > latest_date:
                 return next_date_candidate
            # If latest_date is today and a draw day, we need the *next* draw day.
            # This case is handled by just continuing to increment next_date_candidate.
            # The condition next_date_candidate > latest_date handles the case where the
            # loop starts on a draw day that is also latest_date.
            # To be certain, if latest_date is a draw date, we must ensure next_date_candidate is a *future* draw.
            # The current logic seems to handle this: we start checking from `latest_date + 1 day`.
            # Let's refine the loop slightly for clarity for the case where latest_date itself is a draw day.
            # The current logic is: start from tomorrow, find the first Tue/Fri. This is correct.
            return next_date_candidate
        next_date_candidate += timedelta(days=1)

if __name__ == '__main__':
    # Create a dummy CSV for testing
    dummy_data_1 = {
        'Date': ['2023-10-20', '2023-10-24', '2023-10-27'] # Fri, Tue, Fri
    }
    dummy_df_1 = pd.DataFrame(dummy_data_1)
    dummy_csv_path = "dummy_test_draws.csv"
    dummy_df_1.to_csv(dummy_csv_path, index=False)

    print(f"Test 1: Latest draw in file is 2023-10-27 (Friday)")
    next_draw_1 = get_next_euromillions_draw_date(dummy_csv_path)
    print(f"Expected next draw: 2023-10-31 (Tuesday), Got: {next_draw_1}")
    assert next_draw_1 == date(2023, 10, 31)

    # Test case: latest date in file is a Tuesday
    dummy_data_2 = {
        'Date': ['2023-10-27', '2023-10-31'] # Fri, Tue
    }
    dummy_df_2 = pd.DataFrame(dummy_data_2)
    dummy_df_2.to_csv(dummy_csv_path, index=False) # Overwrite
    print(f"\nTest 2: Latest draw in file is 2023-10-31 (Tuesday)")
    next_draw_2 = get_next_euromillions_draw_date(dummy_csv_path)
    print(f"Expected next draw: 2023-11-03 (Friday), Got: {next_draw_2}")
    assert next_draw_2 == date(2023, 11, 3)

    # Test case: script is run on a draw day, data is up to the previous draw day
    # Example: Today is Friday 2023-11-03. Latest data in file is Tuesday 2023-10-31.
    # Should give 2023-11-03
    # This is covered by Test 2 if we imagine "today" is after 2023-10-31.

    # Test case: script is run on a draw day, data is up to date (includes today's date as latest)
    # Example: Today is Friday 2023-11-03. Latest data in file is Friday 2023-11-03.
    # Should give next draw date, i.e., Tuesday 2023-11-07
    dummy_data_3 = {
        'Date': ['2023-10-31', '2023-11-03'] # Tue, Fri
    }
    dummy_df_3 = pd.DataFrame(dummy_data_3)
    dummy_df_3.to_csv(dummy_csv_path, index=False) # Overwrite
    print(f"\nTest 3: Latest draw in file is 2023-11-03 (Friday)")
    next_draw_3 = get_next_euromillions_draw_date(dummy_csv_path)
    print(f"Expected next draw: 2023-11-07 (Tuesday), Got: {next_draw_3}")
    assert next_draw_3 == date(2023, 11, 7)

    print("\nTest 4: Non-existent file (should default from today)")
    # Determine expected date from today for verification printout
    # This logic must be identical to the function's fallback logic path.
    expected_from_today_calc_base = datetime.now().date()
    expected_from_today = expected_from_today_calc_base + timedelta(days=1)
    while True:
        if expected_from_today.weekday() == 1 or expected_from_today.weekday() == 4: break
        expected_from_today += timedelta(days=1)

    next_draw_non_existent = get_next_euromillions_draw_date("non_existent_test_file.csv")
    print(f"Expected next draw (from today's date {datetime.now().date()}): {expected_from_today}, Got: {next_draw_non_existent}")
    assert next_draw_non_existent == expected_from_today

    print("\nTest 5: Empty CSV file")
    empty_csv_path = "empty_test_draws.csv"
    with open(empty_csv_path, 'w') as f:
        f.write("Date\n") # Header only, no data lines
    next_draw_empty_csv = get_next_euromillions_draw_date(empty_csv_path)
    print(f"Expected next draw (from today for empty CSV): {expected_from_today}, Got: {next_draw_empty_csv}")
    assert next_draw_empty_csv == expected_from_today

    print("\nTest 6: CSV file with Date column but all unparsable dates")
    dummy_data_invalid_dates = {'Date': ['invalid1', 'invalid2']}
    dummy_df_invalid = pd.DataFrame(dummy_data_invalid_dates)
    dummy_df_invalid.to_csv(dummy_csv_path, index=False)
    next_draw_invalid_dates = get_next_euromillions_draw_date(dummy_csv_path)
    print(f"Expected next draw (from today for invalid dates): {expected_from_today}, Got: {next_draw_invalid_dates}")
    assert next_draw_invalid_dates == expected_from_today

    print("\nAll tests passed.")

    # Clean up dummy files
    import os
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
    if os.path.exists(empty_csv_path):
        os.remove(empty_csv_path)
