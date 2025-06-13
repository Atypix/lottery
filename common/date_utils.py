import pandas as pd
from datetime import datetime, timedelta, date
import os # Added
    """
    Calculates the next Euromillions draw date.
    Euromillions draws are on Tuesdays and Fridays.
    This function first determines the latest draw date recorded in the provided data file.
    Then, it searches for the next Tuesday or Friday strictly after this latest recorded date.
    If the data file is missing, empty, or contains no valid dates, the search for the
    next draw date defaults to starting from the current system date.

    Args:
        data_file_path: Path to the CSV dataset (e.g., "euromillions_enhanced_dataset.csv").

    Returns:
        The next Euromillions draw date as a datetime.date object.
    """
    latest_date_from_file = None
    original_path_arg = data_file_path # Keep original for messages

    # Determine paths to try
    path_in_data_subdir = None
    # Check if data_file_path is a simple filename (no directory part)
    if os.path.basename(data_file_path) == data_file_path:
        path_in_data_subdir = os.path.join("data", data_file_path)

    paths_to_try = []
    if path_in_data_subdir:
        paths_to_try.append(path_in_data_subdir)
    paths_to_try.append(data_file_path) # Original path (could be absolute or relative)

    df_loaded = False
    for path_attempt in paths_to_try:
        try:
            df = pd.read_csv(path_attempt)
            if 'Date' not in df.columns or df.empty:
                print(f"Warning: 'Date' column not found or DataFrame empty in {path_attempt}. Trying next path or defaulting.")
                # Don't set latest_date_from_file here, let loop continue or fall to final default
            else:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
                if df.empty:
                    print(f"Warning: No valid dates in 'Date' column in {path_attempt} after parsing. Trying next path or defaulting.")
                else:
                    latest_date_from_file = df['Date'].max().date()
                    print(f"Info: Successfully loaded and parsed dates from {path_attempt}")
                    df_loaded = True
                    break # Exit loop on successful load and parse
        except FileNotFoundError:
            print(f"Info: Data file {path_attempt} not found. Trying next path or defaulting.")
        except Exception as e:
            print(f"Error reading or parsing {path_attempt}: {e}. Trying next path or defaulting.")
            # Potentially set self.df to None or handle more explicitly if this class had state

    if not df_loaded:
        print(f"Warning: Could not load data from any attempted path ({', '.join(paths_to_try)}). Defaulting to current date for latest_date_from_file determination.")
        latest_date_from_file = datetime.now().date()

    current_search_date = datetime.now().date()

    while True:
        weekday = current_search_date.weekday()  # Monday is 0, Tuesday is 1, ..., Friday is 4, ...
        if weekday == 1 or weekday == 4: # Tuesday or Friday
            if current_search_date > latest_date_from_file:
                return current_search_date
        current_search_date += timedelta(days=1)

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
