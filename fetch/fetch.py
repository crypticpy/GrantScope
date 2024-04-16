import requests
import time
import json
import os
from tqdm import tqdm

def get_grants_transactions(page_number, year_range, dollar_range, subjects, populations, locations, transaction_types):
    start_year, end_year = year_range
    min_amt, max_amt = dollar_range

    url = f"https://api.candid.org/grants/v1/transactions?page={page_number}&location={','.join(locations)}&geo_id_type=geonameid&location_type=area_served&year={','.join(map(str, range(start_year, end_year + 1)))}&subject={','.join(subjects)}&population={','.join(populations)}&support=&transaction={','.join(transaction_types)}&recip_id=&funder_id=&include_gov=yes&min_amt={min_amt}&max_amt={max_amt}&sort_by=year_issued&sort_order=desc&format=json"

    headers = {
        "accept": "application/json",
        "Subscription-Key": "KEY"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error getting grants transactions: {e}")

def validate_input(value, value_type, min_value=None, max_value=None):
    try:
        parsed_value = value_type(value)
        if min_value is not None and parsed_value < min_value:
            raise ValueError(f"Value should be greater than or equal to {min_value}")
        if max_value is not None and parsed_value > max_value:
            raise ValueError(f"Value should be less than or equal to {max_value}")
        return parsed_value
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

def get_unique_file_name(file_name):
    base_name, extension = os.path.splitext(file_name)
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return file_name

def main():
    calls_per_minute = 9
    delay = 60 / calls_per_minute

    print("Welcome to the Candid API Grants Data Fetcher!")
    print("This tool will guide you through the process of fetching grants data from the Candid API.")

    try:
        start_year = validate_input(input("Enter the start year: "), int, min_value=1900, max_value=2100)
        end_year = validate_input(input("Enter the end year: "), int, min_value=start_year, max_value=2100)
        year_range = (start_year, end_year)

        min_amt = validate_input(input("Enter the minimum dollar amount (e.g., 25000): "), int, min_value=0)
        max_amt = validate_input(input("Enter the maximum dollar amount (e.g., 10000000): "), int, min_value=min_amt)
        dollar_range = (min_amt, max_amt)

        subjects = input("Enter the subjects (comma-separated, e.g., SJ02,SJ05): ").split(",")
        populations = input("Enter the populations (comma-separated, e.g., PA010000,PC040000): ").split(",")
        locations = input("Enter the locations (comma-separated geonameid, e.g., 4671654,4736286): ").split(",")
        transaction_types = input("Enter the transaction types (comma-separated, e.g., TA,TG): ").split(",")

        num_pages = input("Enter the number of pages to retrieve (or 'all' for all pages): ")
        if num_pages.lower() == 'all':
            num_pages = None
        else:
            num_pages = validate_input(num_pages, int, min_value=1)

        output_file = input("Enter the output file name (e.g., grants_data.json): ")
        output_file = get_unique_file_name(output_file)

        all_grants = []
        page_number = 1

        print("Fetching grants data...")
        while True:
            grants_data = get_grants_transactions(page_number, year_range, dollar_range, subjects, populations, locations, transaction_types)
            all_grants.extend(grants_data["grants"])

            total_pages = grants_data["total_pages"]
            progress_bar = tqdm(total=total_pages, unit='page', desc='Progress', initial=page_number)

            if num_pages is None and total_pages == page_number:
                break
            elif num_pages is not None and page_number == num_pages:
                break

            page_number += 1
            progress_bar.update(1)
            time.sleep(delay)  # Pause for required delay time

        progress_bar.close()

        with open(output_file, "w") as f:
            json.dump({"grants": all_grants}, f, indent=2)

        print(f"Grants data saved to {output_file}")
        print("Thank you for using the Candid API Grants Data Fetcher!")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()