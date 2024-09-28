import json
import csv

# Define the path to your CSV file
file_path = "data_analysis/nodes.csv"

# Open the file and read line by line, skipping the first line
with open(file_path, "r", encoding="utf-8") as infile:
    # Skip the first line (assuming the first line needs to be skipped)
    next(infile)

    # Initialize the list for dynamically gathered field names and data
    all_keys = set()
    data_rows = []

    # Process the remaining lines
    for line in infile:
        try:
            # Parse the JSON data
            json_data = json.loads(line.strip())

            # Flatten the dictionary and collect all unique keys dynamically
            flat_data = {}
            for key, value in json_data.items():
                if isinstance(value, dict):
                    # Flatten nested dictionaries
                    for sub_key, sub_value in value.items():
                        flat_data[f"{key}_{sub_key}"] = sub_value
                        all_keys.add(f"{key}_{sub_key}")
                else:
                    flat_data[key] = value
                    all_keys.add(key)

            # Store the flattened data
            data_rows.append(flat_data)

        except json.JSONDecodeError:
            print(f"Error decoding JSON for line: {line}")
        except KeyError as e:
            print(f"Missing expected field {e} in line: {line}")

# Convert the set of all keys into a sorted list (to ensure consistent column ordering)
all_keys = sorted(all_keys)

# Write to CSV
with open(
    "data_analysis/processed_nodes.csv", "w", newline="", encoding="utf-8"
) as outfile:
    writer = csv.DictWriter(outfile, fieldnames=all_keys)

    # Write header
    writer.writeheader()

    # Write rows
    for row in data_rows:
        writer.writerow(row)

print("Dynamic CSV file has been written successfully!")
