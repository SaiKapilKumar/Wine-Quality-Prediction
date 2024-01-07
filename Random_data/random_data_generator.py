import csv
import numpy as np

# Define the ranges for each column
ranges = {
    'fixed acidity': (4.6, 15.9),
    'volatile acidity': (0.12, 1.58),
    'citric acid': (0.0, 1.0),
    'residual sugar': (0.9, 15.5),
    'chlorides': (0.012, 0.611),
    'free sulfur dioxide': (1, 72),
    'total sulfur dioxide': (6, 289),
    'density': (0.99007, 1.00369),
    'pH': (2.74, 4.01),
    'sulphates': (0.33, 2.0),
    'alcohol': (8.4, 14.9)
}

# Generate 30 lists of random values within the specified ranges in the provided format
num_lists = 30
all_lists = []
for _ in range(num_lists):
    list_format_data = []
    random_data = {}
    for column, (min_val, max_val) in ranges.items():
        random_data[column] = round(np.random.uniform(min_val, max_val),2)
        list_format_data.append(random_data[column])
    all_lists.append(list_format_data)

# Define the CSV file name
csv_file = 'random_data.csv'

# Save the generated 30 lists of random data into a CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header with column names
    writer.writerow(ranges.keys())
    # Write each list as a row in the CSV file
    writer.writerows(all_lists)

print(f"Data has been saved to '{csv_file}'.")
