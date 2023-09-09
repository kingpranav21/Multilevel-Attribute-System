import csv

# In this code:

# columns is a list containing the column names for your CSV file.
# data is a list of lists, where each inner list represents a row of data.
# file_path is the path where the CSV file will be created. You can change this path to specify where you want to save the file.
# When you run this code, it will create a CSV file named "attributed_social_network_data.csv" with the specified column names and sample data. You can customize the data and column names to match your actual dataset.

# If you have specific sample data you'd like to include or if you'd like to create a CSV file with different columns, please provide that information, and I can modify the code accordingly.
# Define column names
columns = ["NodeID", "Attribute1", "Attribute2", "Attribute3", "Target"]

# Define sample data rows
data = [
    [1, 0.5, 25, "Male", 0],
    [2, 0.8, 30, "Female", 1],
    [3, 0.6, 22, "Male", 1],
    [4, 0.7, 28, "Female", 0],
    [5, 0.9, 35, "Female", 1],
    [6, 0.4, 20, "Male", 0],
]

# Specify the file path where you want to save the CSV file
file_path = "attributed_social_network_data.csv"

# Create and write to the CSV file
with open(file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the column names as the header
    csv_writer.writerow(columns)
    
    # Write the sample data rows
    csv_writer.writerows(data)

print(f'CSV file "{file_path}" has been created with sample data.')


