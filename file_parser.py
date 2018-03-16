import csv

training_data_file = "training_data_example.csv"
validation_data_file = "validation_data_example.csv"
employee_file = "employee.csv"

def parse_csv_data(input_file):
    data_list = [] # list of lists
    header_list = [] # list containing just the header row
    with open(input_file, "r") as csv_file:
        parsed_data = csv.reader(csv_file, delimiter=",")
        i = 0
        for row in parsed_data:
            if i == 0:
                header_list = row
            else:
                data_list.append(row)
            i = i + 1

    return data_list, header_list

