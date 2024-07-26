
import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    """
    Reads a CSV file and returns the data as a list of dictionaries.
    
    :param file_path: The path to the CSV file to read.
    :return: A list of dictionaries, where each dictionary represents a row in the CSV.
    """
    data = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    return data

# Example usage
file_path = '/home/sagemaker-user/DeepLearning_project/Results/test_16_07_ONLY_MEMO_PLUS.csv'
data = read_csv(file_path)

data_before_MEMO = []
data_after_MEMO = []
data_after_MEMO_PLUS = []

for index, row in enumerate(data):

    if index != len(data) - 1:
        split_name = row['Class'].split('_')
        # print(split_name)
        if split_name[1] == "before":
            data_row = eval(row['Result_for_image'])
            tot_correct = 0
            for elem in data_row:
                tot_correct += elem[0]
            accuracy = (tot_correct/len(data_row))*100
            data_before_MEMO.append(accuracy)

        elif split_name[1] == "after":
            data_row = eval(row['Result_for_image'])
            tot_correct = 0
            for elem in data_row:
                tot_correct += elem[0]
            accuracy = (tot_correct/len(data_row))*100
            if len(split_name) == 3:
                data_after_MEMO_PLUS.append(accuracy)
            else:
                data_after_MEMO.append(accuracy)

print(len(data))
print(len(data_before_MEMO))
print(len(data_after_MEMO))

# Settings for the plot
num_plots = 4
elements_per_plot = 50

# Create subplots (2 plots per row, 2 rows)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i in range(num_plots):
    row = i // 2
    col = i % 2
    start_index = i * elements_per_plot
    end_index = start_index + elements_per_plot
    index = np.arange(start_index, end_index)
    
    # Line for Serie A
    axs[row, col].plot(index, data_before_MEMO[start_index:end_index], marker='o', label='before MEMO_PLUS')
    # Line for Serie B
    axs[row, col].plot(index, data_after_MEMO[start_index:end_index], marker='x', label='after MEMO_PLUS')
    
    axs[row, col].set_title(f'Classes {start_index + 1} to {end_index}')
    axs[row, col].set_xlabel('Classes Index')
    axs[row, col].set_ylabel('Value')
    
    if(i == 1):
        axs[row, col].legend()

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plot

# Save the plot to a file
output_path = '/home/sagemaker-user/DeepLearning_project/analize_script/stacked_bar_chart.jpg'
plt.savefig(output_path)
