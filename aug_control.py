import pandas as pd
from collections import Counter
import csv
import ast
# Sample CSV data

# df = pd.read_csv('/home/bonnie/Desktop/deep_learning/project/DeepLearning_project/accuracy_results.csv')

after_MEMO_list = []
before_MEMO_list = []
names = []
correct_after = []
correct_before = []

with open('accuracy_results.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if 'after_MEMO' in row[0]:
            after_MEMO_list.append(float(row[1]))
            names.append(eval(row[3]))
            correct_after.append(row[2])
        elif 'before_MEMO' in row[0]:
            before_MEMO_list.append(float(row[1]))
            correct_before.append(row[2])

with open("accuracy_results_2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Scrivere l'intestazione solo se il file non esiste
    # if not file_exists:
    writer.writerow(["class", "before", "after", "delta", "correct_before", "correct_after", "augmentations"])
    i = 0
    for before, after, corr_bef, corr_aft, name in zip(before_MEMO_list, after_MEMO_list, correct_before, correct_after, names):
        # if len(accuracy_classes[elem]) > 0:
            # accuracy = round((sum(accuracy_classes[elem]) / len(accuracy_classes[elem])) * 100, 2)
        # string_name = ""
        # for aug in name:
        #     string_name += str(aug)
        #     string_name += "\n"
        writer.writerow([i, before, after, round(after-before, 2), corr_bef, corr_aft, name])
        i+=1




with open("accuracy_results_2.csv", mode='r', newline='') as file:
    reader = csv.reader(file)
    selected_rows = []
    selected_augmentations = []
    for row in reader:
        if row[3] != "delta" and float(row[3]) > 0:
            selected_rows.append(row)
    for row in selected_rows:
        # print(type(ast.literal_eval(row[5])[0]))
        for id, elem in enumerate(ast.literal_eval(row[5])):
            if elem == 1:
                selected_augmentations.append(ast.literal_eval(row[6])[id])
    
    merged_augmentations = [aug for sublist in selected_augmentations for aug in sublist]

    print(merged_augmentations)

    augmentation_counts = Counter(merged_augmentations)

    print("Most frequent augmentations in improved classes:")
    for augmentation, count in augmentation_counts.items():
        if augmentation != 'Original':
            print(f"{augmentation}: {count}")



with open("accuracy_results_2.csv", mode='r', newline='') as file:
    reader = csv.reader(file)
    bad_aug = []
    
    for row in reader:
        if row[4] != "correct_before":
            for i, elem in enumerate(ast.literal_eval(row[4])):
                if elem == 1 and ast.literal_eval(row[5])[i] == 0:
                    bad_aug.append(ast.literal_eval(row[6])[i])    

bad_aug = [aug for sublist in bad_aug for aug in sublist]


augmentation_counts = Counter(bad_aug)

print("Most frequent augmentations in 0 to 1:")
for augmentation, count in augmentation_counts.most_common():
    if augmentation != 'Original':
        print(f"{augmentation}: {count}")


# df['improvement'] = df['after_accuracy'] > df['before_accuracy']
# improved_classes = df[df['improvement']]

# # Extract and count augmentations
# augmentation_counts = Counter()
# for augmentations in improved_classes['augmentations']:
#     augmentations_list = eval(augmentations)  # Convert string representation of list to list
#     for augment in augmentations_list:
#         augmentation_counts.update(augment)

# # Display the most frequent augmentations
# print("Most frequent augmentations in improved classes:")
# for augmentation, count in augmentation_counts.most_common():
#     print(f"{augmentation}: {count}")