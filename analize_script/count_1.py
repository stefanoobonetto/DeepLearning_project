import csv

filename = "/home/sagemaker-user/DeepLearning_project/Results/test_16_07_ONLY_MEMO_PLUS_ALL.csv"


count_before = 0
count_after = 0
count_after_plus = 0
elem_before = []
elem_after = []
elem_after_plus = []

count_all_images = 0
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        name = row[0]
        
        # Do something with the result_for_image variable
        if (len(name.split('_')) == 1):
            continue
        result_for_image = eval(row[2])  # Assuming Result_for_image is the first column
        if name.split('_')[1] == "before":
            for elem in result_for_image:
                count_all_images += 1
                elem_before.append(elem[0])
                if elem[0] == 1:
                    count_before += 1
        elif name.split('_')[-1] == "PLUS":
            for elem in result_for_image:
                elem_after_plus.append(elem[0])
                if elem[0] == 1:
                    count_after_plus += 1
        elif name.split('_')[1] == "after":
            for elem in result_for_image:
                elem_after.append(elem[0])
                if elem[0] == 1:
                    count_after += 1
        else:
            continue

before_correct_memo = 0
after_correct_memo = 0
before_correct_plus = 0
after_correct_plus = 0


for before, after, after_plus in zip(elem_before, elem_after, elem_after_plus):
    if (before != after) and before == 1:
        before_correct_memo += 1
    if (before != after_plus) and before == 1:
        before_correct_plus += 1
    if (before != after) and after == 1:
        after_correct_memo += 1
    if (before != after_plus) and after_plus == 1:
        after_correct_plus += 1


print(count_all_images)

print(f"Count before: {count_before}  Accuracy: {(count_before)/7500*100}")
print(f"Count after: {count_after}. Accuracy: {(count_after)/7500*100}")
print(f"Count after plus: {count_after_plus}. Accuracy: {count_after_plus/7500*100}")

print(f"MEMO-> before correct after wrong : {before_correct_memo} - before wrong after correct : {after_correct_memo}")
print(f"PLUS-> before correct after wrong : {before_correct_plus} - before wrong after correct : {after_correct_plus}")