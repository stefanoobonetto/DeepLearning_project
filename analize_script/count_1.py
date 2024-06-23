import csv

filename = "/home/disi/DeepLearning_project/Results/test.csv"


count_before = 0
count_after = 0
count_after_plus = 0
elem_before = []
elem_after = []
elem_after_plus = []


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



print(f"Count before: {count_before}")
print(f"Count after: {count_after}")
print(f"Count after plus: {count_after_plus}")

print(f"MEMO-> before correct after wrong : {before_correct_memo} - before wrong after correct : {after_correct_memo}")
print(f"PLUS-> before correct after wrong : {before_correct_plus} - before wrong after correct : {after_correct_plus}")