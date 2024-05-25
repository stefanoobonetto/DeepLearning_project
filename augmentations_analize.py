import csv

def save_accuracy_and_names_to_csv(output_path):
    # Creare il file se non esiste
    
    after_MEMO_list = []
    before_MEMO_list = []

    with open('accuracy_results.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'after_MEMO' in row[0]:
                after_MEMO_list.append(float(row[1]))
            elif 'before_MEMO' in row[0]:
                before_MEMO_list.append(float(row[1]))

    # print("after_MEMO_list:", after_MEMO_list)
    # print("before_MEMO_list:", before_MEMO_list)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Scrivere l'intestazione solo se il file non esiste
        # if not file_exists:
        writer.writerow(["Class", "Accuracy Before MEMO", "Accuracy After MEMO", "Delta Accuracy", "Augmentations applied"])
        i = 0
        for before, after, name in zip(before_MEMO_list, after_MEMO_list, names):
            # if len(accuracy_classes[elem]) > 0:
                # accuracy = round((sum(accuracy_classes[elem]) / len(accuracy_classes[elem])) * 100, 2)
            string_name = ""
            for aug in name:
                string_name += aug
                string_name += "\n"
            writer.writerow([i, before, after, round(after-before, 2), string_name])
            i+=1


save_accuracy_and_names_to_csv()
