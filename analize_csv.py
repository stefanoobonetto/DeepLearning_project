import csv
import matplotlib.pyplot as plt

def leggi_csv(file_path):
    """
    Legge un file CSV e restituisce i dati sotto forma di lista di dizionari.
    
    :param file_path: Il percorso del file CSV da leggere.
    :return: Una lista di dizionari, dove ogni dizionario rappresenta una riga del CSV.
    """
    dati = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dati.append(row)

    return dati

# Esempio di utilizzo
file_path = '/home/bonnie/Desktop/deep_learning/project/DeepLearning_project/accuracy_results_VitB16.csv'
dati = leggi_csv(file_path)

dati_before_MEMO = []
dati_after_MEMO = []
dati_after_MEMO_PLUS = []

# class,accuracy,corr_conf,augmentations

for id, riga in enumerate(dati):
    if id != len(dati)-1:
        split_name = riga['class'].split('_')
        if split_name[1] == "before":
            dati_before_MEMO.append(float(riga['accuracy']))
        elif split_name[1] == "after":
            if len(split_name) > 3:
                dati_after_MEMO_PLUS.append(float(riga['accuracy']))
            else:
                dati_after_MEMO.append(float(riga['accuracy']))

print(len(dati_before_MEMO))
print(len(dati_after_MEMO))
print(len(dati_after_MEMO_PLUS))

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(dati_before_MEMO, label='Before MEMO', marker='o')
plt.plot(dati_after_MEMO, label='After MEMO', marker='o')
plt.plot(dati_after_MEMO_PLUS, label='After MEMO PLUS', marker='o')

plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()


# # Creazione del plot
# plt.figure(figsize=(10, 6))

# # Assumendo che i dati siano ordinati per classe o in ordine temporale
# plt.plot(dati_before_MEMO, label='Before MEMO', marker='o')
# plt.plot(dati_after_MEMO, label='After MEMO', marker='o')
# # plt.plot(dati_after_MEMO_PLUS, label='After MEMO PLUS', marker='o')

# # Aggiungere etichette e titolo
# plt.xlabel('Class')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Before and After MEMO')
# plt.legend()

# # Salva il grafico in un file
# output_path = 'accuracy_plot.pdf'
# plt.savefig(output_path)

# # Mostrare il grafico
# plt.show()

# print(f"Il grafico è stato salvato in {output_path}")
