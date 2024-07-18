import csv
import matplotlib.pyplot as plt
import numpy as np

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
file_path = '/home/sagemaker-user/DeepLearning_project/Results/test_15_07_ONLY_MEMO.csv'
dati = leggi_csv(file_path)

dati_before_MEMO = []
dati_after_MEMO = []
dati_after_MEMO_PLUS = []

for index, riga in enumerate(dati):

    if index != len(dati) - 1:
        split_name = riga['Class'].split('_')
        # print(split_name)
        if split_name[1] == "before":
      
            data = eval(riga['Result_for_image'])
            tot_correct = 0
            for elem in data:
                tot_correct += elem[0]
            accuracy = (tot_correct/len(data))*100
            dati_before_MEMO.append(accuracy)

        elif split_name[1] == "after":
            if len(split_name) == 3:
                data = eval(riga['Result_for_image'])
                tot_correct = 0
                for elem in data:
                    tot_correct += elem[0]
                accuracy = (tot_correct/len(data))*100
                dati_after_MEMO_PLUS.append(accuracy)
            else:
                data = eval(riga['Result_for_image'])
                tot_correct = 0
                for elem in data:
                    tot_correct += elem[0]
                accuracy = (tot_correct/len(data))*100
                dati_after_MEMO.append(accuracy)

print(len(dati))
print(len(dati_before_MEMO))
print(len(dati_after_MEMO))


# Impostazioni per il grafico
num_plots = 4
elements_per_plot = 50

# Creazione dei subplots
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 20))

for i in range(num_plots):
    start_index = i * elements_per_plot
    end_index = start_index + elements_per_plot
    index = np.arange(start_index, end_index)
    
    # Linea per la Serie A
    axs[i].plot(index, dati_before_MEMO[start_index:end_index], marker='o', label='Serie A')
    # Linea per la Serie B
    axs[i].plot(index, dati_after_MEMO[start_index:end_index], marker='x', label='Serie B')
    
    axs[i].set_title(f'Elementi {start_index + 1} a {end_index}')
    axs[i].set_xlabel('Indice Elemento')
    axs[i].set_ylabel('Value')
    axs[i].legend()

# Ajustar l'layout per evitare sovrapposizioni
plt.tight_layout()

# Mostra il grafico

# Save the plot to a file
output_path = '/home/sagemaker-user/DeepLearning_project/analize_script/stacked_bar_chart.pdf'
plt.savefig(output_path)