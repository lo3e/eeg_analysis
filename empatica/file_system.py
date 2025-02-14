import os

def describe_file_system(directory, target_folder, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(os.path.join(directory, target_folder)):
            level = root.replace(os.path.join(directory, target_folder), '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}{os.path.basename(root)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f'{sub_indent}{file}\n')

# Percorso della directory principale
start_directory = 'C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/Empatica/participant_data' # Modifica con il percorso corretto
target_folder = ''  # Sottocartella target
output_file = 'dataset_file_system.txt'

# Descrivi il file system relativo alla sottocartella target e scrivi il risultato in un file di testo
describe_file_system(start_directory, target_folder, output_file)

print(f"Descrizione della struttura del file system salvata in: {output_file}")