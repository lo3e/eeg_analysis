import mne
import pandas as pd
import os
import glob

def extract_subject_info(filepath):
    """Estrae le informazioni del soggetto dal percorso del file"""
    subject = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    clip = os.path.basename(filepath).replace('.set', '')
    return subject, clip

def process_eeg_files(root_dir):
    """
    Processa tutti i file .set nella directory e crea un dataframe
    mantenendo tutti i campioni temporali
    
    Parameters:
    root_dir (str): Percorso della directory principale del dataset
    
    Returns:
    pd.DataFrame: Dataframe contenente tutti i dati EEG organizzati
    """
    all_data = []
    
    pattern = os.path.join(root_dir, "**", "*.set")
    eeg_files = glob.glob(pattern, recursive=True)
    
    for file_path in eeg_files:
        try:
            # Estrai informazioni sul soggetto e clip
            subject, clip = extract_subject_info(file_path)
            
            # Carica il file EEG
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            
            # Converti i dati EEG in un array
            data = raw.get_data()
            
            # Ottieni il numero di campioni temporali
            n_times = data.shape[1]
            
            # Per ogni punto temporale
            for time_point in range(n_times):
                # Crea un dizionario per questo punto temporale
                time_data = {
                    'Soggetto': subject,
                    'Clip': clip,
                    'TimePoint': time_point,
                    'Time_sec': time_point / raw.info['sfreq']  # converti in secondi
                }
                
                # Aggiungi i valori di tutti i canali per questo punto temporale
                for i, channel in enumerate(raw.ch_names):
                    time_data[f'channel_{channel}'] = data[i, time_point]
                
                all_data.append(time_data)
            
            print(f"Processato {subject} - {clip}")
            
        except Exception as e:
            print(f"Errore nel processare {file_path}: {str(e)}")
    
    # Crea il dataframe finale
    df = pd.DataFrame(all_data)
    
    # Ordina il dataframe
    df = df.sort_values(['Soggetto', 'Clip', 'TimePoint'])
    
    return df

def main():
    # Specifica il percorso della directory principale del dataset
    dataset_dir = "C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/dataset_spectra"
    
    # Processa i file e crea il dataframe
    df = process_eeg_files(dataset_dir)
    
    # Salva il dataframe in un file CSV
    df.to_csv('eeg_dataset.csv', index=False)
    
    print("Elaborazione completata!")
    print(f"Shape del dataframe: {df.shape}")
    print("\nPrime righe del dataframe:")
    print(df.head())

if __name__ == "__main__":
    main()