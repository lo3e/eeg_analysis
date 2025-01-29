import mne
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
import json

class EEGEmotionAnalyzer:
    def __init__(self, raw_dir, metadata_file):
        """
        Inizializza l'analizzatore
        
        Parameters:
        raw_dir: str - Directory contenente i file .set
        metadata_file: str - File JSON con metadati (gruppi, valence/arousal di riferimento)
        """
        self.raw_dir = Path(raw_dir)
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        
        # Parametri di pre-processing
        self.l_freq = 0.5  # Hz
        self.h_freq = 45   # Hz
        self.epoch_length = 2.0  # secondi
        self.overlap = 0.5  # 50% overlap
        
        # Bande di frequenza
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
    
    def preprocess_single_file(self, file_path):
        """Preprocess di un singolo file EEG"""
        # Carica il file
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        print(f"Canali presenti nel file: {raw.ch_names}")
        
        # Filtraggio
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        
        # ICA per rimozione artefatti
        ica = mne.preprocessing.ICA(n_components=14, random_state=42)
        ica.fit(raw)
        
        # Rimuovi componenti correlate con EOG (solo se il canale EOG è presente)
        if 'EOG' in raw.ch_names:
            eog_indices = []
            eog_scores = ica.score_sources(raw, target='EOG')
            eog_indices.extend(np.where(np.abs(eog_scores) > 0.3)[0])
            if eog_indices:
                ica.exclude = eog_indices
                raw = ica.apply(raw)
        else:
            print("Canale EOG non trovato.")

        # Creazione epoche
        events = mne.make_fixed_length_events(
            raw, 
            duration=self.epoch_length,
            overlap=self.overlap
        )
        epochs = mne.Epochs(
            raw, events,
            tmin=0, tmax=self.epoch_length,
            baseline=None,
            preload=True
        )
        
        return epochs
    
    def extract_features(self, epochs):
        """Estrae feature rilevanti per valence e arousal"""
        features = []
        
        for epoch in epochs:
            epoch_features = {}
            
            # Calcola PSD per ogni canale
            freqs, psd = signal.welch(epoch, fs=epochs.info['sfreq'],
                                    nperseg=int(epochs.info['sfreq']))
            
            # Calcola potenza per bande di frequenza
            for band_name, (fmin, fmax) in self.freq_bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(psd[:, freq_mask], axis=1)
                
                for ch_idx, ch_name in enumerate(epochs.ch_names):
                    epoch_features[f'{band_name}_{ch_name}'] = band_power[ch_idx]
            
            # Calcola asimmetria frontale
            if 'F3' in epochs.ch_names and 'F4' in epochs.ch_names:
                f3_idx = epochs.ch_names.index('F3')
                f4_idx = epochs.ch_names.index('F4')
                
                for band_name in self.freq_bands:
                    f3_power = epoch_features[f'{band_name}_F3']
                    f4_power = epoch_features[f'{band_name}_F4']
                    asymmetry = np.log(f4_power) - np.log(f3_power)
                    epoch_features[f'asymmetry_{band_name}'] = asymmetry
            
            # Calcola rapporti tra bande
            for ch_name in epochs.ch_names:
                beta_power = epoch_features[f'beta_{ch_name}']
                alpha_power = epoch_features[f'alpha_{ch_name}']
                if alpha_power > 0:  # evita divisione per zero
                    epoch_features[f'beta_alpha_ratio_{ch_name}'] = beta_power / alpha_power
            
            features.append(epoch_features)
        
        return pd.DataFrame(features)
    
    def process_dataset(self):
        """Processa l'intero dataset"""
        results = []
        
        for file_path in self.raw_dir.glob('**/*.set'):
            try:
                # Estrai informazioni dal path
                subject = file_path.parent.parent.name
                clip = file_path.stem
                
                print(f"Processing {subject} - {clip}")
                
                # Preprocessing
                epochs = self.preprocess_single_file(file_path)
                
                # Estrazione feature
                features_df = self.extract_features(epochs)
                
                # Aggiungi metadati
                features_df['subject'] = subject
                features_df['clip'] = clip
                features_df['group'] = self.metadata['groups'].get(subject)
                features_df['reference_valence'] = self.metadata['clips'][clip]['valence']
                features_df['reference_arousal'] = self.metadata['clips'][clip]['arousal']
                
                results.append(features_df)
                
            except Exception as e:
                print(f"Errore nel processare {file_path}: {str(e)}")
        
        # Combina tutti i risultati (solo se ci sono risultati)
        if results:
            final_df = pd.concat(results, ignore_index=True)
            return final_df
        else:
            print("Nessun file processato correttamente.")
            return pd.DataFrame()  # Restituisce un DataFrame vuoto

def main():
    # Configurazione
    raw_dir = "C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/dataset_spectra"
    metadata_file = "info.json"
    output_file = "emotion_features.csv"
    
    # Crea e esegui l'analizzatore
    analyzer = EEGEmotionAnalyzer(raw_dir, metadata_file)
    results_df = analyzer.process_dataset()
    
    # Salva i risultati
    results_df.to_csv(output_file, index=False)
    print(f"Analisi completata. Risultati salvati in {output_file}")

if __name__ == "__main__":
    main()