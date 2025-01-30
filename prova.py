import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class EmotionAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [col for col in self.df.columns if any(col.startswith(prefix) 
                           for prefix in ['theta_', 'alpha_', 'beta_', 'asymmetry_', 'beta_alpha_ratio_'])]

    def adaptive_normalize(self, values, new_scale=10):
        """
        Normalizzazione adattiva che tiene conto della distribuzione dei dati
        """
        # Test di normalità
        _, normality_p_value = stats.normaltest(values)
        
        if normality_p_value > 0.05:  # Distribuzione approssimativamente normale
            # Usa z-score e mapping su scala 0-10
            z_scores = stats.zscore(values)
            normalized = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min()) * new_scale
        else:  # Distribuzione non normale
            # Usa trasformazione logaritmica per distribuzioni asimmetriche
            if np.all(values >= 0):  # Solo per valori positivi
                # Aggiungi una piccola costante per gestire gli zeri
                log_values = np.log1p(values)
                normalized = (log_values - log_values.min()) / (log_values.max() - log_values.min()) * new_scale
            else:
                # Per valori che possono essere negativi, usa normalizzazione quantile
                normalized = stats.rankdata(values) / len(values) * new_scale
        
        return normalized

    def winsorize_normalize(self, values, new_scale=10, limits=(0.05, 0.95)):
        """
        Normalizzazione con winsorization per gestire gli outlier
        """
        # Applica winsorization
        winsorized = stats.mstats.winsorize(values, limits=limits)
        
        # Normalizza nella nuova scala
        normalized = (winsorized - winsorized.min()) / (winsorized.max() - winsorized.min()) * new_scale
        return normalized

    def plot_distributions(self, raw_values, normalized_values, title):
        """
        Visualizza la distribuzione dei dati prima e dopo la normalizzazione
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot dei dati originali - Istogramma
        sns.histplot(raw_values, ax=ax1, kde=True, color='skyblue')
        ax1.set_title(f'Istogramma {title} (Prima)\n' + 
                     f'Skewness: {stats.skew(raw_values):.2f}\n' +
                     f'Kurtosis: {stats.kurtosis(raw_values):.2f}')
        
        # Plot dei dati originali - Boxplot
        ax2.boxplot(raw_values)
        ax2.set_title(f'Boxplot {title} (Prima)')
        
        # Plot dei dati normalizzati - Istogramma
        sns.histplot(normalized_values, ax=ax3, kde=True, color='lightgreen')
        ax3.set_title(f'Istogramma {title} (Dopo)\n' + 
                     f'Skewness: {stats.skew(normalized_values):.2f}\n' +
                     f'Kurtosis: {stats.kurtosis(normalized_values):.2f}')
        
        # Plot dei dati normalizzati - Boxplot
        ax4.boxplot(normalized_values)
        ax4.set_title(f'Boxplot {title} (Dopo)')
        
        plt.tight_layout()
        plt.show()

    def plot_valence_arousal_space(self):
        """
        Visualizza i gruppi e le clip nello spazio valence-arousal
        """
        # Calcola i valori grezzi
        raw_valence = self.df.apply(lambda row: (row['asymmetry_alpha'] * 0.6 + 
                                               (row['beta_alpha_ratio_F4'] - row['beta_alpha_ratio_F3']) * 0.4), 
                                   axis=1)
        
        raw_arousal = self.df.apply(lambda row: (row['beta_AF3'] + row['beta_AF4'] + 
                                               row['beta_F3'] + row['beta_F4']) / 4, 
                                   axis=1)
        
        # Applica le nuove normalizzazioni
        self.df['calculated_valence'] = self.adaptive_normalize(raw_valence)
        self.df['calculated_arousal'] = self.winsorize_normalize(raw_arousal)
        
        # Visualizza le distribuzioni separatamente
        print("Plotting distributions for Valence...")
        self.plot_distributions(raw_valence, self.df['calculated_valence'], 'Valence')
        
        print("\nPlotting distributions for Arousal...")
        self.plot_distributions(raw_arousal, self.df['calculated_arousal'], 'Arousal')
        
        # Plot dello spazio valence-arousal
        print("\nPlotting Valence-Arousal space...")
        self.plot_va_space_by_group()
        self.plot_va_space_by_clip()

    def plot_va_space_by_group(self):
        """
        Visualizza lo spazio valence-arousal raggruppato per gruppi
        """
        plt.figure(figsize=(12, 8))
        
        # Colori per i gruppi
        group_colors = {'control': 'blue', 'resistant': 'red', 'responsive': 'green'}
        markers = {'Clip_1': 'o', 'Clip_2': 's', 'Clip_3': '^'}
        
        # Plot per ogni gruppo
        for group in self.df['group'].unique():
            group_data = self.df[self.df['group'] == group]
            
            for clip in group_data['clip'].unique():
                clip_data = group_data[group_data['clip'] == clip]
                plt.scatter(clip_data['calculated_valence'], 
                          clip_data['calculated_arousal'],
                          label=f'{group} - {clip}',
                          color=group_colors.get(group, 'gray'),
                          marker=markers.get(clip, 'o'),
                          alpha=0.6,
                          s=100)  # Aumentato dimensione punti

        # Aggiungi i valori di riferimento
        for clip in self.df['clip'].unique():
            ref_data = self.df[self.df['clip'] == clip].iloc[0]
            plt.scatter(ref_data['reference_valence'], 
                      ref_data['reference_arousal'],
                      marker='*',
                      s=300,  # Aumentato dimensione stella
                      color='black',
                      label=f'{clip} (reference)')
            
            plt.annotate(clip,
                        (ref_data['reference_valence'], ref_data['reference_arousal']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.title('Spazio Valence-Arousal per Gruppo', fontsize=14)
        plt.xlabel('Valence', fontsize=12)
        plt.ylabel('Arousal', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.5, 10.5])
        plt.ylim([-0.5, 10.5])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_va_space_by_clip(self):
        """
        Visualizza lo spazio valence-arousal raggruppato per clip
        """
        plt.figure(figsize=(12, 8))
        
        # Colori per i gruppi
        group_colors = {'control': 'blue', 'resistant': 'red', 'responsive': 'green'}
        markers = {'Clip_1': 'o', 'Clip_2': 's', 'Clip_3': '^'}
        
        # Plot per ogni clip
        for clip in self.df['clip'].unique():
            clip_data = self.df[self.df['clip'] == clip]
            
            for group in clip_data['group'].unique():
                group_data = clip_data[clip_data['group'] == group]
                plt.scatter(group_data['calculated_valence'], 
                          group_data['calculated_arousal'],
                          label=f'{clip} - {group}',
                          color=group_colors.get(group, 'gray'),
                          marker=markers.get(clip, 'o'),
                          alpha=0.6,
                          s=100)  # Aumentato dimensione punti

        # Aggiungi i valori di riferimento
        for clip in self.df['clip'].unique():
            ref_data = self.df[self.df['clip'] == clip].iloc[0]
            plt.scatter(ref_data['reference_valence'], 
                      ref_data['reference_arousal'],
                      marker='*',
                      s=300,  # Aumentato dimensione stella
                      color='black',
                      label=f'{clip} (reference)')
            
            plt.annotate(clip,
                        (ref_data['reference_valence'], ref_data['reference_arousal']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.title('Spazio Valence-Arousal per Clip', fontsize=14)
        plt.xlabel('Valence', fontsize=12)
        plt.ylabel('Arousal', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.5, 10.5])
        plt.ylim([-0.5, 10.5])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def main():
    analyzer = EmotionAnalyzer('emotion_features.csv')
    analyzer.plot_valence_arousal_space()

if __name__ == "__main__":
    main()