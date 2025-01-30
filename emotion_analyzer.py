import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class EmotionAnalyzer:
    def __init__(self, csv_path):
        """
        Inizializza l'analizzatore con il dataset
        """
        self.df = pd.read_csv(csv_path)
        # Estrai solo le colonne delle feature
        self.feature_cols = [col for col in self.df.columns if any(col.startswith(prefix) 
                           for prefix in ['theta_', 'alpha_', 'beta_', 'asymmetry_', 'beta_alpha_ratio_'])]
        # Stampa le colonne disponibili per debug
        print("Colonne disponibili nel DataFrame:")
        print(self.df.columns.tolist())
        print("\nFeature columns:")
        print(self.feature_cols)

    def robust_normalize(self, values, new_scale=10, lower_percentile=5, upper_percentile=95):
        """
        Normalizza i valori in una nuova scala (es. 0-10) utilizzando i percentili per gestire gli outlier.
        
        Parameters:
        values: pd.Series - Valori da normalizzare
        new_scale: int - Valore massimo della nuova scala (default: 10)
        lower_percentile: float - Percentile inferiore per il calcolo del minimo (default: 1)
        upper_percentile: float - Percentile superiore per il calcolo del massimo (default: 99)
        
        Returns:
        pd.Series - Valori normalizzati
        """
        # Calcola i percentili inferiore e superiore
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        
        # Clip dei valori per rimuovere gli outlier
        clipped_values = np.clip(values, lower_bound, upper_bound)
        
        # Normalizza i valori nella nuova scala
        normalized_values = ((clipped_values - lower_bound) / (upper_bound - lower_bound)) * new_scale
        
        return normalized_values

    def plot_distributions(self, raw_values, normalized_values, title):
        """
        Visualizza la distribuzione dei dati prima e dopo la normalizzazione.
        
        Parameters:
        raw_values: pd.Series - Valori non normalizzati
        normalized_values: pd.Series - Valori normalizzati
        title: str - Titolo del grafico
        """
        # Crea una figura con due subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Boxplot dei valori non normalizzati
        sns.boxplot(raw_values, ax=ax1, color='skyblue')
        ax1.set_title(f'Boxplot - {title} (Prima)')
        ax1.set_xlabel('Valori')
        
        # Boxplot dei valori normalizzati
        sns.boxplot(normalized_values, ax=ax2, color='lightgreen')
        ax2.set_title(f'Boxplot - {title} (Dopo)')
        ax2.set_xlabel('Valori normalizzati')
        
        plt.tight_layout()
        plt.show()

        # Istogrammi dei valori non normalizzati e normalizzati
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Istogramma dei valori non normalizzati
        sns.histplot(raw_values, ax=ax1, color='skyblue', kde=True)
        ax1.set_title(f'Istogramma - {title} (Prima)')
        ax1.set_xlabel('Valori')
        ax1.set_ylabel('Frequenza')
        
        # Istogramma dei valori normalizzati
        sns.histplot(normalized_values, ax=ax2, color='lightgreen', kde=True)
        ax2.set_title(f'Istogramma - {title} (Dopo)')
        ax2.set_xlabel('Valori normalizzati')
        ax2.set_ylabel('Frequenza')
        
        plt.tight_layout()
        plt.show()

    def plot_valence_arousal_space(self):
        """
        Visualizza i gruppi e le clip nello spazio valence-arousal usando le feature EEG
        e mostra i valori di riferimento per confronto
        """

        # Stampa i nomi delle colonne di asimmetria disponibili
        asymmetry_cols = [col for col in self.df.columns if 'asymmetry_' in col]
        print("\nColonne di asimmetria disponibili:")
        print(asymmetry_cols)
        
        # Verifica quali colonne beta e alpha sono disponibili
        beta_cols = [col for col in self.df.columns if 'beta_' in col]
        print("\nColonne beta disponibili:")
        print(beta_cols)

        # Calcola prima i valori non normalizzati
        raw_valence = self.df.apply(lambda row: (row['asymmetry_alpha'] * 0.6 + 
                                               (row['beta_alpha_ratio_F4'] - row['beta_alpha_ratio_F3']) * 0.4), 
                                   axis=1)
        
        raw_arousal = self.df.apply(lambda row: (row['beta_AF3'] + row['beta_AF4'] + 
                                               row['beta_F3'] + row['beta_F4']) / 4, 
                                   axis=1)
        
        # Applica la normalizzazione robusta
        self.df['calculated_valence'] = self.robust_normalize(raw_valence)
        self.df['calculated_arousal'] = self.robust_normalize(raw_arousal)
        
        # Visualizza i dati prima e dopo la normalizzazione
        self.plot_distributions(raw_valence, self.df['calculated_valence'], 'Valence')
        self.plot_distributions(raw_arousal, self.df['calculated_arousal'], 'Arousal')

        # Crea due subplot affiancati
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Colori per i gruppi e marker per le clip
        group_colors = {'control': 'blue', 'resistant': 'red', 'responsive': 'green'}
        clip_markers = {'Clip_1': 'o', 'Clip_2': 's', 'Clip_3': '^'}

        # Raggruppa i dati per gruppo e clip, calcolando la media
        grouped_data = self.df.groupby(['group', 'clip'], as_index=False).agg({
            'calculated_valence': 'mean',
            'calculated_arousal': 'mean'
        })

        # Plot 1: Visualizzazione per gruppo
        for group in grouped_data['group'].unique():
            group_data = grouped_data[grouped_data['group'] == group]
            
            # Plot per ogni clip all'interno del gruppo
            for clip in grouped_data['clip'].unique():
                clip_data = group_data[group_data['clip'] == clip]
                if not clip_data.empty:
                    ax1.scatter(clip_data['calculated_valence'], 
                            clip_data['calculated_arousal'],
                            label=f'{group} - {clip}',
                            color=group_colors.get(group, 'gray'),
                            marker=clip_markers.get(clip, 'o'),
                            s=100,
                            alpha=0.6)

        # Plot 2: Visualizzazione per clip
        for clip in grouped_data['clip'].unique():
            clip_data = grouped_data[grouped_data['clip'] == clip]
            
            # Plot per ogni gruppo all'interno della clip
            for group in grouped_data['group'].unique():
                group_data = clip_data[clip_data['group'] == group]
                if not group_data.empty:
                    ax2.scatter(group_data['calculated_valence'], 
                            group_data['calculated_arousal'],
                            label=f'{clip} - {group}',
                            color=group_colors.get(group, 'gray'),
                            marker=clip_markers.get(clip, 'o'),
                            s=100,
                            alpha=0.6)

        # Aggiungi i valori di riferimento in entrambi i plot
        reference_values = {}
        for clip in self.df['clip'].unique():
            clip_data = self.df[self.df['clip'] == clip].iloc[0]
            reference_values[clip] = (clip_data['reference_valence'], 
                                    clip_data['reference_arousal'])

        # Plot dei valori di riferimento
        for clip, (val, aro) in reference_values.items():
            # Plot riferimenti in entrambi i grafici
            for ax in [ax1, ax2]:
                ax.scatter(val, aro, 
                        label=f'{clip} (reference)',
                        marker='*',
                        s=300,
                        color='black',
                        alpha=0.7)
                
                # Aggiungi annotazione con il nome della clip
                ax.annotate(clip, 
                        (val, aro),
                        xytext=(10, 10),  # Regola questa tupla per spostare il testo
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7,
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))  # Aggiungi una freccia

        # Configurazione dei plot
        for ax in [ax1, ax2]:
            ax.set_xlabel('Valence')
            ax.set_ylabel('Arousal')
            ax.grid(True)
            ax.set_xlim([0, 10])
            ax.set_ylim([0, 10])

        ax1.set_title('Distribuzione per Gruppi (con valori di riferimento)')
        ax2.set_title('Distribuzione per Clip (con valori di riferimento)')

        # Aggiungi legende
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Rimuovi duplicati nelle legende
        unique_labels1 = dict(zip(labels1, handles1))
        unique_labels2 = dict(zip(labels2, handles2))

        ax1.legend(unique_labels1.values(), unique_labels1.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(unique_labels2.values(), unique_labels2.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
    '''   
    def perform_statistical_analysis(self):
        """
        Esegue analisi statistiche tra i gruppi
        """
        results = {}
        
        # Per ogni feature
        for feature in self.feature_cols:
            # Raggruppa i dati per gruppo
            groups_data = [group_data[feature].values 
                         for name, group_data in self.df.groupby('group')]
            
            # ANOVA tra i gruppi
            f_stat, p_val = stats.f_oneway(*groups_data)
            
            if p_val < 0.05:  # Se significativo
                results[feature] = {
                    'f_statistic': f_stat,
                    'p_value': p_val
                }
        
        return pd.DataFrame(results).T
    
    def correlation_analysis(self):
        """
        Analizza correlazioni tra feature e valence/arousal
        """
        correlations = {}
        
        for feature in self.feature_cols:
            # Correlazione con valence
            val_corr, val_p = stats.pearsonr(self.df[feature], 
                                           self.df['reference_valence'])
            # Correlazione con arousal
            aro_corr, aro_p = stats.pearsonr(self.df[feature], 
                                           self.df['reference_arousal'])
            
            correlations[feature] = {
                'valence_correlation': val_corr,
                'valence_p_value': val_p,
                'arousal_correlation': aro_corr,
                'arousal_p_value': aro_p
            }
        
        return pd.DataFrame(correlations).T
    
    def plot_feature_distributions(self):
        """
        Visualizza distribuzioni delle feature più significative per gruppo
        """
        # Seleziona le feature più significative (top 5 dalle correlazioni)
        corr_analysis = self.correlation_analysis()
        top_features = corr_analysis.nlargest(5, 'valence_correlation').index
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(x='group', y=feature, data=self.df)
            plt.xticks(rotation=45)
            plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
    
    def train_prediction_model(self):
        """
        Addestra un modello per predire valence e arousal
        """
        # Prepara i dati
        X = self.df[self.feature_cols]
        y_valence = self.df['reference_valence']
        y_arousal = self.df['reference_arousal']
        
        # Split train/test
        X_train, X_test, y_val_train, y_val_test, y_aro_train, y_aro_test = train_test_split(
            X, y_valence, y_arousal, test_size=0.2, random_state=42)
        
        # Modello per valence
        val_model = RandomForestRegressor(n_estimators=100, random_state=42)
        val_model.fit(X_train, y_val_train)
        val_pred = val_model.predict(X_test)
        
        # Modello per arousal
        aro_model = RandomForestRegressor(n_estimators=100, random_state=42)
        aro_model.fit(X_train, y_aro_train)
        aro_pred = aro_model.predict(X_test)
        
        # Valuta i modelli
        results = {
            'valence_r2': r2_score(y_val_test, val_pred),
            'valence_rmse': np.sqrt(mean_squared_error(y_val_test, val_pred)),
            'arousal_r2': r2_score(y_aro_test, aro_pred),
            'arousal_rmse': np.sqrt(mean_squared_error(y_aro_test, aro_pred))
        }
        
        return results, val_model, aro_model
    '''
def main():
    # Inizializza l'analizzatore
    analyzer = EmotionAnalyzer('emotion_features.csv')
    
    # 1. Visualizza distribuzione nello spazio valence-arousal
    print("Generazione plot valence-arousal...")
    analyzer.plot_valence_arousal_space()
    
    '''
    # 2. Esegui analisi statistiche
    print("\nAnalisi statistica tra gruppi:")
    stats_results = analyzer.perform_statistical_analysis()
    print("\nFeature significativamente diverse tra i gruppi:")
    print(stats_results)
    
    # 3. Analisi correlazioni
    print("\nAnalisi correlazioni:")
    corr_results = analyzer.correlation_analysis()
    print("\nCorrelazioni più significative:")
    print(corr_results.sort_values('valence_correlation', ascending=False).head())
    
    # 4. Plot distribuzioni
    print("\nGenerazione plot distribuzioni...")
    analyzer.plot_feature_distributions()
    
    # 5. Machine Learning
    print("\nTraining modelli predittivi...")
    ml_results, val_model, aro_model = analyzer.train_prediction_model()
    print("\nRisultati Machine Learning:")
    for metric, value in ml_results.items():
        print(f"{metric}: {value:.3f}")
    '''
if __name__ == "__main__":
    main()