import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from scipy.stats import f_oneway
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

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
        #print("Colonne disponibili nel DataFrame:")
        #print(self.df.columns.tolist())
        #print("\nFeature columns:")
        #print(self.feature_cols)

    def min_max_normalize(self, values, new_min=0, new_max=10):
        """
        Normalizza i valori in un intervallo [new_min, new_max].
        """
        min_val = np.min(values)
        max_val = np.max(values)
        normalized_values = (values - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

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
        #print("\nColonne di asimmetria disponibili:")
        #print(asymmetry_cols)
        
        # Verifica quali colonne beta e alpha sono disponibili
        beta_cols = [col for col in self.df.columns if 'beta_' in col]
        #print("\nColonne beta disponibili:")
        #print(beta_cols)

        # Calcola prima i valori non normalizzati
        raw_valence = self.df.apply(lambda row: (row['asymmetry_alpha'] * 0.6 + 
                                               (row['beta_alpha_ratio_F4'] - row['beta_alpha_ratio_F3']) * 0.4), 
                                   axis=1)
        
        raw_arousal = self.df.apply(lambda row: (row['beta_AF3'] + row['beta_AF4'] + row['beta_F3'] + row['beta_F4']) / 4, axis=1)
        '''
        # Visualizza la distribuzione originale di valence e arousal
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(raw_valence, kde=True, color='skyblue')
        plt.title('Distribuzione originale - Valence')

        plt.subplot(1, 2, 2)
        sns.histplot(raw_arousal, kde=True, color='lightgreen')
        plt.title('Distribuzione originale - Arousal')
        plt.show()
        '''
        #normalizzazione di valence
        self.df['calculated_valence'] = self.min_max_normalize(raw_valence)

        #normalizzazione di arousal
        # Trasformazione di Box-Cox con lambda personalizzato
        transformed_arousal, _ = boxcox(raw_arousal + 1)  # Aggiungi 1 per evitare valori negativi

        # Calcola i quartili e l'IQR
        Q1 = np.percentile(transformed_arousal, 25)
        Q3 = np.percentile(transformed_arousal, 75)
        IQR = Q3 - Q1

        # Definisci i limiti per gli outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtra i valori per rimuovere gli outlier
        filtered_arousal = transformed_arousal[(transformed_arousal >= lower_bound) & 
        (transformed_arousal <= upper_bound)]

        # Filtra il DataFrame originale per rimuovere le righe corrispondenti agli outlier
        self.df = self.df[(transformed_arousal >= lower_bound) & (transformed_arousal <= upper_bound)]

        # Normalizzazione Z-score
        self.df['calculated_arousal'] = self.min_max_normalize(filtered_arousal)
        '''
        # Visualizza le distribuzioni dopo la normalizzazione
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['calculated_valence'], kde=True, color='skyblue')
        plt.title('Distribuzione normalizzata - Valence')

        plt.subplot(1, 2, 2)
        sns.histplot(self.df['calculated_arousal'], kde=True, color='lightgreen')
        plt.title('Distribuzione normalizzata - Arousal')
        plt.show()
        '''
        # Crea due subplot affiancati
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Colori per i gruppi e marker per le clip
        group_colors = {'control': 'blue', 'resistant': 'red', 'responsive': 'green'}
        clip_markers = {'Clip_1': 'o', 'Clip_2': 's', 'Clip_3': '^'}

        # Raggruppa i dati per gruppo e clip, calcolando media e deviazione standard
        grouped_data = self.df.groupby(['group', 'clip'], as_index=False).agg({
            'calculated_valence': ['mean', 'std'],
            'calculated_arousal': ['mean', 'std']
        })
        # Rinomina le colonne per facilitare l'accesso
        grouped_data.columns = ['group', 'clip', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']

        # Aggiungi jitter per evitare sovrapposizioni
        jitter = 0.5  # Aumenta il jitter

        # Plot 1: Visualizzazione per gruppo con barre di errore
        for group in grouped_data['group'].unique():
            group_data = grouped_data[grouped_data['group'] == group]
            
            # Plot per ogni clip all'interno del gruppo
            for clip in grouped_data['clip'].unique():
                clip_data = group_data[group_data['clip'] == clip]
                if not clip_data.empty:
                    ax1.errorbar(
                        clip_data['valence_mean'] + np.random.uniform(-jitter, jitter), 
                        clip_data['arousal_mean'] + np.random.uniform(-jitter, jitter),
                        xerr=clip_data['valence_std'],
                        yerr=clip_data['arousal_std'],
                        label=f'{group} - {clip}',
                        color=group_colors.get(group, 'gray'),
                        marker=clip_markers.get(clip, 'o'),
                        markersize=10,
                        capsize=5,
                        alpha=0.8
                    )

        # Plot 2: Grafico a densità per gruppi
        sns.kdeplot(
            x='calculated_valence', 
            y='calculated_arousal', 
            hue='group', 
            data=self.df, 
            ax=ax2,
            palette=group_colors,
            alpha=0.6
        )
        
        # Aggiungi ellissi di confidenza per ogni gruppo
        for group in self.df['group'].unique():
            group_data = self.df[self.df['group'] == group]
            if len(group_data) > 1:  # Richiede almeno 2 punti per calcolare la covarianza
                gmm = GaussianMixture(n_components=1)
                gmm.fit(group_data[['calculated_valence', 'calculated_arousal']])
                
                # Estrai media e covarianza
                mean = gmm.means_[0]
                cov = gmm.covariances_[0]
                
                # Calcola autovalori e autovettori per disegnare l'ellisse
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * np.sqrt(eigenvalues)
                
                # Disegna l'ellisse
                ellipse = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor=group_colors.get(group, 'gray'),
                    facecolor='none',
                    linewidth=2,
                    alpha=0.8
                )
                ax2.add_patch(ellipse)
        
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
            ax.set_xlim([0, 10])  # Scala completa
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

        # Analisi statistica: ANOVA per confrontare i gruppi
        groups_data = [self.df[self.df['group'] == group]['calculated_valence'] for group in self.df['group'].unique()]
        f_stat, p_val = f_oneway(*groups_data)
        print(f"ANOVA per Valence: F-statistic = {f_stat:.2f}, p-value = {p_val:.4f}")

        groups_data = [self.df[self.df['group'] == group]['calculated_arousal'] for group in self.df['group'].unique()]
        f_stat, p_val = f_oneway(*groups_data)
        print(f"ANOVA per Arousal: F-statistic = {f_stat:.2f}, p-value = {p_val:.4f}")

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