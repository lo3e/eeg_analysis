import pandas as pd

# Carica il tuo DataFrame originale (in formato lungo)
df = pd.read_csv("emotion_features.csv")

# Lista di variabili di asimmetria (theta, alpha, beta) e rapporti alfa/beta
asimmetry_vars = [
    'asymmetry_theta', 'asymmetry_alpha', 'asymmetry_beta'
]

# Lista di variabili di rapporto alfa/beta per ogni canale
beta_alpha_ratio_vars = [
    'beta_alpha_ratio_F3', 'beta_alpha_ratio_F4', 'beta_alpha_ratio_F7', 'beta_alpha_ratio_F8'
]

# Combina entrambe le liste di variabili per l'analisi
all_vars = asimmetry_vars + beta_alpha_ratio_vars

# Pivot per trasformare le righe in colonne separate per ogni clip
df_wide = df.pivot_table(index=["subject", "group"], columns="clip", values=all_vars, aggfunc="first")

# Rinomina le colonne per aggiungere il suffisso per ogni clip
df_wide.columns = [f"{var}_{clip}" for var in all_vars for clip in df["clip"].unique()]

# Resetta l'indice se desideri una tabella più pulita
df_wide.reset_index(inplace=True)

# Salva il DataFrame trasformato in un nuovo file CSV
df_wide.to_csv("data_transformed.csv", index=False)

# Ora df_wide è in formato largo, con una colonna per ogni clip
print(df_wide.head())
