import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks, butter, filtfilt
import os
from datetime import datetime, timedelta

# Percorsi dei file
file_path = "C:/Users/Enzo/Desktop/Tesi/DatiEmpatica/1/1/participant_data/2024-12-18/HC01-3YK32132K7/raw_data/v6/eda.csv"
output_dir = "C:/Users/Enzo/Desktop/Tesi/DatiEmpatica/1/1/participant_data/2024-12-18/HC01-3YK32132K7/raw_data/v6"

eda_data = pd.read_csv(file_path)

if 'unix_timestamp' not in eda_data.columns or 'eda' not in eda_data.columns:
    raise ValueError("Il file CSV deve contenere le colonne 'unix_timestamp' e 'eda'.")

eda_data['unix_timestamp'] = pd.to_numeric(eda_data['unix_timestamp'], errors='coerce')  
eda_data = eda_data.dropna(subset=['unix_timestamp', 'eda'])  
eda_data['unix_timestamp'] = pd.to_datetime(eda_data['unix_timestamp'], unit='us')

eda_values = eda_data['eda'].astype(float)  # Assicura che i dati siano numerici

sampling_freq = 4  

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

cutoff_freq = 0.5  
eda_filtered = butter_lowpass_filter(eda_values, cutoff_freq, sampling_freq)

scr_peaks, _ = find_peaks(eda_filtered, prominence=0.05)

level_markers = [
    {'time': 1734538984027, 'type': 'level start', 'name': 'L:1_1'},
    {'time': 1734539041625, 'type': 'level end', 'name': 'L:1_1_Win'},
    {'time': 1734539057620, 'type': 'level start', 'name': 'L:1_2'},
    {'time': 1734539113663, 'type': 'level end', 'name': 'L:1_2_Win'},
    {'time': 1734539129699, 'type': 'level start', 'name': 'L:2_1'},
    {'time': 1734539179063, 'type': 'level end', 'name': 'L:2_1_Win'},
    {'time': 1734539194943, 'type': 'level start', 'name': 'L:2_2'},
    {'time': 1734539320465, 'type': 'level end', 'name': 'L:2_2_Lose'},
    {'time': 1734539341499, 'type': 'level start', 'name': 'L:3_1'},
    {'time': 1734539398085, 'type': 'level end', 'name': 'L:3_1_Win'}
]

clip_markers = [
    {'time': 1734538894.496149, 'type': 'clip start', 'name': 'Baseline_Start'},
    {'time': 1734538914.621888, 'type': 'clip end', 'name': 'Baseline_End'},
    {'time': 1734539423.762076, 'type': 'clip start', 'name': 'Clip 2 Start'},
    {'time': 1734539540.194645, 'type': 'clip end', 'name': 'Clip 2 End'},
    {'time': 1734539619.354581, 'type': 'clip start', 'name': 'Clip 3 Start'},
    {'time': 1734539734.924340, 'type': 'clip end', 'name': 'Clip 3 End'},
    {'time': 1734539785.943380, 'type': 'clip start', 'name': 'Clip 4 Start'},
    {'time': 1734539918.539006, 'type': 'clip end', 'name': 'Clip 4 End'}
]

for marker in level_markers + clip_markers:
    if isinstance(marker['time'], float):  # Se è un timestamp con frazioni di secondo
        marker['time'] = pd.to_datetime(marker['time'], unit='s')
    else:  # Se è un timestamp in millisecondi
        marker['time'] = pd.to_datetime(marker['time'], unit='ms')

eda_trace = go.Scatter(
    x=eda_data['unix_timestamp'],
    y=eda_filtered,
    mode='lines',
    name='EDA Signal',
    line=dict(color='blue')
)

scr_trace = go.Scatter(
    x=eda_data['unix_timestamp'].iloc[scr_peaks],
    y=eda_filtered[scr_peaks],
    mode='markers',
    name='SCR Peaks',
    marker=dict(color='red', size=8, symbol='x')
)

manual_marker_traces = []
highlighted_area_traces = [] 
for marker in level_markers + clip_markers:
    marker_trace = go.Scatter(
        x=[marker['time'], marker['time']],  # Linea verticale
        y=[min(eda_filtered), max(eda_filtered)], 
        mode='lines',
        name=f"{marker['type']} - {marker.get('name', '')}",
        line=dict(color='green' if 'clip' in marker['type'] else 'blue', width=2, dash='dash'),
    )
    manual_marker_traces.append(marker_trace)

    if marker['type'] == 'clip start':
        corresponding_end_marker = next(
            (m for m in clip_markers if m['type'] == 'clip end' and m['name'] == f"{marker['name'].split()[0]} End"),
            None
        )
        if corresponding_end_marker:
            highlighted_area_trace = go.Scatter(
                x=[marker['time'], corresponding_end_marker['time'], corresponding_end_marker['time'], marker['time']],
                y=[min(eda_filtered), min(eda_filtered), max(eda_filtered), max(eda_filtered)],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='green', width=0),
                name=f"Highlight {marker['name']}",
                showlegend=False
            )
            highlighted_area_traces.append(highlighted_area_trace)

    elif marker['type'] == 'level start':
        # Trova il marker end corrispondente
        try:
            corresponding_end_marker = next(
                m for m in level_markers if m['type'] == 'level end' and m['name'] == f"{marker['name']} Win"
            )
            highlighted_area_trace = go.Scatter(
                x=[marker['time'], corresponding_end_marker['time'], corresponding_end_marker['time'], marker['time']],
                y=[min(eda_filtered), min(eda_filtered), max(eda_filtered), max(eda_filtered)],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='red', width=0),
                name=f"Highlight {marker['name']}",
                showlegend=False
            )
            highlighted_area_traces.append(highlighted_area_trace)
        except StopIteration:
            print(f"Warning: No 'level end' found for {marker['name']}. Skipping highlighting for this level.")
            continue 

fig = go.Figure(data=[eda_trace, scr_trace] + manual_marker_traces + highlighted_area_traces)

fig.update_layout(
    title='Electrodermal Activity (EDA) Signal with SCR Peaks and Event Markers',
    xaxis_title='Time',
    yaxis_title='EDA Signal (µS)',
    xaxis=dict(showgrid=True, type='date'),
    yaxis=dict(showgrid=True),
    plot_bgcolor='white', 
)
output_csv_file = os.path.join(output_dir, 'eda_filtered.csv')
eda_data['eda_filtered'] = eda_filtered
eda_data.to_csv(output_csv_file, index=False)
print(f"Dati filtrati salvati come {output_csv_file}")
output_file = os.path.join(output_dir, 'eda_plot.html')
fig.write_html(output_file)

print(f"Plot salvato come {output_file}")
