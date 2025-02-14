from avro.datafile import DataFileReader
from avro.io import DatumReader
import csv
import os
from datetime import datetime, timedelta

# Funzione per scrivere i dati in un CSV
def write_csv(filename, data, header):
    with open(os.path.join(output_dir, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def convert_to_iso(timestamp):
    return (datetime(1970, 1, 1) + timedelta(microseconds=timestamp)).isoformat()

# Directory di input e output
directory = 'C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/Empatica/participant_data'
output_dir = directory

# Dati accumulati
bvp_data = []
accelerometer_data = []
gyroscope_data = []
eda_data = []
temperature_data = []
tags_data = []
systolic_peaks_data = []
steps_data = []

# Dizionario per i dati uniti
merged_data = {}

# Elaborazione dei file AVRO
for filename in os.listdir(directory):
    if filename.endswith(".avro"):
        file_path = os.path.join(directory, filename)
        print(f"Processing {file_path}...")
        with DataFileReader(open(file_path, "rb"), DatumReader()) as reader:
            data = next(reader)
            # Accelerometer
            acc = data["rawData"]["accelerometer"]
            timestamp = [round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"])) for i in range(len(acc["x"]))]
            delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
            delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
            x_g = [val * delta_physical / delta_digital for val in acc["x"]]
            y_g = [val * delta_physical / delta_digital for val in acc["y"]]
            z_g = [val * delta_physical / delta_digital for val in acc["z"]]
            accelerometer_data.extend(zip(timestamp, x_g, y_g, z_g))

            # Gyroscope
            gyro = data["rawData"]["gyroscope"]
            timestamp = [round(gyro["timestampStart"] + i * (1e6 / gyro["samplingFrequency"])) for i in range(len(gyro["x"]))]
            gyroscope_data.extend(zip(timestamp, gyro["x"], gyro["y"], gyro["z"]))

            # EDA
            eda = data["rawData"]["eda"]
            timestamp = [round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"])) for i in range(len(eda["values"]))]
            eda_data.extend(zip(timestamp, eda["values"]))

            # Temperature
            tmp = data["rawData"]["temperature"]
            timestamp = [round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"])) for i in range(len(tmp["values"]))]
            temperature_data.extend(zip(timestamp, tmp["values"]))

            # Tags
            tags = data["rawData"]["tags"]
            tags_data.extend(tags["tagsTimeMicros"])  # Estrai solo i timestamp

            # BVP
            bvp = data["rawData"]["bvp"]
            timestamp = [round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"])) for i in range(len(bvp["values"]))]
            bvp_data.extend(zip(timestamp, bvp["values"]))

            sps = data["rawData"]["systolicPeaks"]
            systolic_peaks_data.extend([sp // 1000 for sp in sps["peaksTimeNanos"]])  # Converti da nanosecondi a microsecondi

            # Steps
            steps = data["rawData"]["steps"]
            timestamp = [round(steps["timestampStart"] + i * (1e6 / steps["samplingFrequency"])) for i in range(len(steps["values"]))]
            steps_data.extend(zip(timestamp, steps["values"]))

# Raccogli i dati nel dizionario merged_data
for ts, step in steps_data:
    merged_data[ts] = {'steps': step}

for ts, eda_val in eda_data:
    if ts in merged_data:
        merged_data[ts]['eda'] = eda_val

for ts, temp in temperature_data:
    if ts in merged_data:
        merged_data[ts]['temperature'] = temp

for ts, bvp_val in bvp_data:
    if ts in merged_data:
        merged_data[ts]['bvp'] = bvp_val

# Scrivi i CSV
write_csv('accelerometer.csv', [(ts, x, y, z, convert_to_iso(ts)) for ts, x, y, z in sorted(accelerometer_data)],
          ["unix_timestamp", "x", "y", "z", "timestamp_iso"])

write_csv('gyroscope.csv', [(ts, x, y, z, convert_to_iso(ts)) for ts, x, y, z in sorted(gyroscope_data)],
          ["unix_timestamp", "x", "y", "z", "timestamp_iso"])

write_csv('eda.csv', [(ts, eda_val, convert_to_iso(ts)) for ts, eda_val in sorted(eda_data)],
          ["unix_timestamp", "eda", "timestamp_iso"])

write_csv('temperature.csv', [(ts, temp, convert_to_iso(ts)) for ts, temp in sorted(temperature_data)],
          ["unix_timestamp", "temperature", "timestamp_iso"])

write_csv('tags.csv', [(tag, convert_to_iso(tag)) for tag in sorted(tags_data)],
          ["tags_timestamp", "timestamp_iso"])

write_csv('bvp.csv', [(ts, bvp_val, convert_to_iso(ts)) for ts, bvp_val in sorted(bvp_data)],
          ["unix_timestamp", "bvp", "timestamp_iso"])

write_csv('systolic_peaks.csv', [(sp, convert_to_iso(sp)) for sp in sorted(systolic_peaks_data)],
          ["systolic_peak_timestamp", "timestamp_iso"])

# Merge dei dati di EDA, temperatura e BVP
merged_data = []

# Creare un dizionario per i dati di temperatura e BVP
temp_dict = {ts: temp for ts, temp in sorted(temperature_data)}
bvp_dict = {ts: bvp_val for ts, bvp_val in sorted(bvp_data)}

# Combinare i dati
for ts, eda_val in sorted(eda_data):
    iso_timestamp = convert_to_iso(ts)
    temperature = temp_dict.get(ts) 
    bvp = bvp_dict.get(ts)
    merged_data.append((ts, eda_val, temperature, bvp, iso_timestamp))

# Scrivi il CSV merged
write_csv('merged.csv', merged_data, ["unix_timestamp", "eda", "temperature", "bvp", "timestamp_iso"])

print("Data export complete!")
