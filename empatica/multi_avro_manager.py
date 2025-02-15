import os
from avro.datafile import DataFileReader
from avro.io import DatumReader
import pandas as pd
from datetime import datetime, timedelta
import re

def convert_to_iso(timestamp):
    return (datetime(1970, 1, 1) + timedelta(microseconds=timestamp)).isoformat()

def process_avro_files(base_dir):
    # Dictionary to store DataFrames for each subject and signal type
    subject_data = {}
    
    # Regular expression to match subject IDs
    subject_pattern = r'((?:HC|TRS|SCZ)\d+)'
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        if 'raw_data' in root and 'v6' in root:
            avro_files = [f for f in files if f.endswith('.avro')]
            
            for avro_file in avro_files:
                # Extract subject ID
                match = re.search(subject_pattern, avro_file)
                if not match:
                    continue
                    
                subject_id = match.group(1)
                file_path = os.path.join(root, avro_file)
                
                print(f"Processing {file_path} for subject {subject_id}...")
                
                try:
                    # Initialize storage for this subject if not exists
                    if subject_id not in subject_data:
                        subject_data[subject_id] = {
                            'accelerometer': [],
                            'gyroscope': [],
                            'eda': [],
                            'temperature': [],
                            'bvp': [],
                            'steps': [],
                            'tags': [],
                            'systolic_peaks': []
                        }
                    
                    # Read AVRO file
                    with DataFileReader(open(file_path, "rb"), DatumReader()) as reader:
                        data = next(reader)
                        raw_data = data["rawData"]
                        
                        # Process each signal type
                        # Accelerometer
                        acc = raw_data["accelerometer"]
                        acc_timestamps = [round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"])) 
                                       for i in range(len(acc["x"]))]
                        delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
                        delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
                        acc_data = pd.DataFrame({
                            'timestamp': acc_timestamps,
                            'x': [val * delta_physical / delta_digital for val in acc["x"]],
                            'y': [val * delta_physical / delta_digital for val in acc["y"]],
                            'z': [val * delta_physical / delta_digital for val in acc["z"]]
                        })
                        subject_data[subject_id]['accelerometer'].append(acc_data)
                        
                        # Similar processing for other signals...
                        # EDA
                        eda = raw_data["eda"]
                        eda_timestamps = [round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"])) 
                                        for i in range(len(eda["values"]))]
                        eda_data = pd.DataFrame({
                            'timestamp': eda_timestamps,
                            'eda': eda["values"]
                        })
                        subject_data[subject_id]['eda'].append(eda_data)
                        
                        # Temperature
                        tmp = raw_data["temperature"]
                        tmp_timestamps = [round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"])) 
                                        for i in range(len(tmp["values"]))]
                        temp_data = pd.DataFrame({
                            'timestamp': tmp_timestamps,
                            'temperature': tmp["values"]
                        })
                        subject_data[subject_id]['temperature'].append(temp_data)
                        
                        # BVP
                        bvp = raw_data["bvp"]
                        bvp_timestamps = [round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"])) 
                                        for i in range(len(bvp["values"]))]
                        bvp_data = pd.DataFrame({
                            'timestamp': bvp_timestamps,
                            'bvp': bvp["values"]
                        })
                        subject_data[subject_id]['bvp'].append(bvp_data)
                        
                except Exception as e:
                    print(f"Error processing file {avro_file}: {str(e)}")
                    continue
    
    return subject_data

def save_processed_data(subject_data, output_dir):
    """Save processed data for each subject and signal type."""
    os.makedirs(output_dir, exist_ok=True)
    
    for subject_id, signal_data in subject_data.items():
        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Process and save each signal type
        for signal_type, data_list in signal_data.items():
            if not data_list:  # Skip empty data
                continue
                
            # Concatenate all DataFrames for this signal type
            combined_df = pd.concat(data_list, ignore_index=True)
            
            # Sort by timestamp and remove duplicates
            combined_df = combined_df.sort_values('timestamp').drop_duplicates()
            
            # Add ISO timestamp
            combined_df['timestamp_iso'] = combined_df['timestamp'].apply(convert_to_iso)
            
            # Save to CSV
            output_file = os.path.join(subject_dir, f'{signal_type}.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {output_file}")

def main():
    base_dir = "C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/Empatica/participant_data"  # Replace with your actual base directory
    output_dir = "C:/Users/lo3e/Documents/Università/PhD/Progetti/SPECTRA/Empatica/output"    # Replace with desired output directory
    
    # Process all AVRO files
    subject_data = process_avro_files(base_dir)
    
    # Save processed data
    save_processed_data(subject_data, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()