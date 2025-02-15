import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class ExtractionReport:
    def __init__(self):
        self.results = []
        
    def add_result(self, subject_id, signal_type, clip_num, start_time, end_time, 
                   segment_length, status, notes=""):
        self.results.append({
            'subject_id': subject_id,
            'signal_type': signal_type,
            'clip_num': clip_num,
            'start_time': start_time,
            'end_time': end_time,
            'segment_length': segment_length,
            'status': status,
            'notes': notes
        })
    
    def save_report(self, output_path):
        report_df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_path, f"extraction_report_{timestamp}.csv")
        report_df.to_csv(filename, index=False)
        return filename

def get_clip_timestamps(marker_file, report, subject_id):
    """Extract start/end timestamps for each clip from marker file."""
    markers_df = pd.read_csv(marker_file)
    
    clip_times = {}
    for clip_num in range(1, 4):  # For clips 1-3
        start_type = 100 + clip_num
        end_type = 200 + clip_num
        
        # Get timestamps
        start_times = markers_df[markers_df['marker_value'] == start_type]['timestamp']
        end_times = markers_df[markers_df['marker_value'] == end_type]['timestamp']
        
        if start_times.empty or end_times.empty:
            note = f"Missing marker: start={start_type}, end={end_type}"
            report.add_result(subject_id, "markers", clip_num, None, None, 0, 
                            "ERROR", note)
            continue
            
        # Convert from seconds to microseconds
        start_time = int(start_times.iloc[0] * 1_000_000)
        end_time = int(end_times.iloc[0] * 1_000_000)
        
        clip_times[f'Clip_{clip_num}'] = (start_time, end_time)
        report.add_result(subject_id, "markers", clip_num, start_time, end_time, 
                         None, "OK", "Markers found")
    
    return clip_times

def extract_signal_segments(signal_file, clip_times, report, subject_id, signal_type):
    """Extract signal segments for each clip period with additional debugging."""
    try:
        signal_df = pd.read_csv(signal_file)
        
        # Add debugging info about signal timestamps
        min_ts = signal_df['timestamp'].min()
        max_ts = signal_df['timestamp'].max()
        signal_range = f"{min_ts} to {max_ts}"
        
        print(f"\nProcessing {subject_id} - {signal_type}")
        print(f"Signal range: {signal_range}")
        
        all_segments = []
        
        for clip_name, (start_time, end_time) in clip_times.items():
            clip_num = int(clip_name.split('_')[1])
            
            # Print debugging info for each clip
            print(f"\nClip {clip_num}:")
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            print(f"Start time in range: {start_time >= min_ts and start_time <= max_ts}")
            print(f"End time in range: {end_time >= min_ts and end_time <= max_ts}")
            
            # Count points just before and after the clip period
            buffer = 1_000_000  # 1 second buffer
            points_before = len(signal_df[
                (signal_df['timestamp'] >= (start_time - buffer)) & 
                (signal_df['timestamp'] < start_time)
            ])
            points_after = len(signal_df[
                (signal_df['timestamp'] > end_time) & 
                (signal_df['timestamp'] <= (end_time + buffer))
            ])
            
            # Extract segment
            segment = signal_df[
                (signal_df['timestamp'] >= start_time) & 
                (signal_df['timestamp'] <= end_time)
            ].copy()
            
            print(f"Points found: {len(segment)}")
            print(f"Points in 1s before clip: {points_before}")
            print(f"Points in 1s after clip: {points_after}")
            
            if len(segment) == 0:
                # Get closest points to start and end time
                closest_before = signal_df[signal_df['timestamp'] < start_time]['timestamp'].max()
                closest_after = signal_df[signal_df['timestamp'] > end_time]['timestamp'].min()
                gap_info = f" | Closest points: before={closest_before}, after={closest_after}"
            else:
                gap_info = ""
            
            status = "OK" if len(segment) > 0 else "WARNING"
            notes = f"Signal range: {signal_range}{gap_info}"
            if len(segment) == 0:
                notes += " | No data found in time range"
            
            report.add_result(subject_id, signal_type, clip_num, start_time, end_time,
                            len(segment), status, notes)
            
            if len(segment) > 0:
                segment['clip'] = clip_name
                all_segments.append(segment)
        
    except Exception as e:
        report.add_result(subject_id, signal_type, None, None, None, 0,
                         "ERROR", f"Failed to read signal file: {str(e)}")
        return pd.DataFrame()
    
    if not all_segments:
        return pd.DataFrame(columns=signal_df.columns.tolist() + ['clip'])
    
    return pd.concat(all_segments, ignore_index=True)

def process_subject(subject_id, base_path, markers_path, output_path, report):
    """Process all signals for a single subject."""
    try:
        # Get clip timestamps from marker file
        marker_file = os.path.join(markers_path, f"{subject_id}_markers.csv")
        
        if not os.path.exists(marker_file):
            report.add_result(subject_id, "all", None, None, None, 0, 
                            "ERROR", "Marker file not found")
            return
        
        clip_times = get_clip_timestamps(marker_file, report, subject_id)
        
        # Process each signal type
        signal_types = ['bvp', 'eda', 'temperature']
        
        for signal_type in signal_types:
            signal_file = os.path.join(base_path, subject_id, f"{signal_type}.csv")
            
            if not os.path.exists(signal_file):
                report.add_result(subject_id, signal_type, None, None, None, 0, 
                                "ERROR", "Signal file not found")
                continue
                
            segments_df = extract_signal_segments(signal_file, clip_times, report, 
                                                subject_id, signal_type)
            
            # Create output directory and save
            output_dir = os.path.join(output_path, subject_id)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{signal_type}_segments.csv")
            segments_df.to_csv(output_file, index=False)
            
    except Exception as e:
        report.add_result(subject_id, "all", None, None, None, 0, 
                         "ERROR", f"Processing failed: {str(e)}")
def analyze_timing(marker_file, signal_file, subject_id):
    """Analyze timing relationships between markers and signal data."""
    # Load data
    markers_df = pd.read_csv(marker_file)
    signal_df = pd.read_csv(signal_file)
    
    # Signal range
    signal_start = signal_df['timestamp'].min()
    signal_end = signal_df['timestamp'].max()
    signal_duration = (signal_end - signal_start) / 1_000_000  # Convert to seconds
    
    print(f"\nAnalysis for {subject_id}")
    print(f"Signal duration: {signal_duration:.2f} seconds")
    print(f"Signal range: {signal_start} to {signal_end}")
    
    # Analyze each clip
    for clip_num in range(1, 4):
        start_marker = 100 + clip_num
        end_marker = 200 + clip_num
        
        start_time = markers_df[markers_df['marker_value'] == start_marker]['timestamp'].iloc[0] * 1_000_000
        end_time = markers_df[markers_df['marker_value'] == end_marker]['timestamp'].iloc[0] * 1_000_000
        
        clip_duration = (end_time - start_time) / 1_000_000  # Convert to seconds
        
        # Check if clip timestamps fall within signal range
        in_range = (start_time >= signal_start and end_time <= signal_end)
        
        # Find closest signal points
        points_in_clip = len(signal_df[
            (signal_df['timestamp'] >= start_time) & 
            (signal_df['timestamp'] <= end_time)
        ])
        
        print(f"\nClip {clip_num}:")
        print(f"Duration: {clip_duration:.2f} seconds")
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        print(f"Within signal range: {in_range}")
        print(f"Points in clip: {points_in_clip}")
        
        if not in_range:
            # Calculate time difference from signal range
            start_diff = (start_time - signal_start) / 1_000_000
            end_diff = (end_time - signal_end) / 1_000_000
            print(f"Start time differs from signal start by: {start_diff:.2f} seconds")
            print(f"End time differs from signal end by: {end_diff:.2f} seconds")


def main():
    # Define paths
    base_path = r"C:\\Users\\lo3e\Documents\\Università\\PhD\\Progetti\\SPECTRA\\Empatica\\output"
    markers_path = r"C:\\Users\\lo3e\\Documents\\Università\\PhD\\Progetti\\SPECTRA\\Empatica\\merged"
    output_path = r"C:\\Users\\lo3e\\Documents\\Università\\PhD\\Progetti\\SPECTRA\\Empatica\\processed"
    
    # Initialize report
    report = ExtractionReport()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get subject directories
    subject_dirs = [d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d)) and 
                   (d.startswith(('HC', 'SCZ', 'TRS')))]
    
    # Process each subject
    for subject_id in subject_dirs:
        process_subject(subject_id, base_path, markers_path, output_path, report)
    
    base_path_empatica = r"C:\\Users\\lo3e\Documents\\Università\\PhD\\Progetti\\SPECTRA\\Empatica"
    for subject_id in ["HC02", "HC03", "HC05", "HC06", "SCZ09", "SCZ11"]:
        marker_file = f"{base_path_empatica}/merged/{subject_id}_markers.csv"
        signal_file = f"{base_path}/{subject_id}/bvp.csv"
        analyze_timing(marker_file, signal_file, subject_id)

    # Save report
    report_file = report.save_report(output_path)
    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    main()