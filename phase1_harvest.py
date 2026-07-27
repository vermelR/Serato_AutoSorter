import os
import csv
from pathlib import Path

from serato_crate import parse_serato_crate as _parse_serato_crate
from serato_ai.settings import load_settings

# --- CONFIGURATION ---
SERATO_FOLDER = str(Path(load_settings().default_serato_root) / "Subcrates")
OUTPUT_FILE = 'serato_training_data.csv'

def parse_serato_crate(file_path):
    """Safely parse binary crate path records; malformed files raise clearly."""
    return [entry.path for entry in _parse_serato_crate(file_path)]

def main(serato_folder: str = SERATO_FOLDER, output_file: str = OUTPUT_FILE):
    dataset = []

    if not os.path.exists(serato_folder):
        print(f"❌ Folder not found: {serato_folder}")
        return

    print("🔍 Scanning your Serato Crates...")
    crate_files = [f for f in os.listdir(serato_folder) if f.endswith('.crate')]

    for filename in crate_files:
        path = os.path.join(serato_folder, filename)
        crate_name = filename.replace('.crate', '').replace('%%%', ' > ')
        
        try:
            tracks = parse_serato_crate(path)
        except ValueError as e:
            print(f"⚠️ Could not read {os.path.basename(path)}: {e}")
            continue
        print(f"📦 {crate_name}: {len(tracks)} tracks")
        
        for t in tracks:
            # --- NEW LOGIC: Extract Track Title from Path ---
            # 1. Get the filename (e.g., "Daft Punk - One More Time.mp3")
            filename_with_ext = os.path.basename(t)
            # 2. Remove the extension (e.g., "Daft Punk - One More Time")
            track_title = os.path.splitext(filename_with_ext)[0]
            
            dataset.append({
                'title': track_title, 
                'path': t, 
                'crate': crate_name
            })

    if dataset:
        # Added 'title' to the fieldnames
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'path', 'crate'])
            writer.writeheader()
            writer.writerows(dataset)
        print(f"\n✅ SUCCESS! {len(dataset)} tracks saved with titles to {output_file}")
    else:
        print("❌ No tracks were found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Harvest a Serato Subcrates folder for training data")
    parser.add_argument("--serato-subcrates", default=SERATO_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.serato_subcrates, args.output)
