import os
import struct
import csv

# --- CONFIGURATION ---
SERATO_FOLDER = '/Users/rohan/Music/_Serato_/Subcrates'
OUTPUT_FILE = 'serato_training_data.csv'

def parse_serato_crate(file_path):
    """Manually parses the binary .crate file to extract track paths."""
    tracks = []
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            
        pos = 0
        while True:
            pos = content.find(b'ptrk', pos)
            if pos == -1: break
            
            size_pos = pos + 4
            size = struct.unpack('>I', content[size_pos:size_pos+4])[0]
            
            path_data = content[size_pos+4 : size_pos+4+size]
            path = path_data.decode('utf-16-be').strip('\x00')
            tracks.append(path)
            
            pos += 8 + size 
            
    except Exception as e:
        print(f"⚠️ Could not read {os.path.basename(file_path)}: {e}")
    
    return tracks

def main():
    dataset = []

    if not os.path.exists(SERATO_FOLDER):
        print(f"❌ Folder not found: {SERATO_FOLDER}")
        return

    print("🔍 Scanning your Serato Crates...")
    crate_files = [f for f in os.listdir(SERATO_FOLDER) if f.endswith('.crate')]

    for filename in crate_files:
        path = os.path.join(SERATO_FOLDER, filename)
        crate_name = filename.replace('.crate', '').replace('%%%', ' > ')
        
        tracks = parse_serato_crate(path)
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
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'path', 'crate'])
            writer.writeheader()
            writer.writerows(dataset)
        print(f"\n✅ SUCCESS! {len(dataset)} tracks saved with titles to {OUTPUT_FILE}")
    else:
        print("❌ No tracks were found.")

if __name__ == "__main__":
    main()