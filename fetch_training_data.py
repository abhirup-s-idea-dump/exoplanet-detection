import lightkurve as lk
import pandas as pd
import os
import time

def download_data(target_list, label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    for target in target_list:
        print(f"Searching for {target} ({label})...")
        try:
            # We fetch a short cadence or just one quarter/sector to keep downloads fast
            search_result = lk.search_lightcurve(target, author=['Kepler', 'SPOC'], limit=1)
            if len(search_result) > 0:
                print(f"  -> Downloading {target}...")
                lc = search_result.download()
                if lc is not None:
                    lc = lc.remove_nans()
                    # Save to CSV
                    df = pd.DataFrame({'Time': lc.time.value, 'Flux': lc.flux.value})
                    filepath = os.path.join(out_dir, f"{target}_{label}.csv")
                    df.to_csv(filepath, index=False)
                    print(f"  -> Saved {filepath}")
            else:
                print(f"  -> No data found for {target}.")
        except Exception as e:
            print(f"  -> Error fetching {target}: {e}")
        
        # Polite sleep to avoid hitting API rate limits too hard
        time.sleep(1)

if __name__ == "__main__":
    print("Starting data fetch from NASA MAST Archive...")
    
    # Curated small list for hackathon prototype (fast download)
    # You can add more targets here for a better model.
    positive_targets = [
        "Kepler-10", "Kepler-90", "TRAPPIST-1", "Kepler-11"
    ]
    
    # Random stars with no known transiting exoplanets (Negative examples)
    # Using random KIC IDs that are generally quiet
    negative_targets = [
        "KIC 3733346", "KIC 11446443", "KIC 8462852", "KIC 5513861"
    ]
    
    # Directories
    base_dir = "training_data"
    pos_dir = os.path.join(base_dir, "positive")
    neg_dir = os.path.join(base_dir, "negative")
    
    # Download
    print("--- Downloading Positive (Transit) Examples ---")
    download_data(positive_targets, "positive", pos_dir)
    
    print("\n--- Downloading Negative (No Transit) Examples ---")
    download_data(negative_targets, "negative", neg_dir)
    
    print("\nDownload complete! Data saved to 'training_data' folder.")
