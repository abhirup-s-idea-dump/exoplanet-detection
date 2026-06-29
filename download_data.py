import lightkurve as lk
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

print("Searching for Kepler-10 data (a known exoplanet host)...")
# Kepler-10 is a well-known star with transiting exoplanets
search_result = lk.search_lightcurve('Kepler-10', author='Kepler', quarter=3)

if len(search_result) > 0:
    print("Downloading Kepler-10 light curve...")
    lc = search_result.download()
    lc = lc.remove_nans()
    
    # Save as CSV so it's easy to upload to the Streamlit app
    df = pd.DataFrame({'Time': lc.time.value, 'Flux': lc.flux.value})
    df.to_csv('data/kepler_10_quarter3.csv', index=False)
    print("Saved -> data/kepler_10_quarter3.csv")
else:
    print("Could not find data for Kepler-10")

print("\nSearching for Kepler-90 data (multi-planet system)...")
search_result = lk.search_lightcurve('Kepler-90', author='Kepler', quarter=1)

if len(search_result) > 0:
    print("Downloading Kepler-90 light curve...")
    lc = search_result.download()
    lc = lc.remove_nans()
    
    df = pd.DataFrame({'Time': lc.time.value, 'Flux': lc.flux.value})
    df.to_csv('data/kepler_90_quarter1.csv', index=False)
    print("Saved -> data/kepler_90_quarter1.csv")
else:
    print("Could not find data for Kepler-90")

print("\nDone! You can now use these CSV files in the Streamlit app.")
