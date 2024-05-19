import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Read the data from CSV
data = pd.read_csv(r"C:\Users\hruth\Downloads\spotify.csv")

# Drop unnecessary columns
df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])

# Select numerical columns for normalization
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = df.select_dtypes(include=datatypes)

# Normalize the data
scaler = MinMaxScaler()
norm_data = scaler.fit_transform(normarization)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10)
cluster_labels = kmeans.fit_predict(norm_data)

# Add cluster labels to the original dataframe
data['cluster'] = cluster_labels

# Define the song for recommendation
song_name = "Track1"

# Check if the song exists in the dataset
if data['name'].str.lower().eq(song_name.lower()).any():
    song = data[data['name'].str.lower() == song_name.lower()].head(1)

    # Calculate distances
    distance = []
    rec = data[data['name'].str.lower() != song_name.lower()]
    rec = rec.copy()  # Create a copy of the DataFrame
    for _, row in tqdm(rec.iterrows(), total=len(rec)):
        d = np.linalg.norm(row[normarization.columns].values - song[normarization.columns].values)
        distance.append(d)

    # Add distances to the dataframe using .loc to avoid the warning
    rec['distance'] = distance

    # Sort by distance and select top 5 similar songs
    rec_sorted = rec.sort_values('distance').head(5)

    # Print the recommended songs
    print("Recommended songs:")
    for idx, song in rec_sorted.iterrows():
        print(f"{song['name']} by {song['artists']}")

else:
    print("Song not found in the dataset.")
