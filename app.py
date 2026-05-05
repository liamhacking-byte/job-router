import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
from urllib.parse import quote
import numpy as np
st.title("Smart Job Router (Pro)")
uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)
# --- Geocode ---
@st.cache_data
def geocode(address):
   url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
   try:
       r = requests.get(url).json()
       return float(r[0]['lat']), float(r[0]['lon'])
   except:
       return None, None
# --- Maps link ---
def maps_link(addresses):
   return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in addresses])
if uploaded_file:
   df = pd.read_excel(uploaded_file)
   # --- Auto detect address column ---
   possible_cols = ['Address', 'address', 'Full Address', 'Job Address']
   address_col = next((col for col in possible_cols if col in df.columns), None)
   if not address_col:
       st.error("No address column found")
   elif 'Slot' not in df.columns:
       st.error("Missing 'Slot' column (AM/PM)")
   else:
       st.write("Geocoding addresses...")
       df[['lat','lon']] = df[address_col].apply(lambda x: pd.Series(geocode(x)))
       df = df.dropna()
       # --- Cluster jobs geographically ---
       kmeans = KMeans(n_clusters=engineers, random_state=0)
       df['Engineer'] = kmeans.fit_predict(df[['lat','lon']])
       # --- Balance workload ---
       max_jobs = int(np.ceil(len(df) / engineers))
       for i in range(engineers):
           over = df[df['Engineer'] == i]
           if len(over) > max_jobs:
               extra = over.iloc[max_jobs:]
               for idx, row in extra.iterrows():
                   distances = ((df['lat'] - row['lat'])**2 + (df['lon'] - row['lon'])**2)
                   nearest_cluster = distances.idxmin()
df.at[idx, 'Engineer'] = df.loc[nearest_cluster, 'Engineer']
       st.success("Routing complete!")
       # --- Output per engineer ---
       for i in range(engineers):
           st.subheader(f"Engineer {i+1}")
           eng_jobs = df[df['Engineer'] == i]
           # --- AM / PM split ---
           am = eng_jobs[eng_jobs['Slot'] == 'AM'].sort_values(by='lat')
           pm = eng_jobs[eng_jobs['Slot'] == 'PM'].sort_values(by='lat')
           route = pd.concat([am, pm])
           st.write(route[[address_col,'Slot']])
           link = maps_link(route[address_col].tolist())
           st.markdown(f"[Open Route in Google Maps]({link})")
