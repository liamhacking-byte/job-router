df[['lat','lon']] = df[address_col].apply(lambda x: pd.Series(geocode(x)))
df = df.dropna(subset=['lat', 'lon'])
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
