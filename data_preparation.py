import pandas as pd
df = pd.read_csv('stars_data.csv')
print(df['Spectral Class'].unique())
df['Spectral Class'] = df['Spectral Class'].replace(['O', 'B', 'A', 'F', 'G', 'K', 'M'], [x for x in range(6, -1, -1)])
print(df['Spectral Class'].unique())
df = df.drop(columns='Star color')
print(df.head())
df = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Spectral Class', 'Star type']]
print(df)
df.to_csv('data.csv', index=False)

