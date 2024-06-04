import plotly.graph_objects as go
import pandas as pd
from pandas import Series, DataFrame

df = pd.read_csv('properties.csv')

# Draw the figure
fig = go.Figure()

# Drop any rows where the Living Area is blank
df = df.dropna(subset=['Living area'])

# Add the data
prices = df['Property price (USD)']
square_feet = df['Living area']

print(prices)
'''
# Add the trace
# Square feet vs. Price
fig.add_trace(go.Scatter(
    x=l18_s['RATE'], 
    y=l18_s['STATE'],
    mode='markers',
    name='Worst Life Expectancies in 2018',
    text=l18_s['STATE']
    )
)'''

# Charts: 
# Zip Codes and Average Price per Living Area Unit