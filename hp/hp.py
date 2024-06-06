# Import the machine learning functions
import sys
sys.path.append('../')
from ml.gradient import gradient_descent, cost_function, gradient_function

import plotly.graph_objects as go
import pandas as pd
from pandas import Series, DataFrame

df = pd.read_csv('properties.csv')

# Draw the figure
fig = go.Figure()

# Drop any rows where the Living Area is blank
df = df.dropna(subset=['Living area'])

# Keep only Single-Family Homes (SFH)
df = df.loc[df['Property type'] == 'Single Family']

# Reset index, drop old index
df = df.reset_index(drop=True)

# Add the data
prices = df['Property price (USD)']
square_feet = df['Living area']

# Add starting weight and bias
w_init = 250 # Increase in price for every 1 square feet
b_init = 200000 # Starting price for the cheapest houses

# Iterations and learning rate for the gradient descent algorithm
iterations = 3
alpha = 5.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(
    square_feet, prices, w_init, b_init, alpha, iterations)

# print(f'w: {w_final}, b: {b_final}')
'''
# Plot Square feet vs. Price
fig.add_trace(
    go.Scatter(
        x=square_feet, 
        y=prices,
        mode='markers',
        name='Actual'
    )
)

# Plot Linear Regression Model
fig.add_trace(
    go.Scatter(
        x=square_feet, 
        y=f_wb,
        mode='lines',
        line=dict(color='skyblue', width=3),
        name='Predicted'
    )
)

fig.update_xaxes(title_text="Square Feet")
fig.update_yaxes(title_text="Price (in USD)")

fig.update_layout(
    title_text='Prices of Single-Family Homes vs. Living Space'
    )

fig.show()

# Charts: 
# Zip Codes and Average Price per Living Area Unit'''
