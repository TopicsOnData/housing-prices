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

prices = prices.div(1000)

# Add starting weight and bias
w_init = 2.5e-1 # Increase in price for every 1 square feet
b_init = 200 # Starting price for the cheapest houses

# Iterations and learning rate for the gradient descent algorithm
iterations = 10000
alpha = 1.0e-6 # Delicate and causes a divergence if it's set too large

w_final, b_final, J_hist, p_hist = gradient_descent(
    square_feet, prices, w_init, b_init, alpha, iterations)

print(f'w: {w_final}, b: {b_final}, J_start: {J_hist}, J_End: {p_hist}')

f_wb = []
for i in range(square_feet.size):
    f_wb.append(w_final * square_feet[i] + b_final)


# Plot Square feet vs. Price, in thousands
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
        line=dict(color='red', width=5),
        name='Predicted'
    )
)

fig.update_xaxes(title_text="Square Feet", range=[square_feet.min(), square_feet.max()])
fig.update_yaxes(title_text="Price (in USD, thousands)")

fig.update_layout(
    title_text='Prices of Single-Family Homes vs. Living Space'
    )

fig.show()

# Charts: 
# Zip Codes and Average Price per Living Area Unit
