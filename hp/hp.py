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

# Create Linear Regression Model

# 1) Number of training examples
m = df.shape[0] # Returns number of rows; df.shape[1] returns number of columns

# 2) Create container for model evaluation at training example & cost function
f_wb = []
cost = []

# 3) Add weight and bias
w = 500 # Increase in price for every 1 square feet
b = 100 # Starting price for the cheapest houses

# 4) Index the training examples and add a cost function
for i in range(m):
    x_i = square_feet[i]
    y_i = prices[i]
    # 5) Compute prediction at every x_i
    y_hat = w*x_i + b
    cost_i = (1/(2*m))*(y_hat-y_i)**2
    f_wb.append(y_hat)
    cost.append(cost_i)

# 5) OPTIONAL METRIC: Total cost of function
total_cost = sum(cost)

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
# Zip Codes and Average Price per Living Area Unit
