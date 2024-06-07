# Cost Function to determine cumulative error between real and predicted values
def cost_function(x, y, w, b):

    # 1) Number of training examples
    m = x.size
    cost = 0

    # 2) Index the training examples and account for cost per instance
    for i in range(m):
        y_hat = w * x[i] + b      
        cost += (y_hat-y[i])**2
        cost /= 2 * m

    # 3) Return total cost
    return cost

# Compute the gradient, i.e., the scalar that improves accuracy
def gradient_function(x, y, w, b):

    m = x.size

    # Partial derivatives of the cost function with respect to weight and bias
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        y_hat = w * x[i] + b
        dj_dw_i = (y_hat - y[i]) * x[i]
        dj_db_i = (y_hat - y[i])
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, learning_rate, 
    num_iters):
    
    # Used for graphing
    J_history = []
    p_history = []
    b = b_init
    w = w_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b) # Gradient

        # Update weight, bias
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db
        
        # Prevents resource exhaustion; unnecessary to store similar costs
        # Past 100000 iterations
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])
    
    return w, b, J_history, p_history
