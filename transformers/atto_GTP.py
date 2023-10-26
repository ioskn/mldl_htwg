from tensorflow.keras.layers import LayerNormalization, Dense, LayerNormalization, Softmax 
import tensorflow as tf 
import numpy as np

N_Token = 16 #
m, H = 3, 2 

B,T,C = 1, 5, m*H #Batch, Time, Channels
data = np.array([[0,  3,  6,  9, 12, 15]]) #T+1 data points

#Creating the input and output pairs
y_observed = data[:,1:]            # 3,6,9,12,15 shape (B, T)
T_input = tf.Variable(data[:,0:T]) # 0,3,6, 9,12 shape (B, T)
T_input.shape, y_observed.shape

#Creating the embedding layer of the input
embedding_matrix = tf.Variable(tf.random.normal([N_Token, C]))
#tf.gather picks the rows of the embedding matrix corresponding to the input tokens
x = tf.gather(embedding_matrix, T_input) #shape (B, T, C)
print(np.round(x[0].numpy().T, 2))

### Attention Layer calculation of the weight matix W
wk = tf.random.normal(shape=(H, T, m)) #The weights for the key
K = tf.einsum('btc, htm -> bhtm', x, wk) # Transformation in the B, H, T, m 
print(np.round(K[0,0].numpy().T, 2))

wq = tf.random.normal(shape=(H, T, m))
Q = tf.einsum('btc, htm -> bhtm', x, wq) 
print(np.round(Q[0,0].numpy().T, 2))

wV = tf.random.normal(shape=(H, T, m))
V = tf.einsum('btc, htm -> bhtm', x, wV) 
print(np.round(V[0,0].numpy().T, 2))

W = tf.einsum('bhtd, bhmd -> bhtm', Q, K) # B, H, T, T

#Normalization
W = W / tf.sqrt(1.0*m)

#Causal Masking (Still a bit ugly code)
# Create a mask with zeros in the lower triangular part, including diagonal
W_shape = W.shape
mask = tf.ones(W_shape[-2:]) - tf.linalg.band_part(tf.ones(W_shape[-2:]), -1, 0)
mask = mask * -tf.float32.max# Broadcast the mask to the shape of W
mask = tf.broadcast_to(mask, W_shape)
W = W + mask
print(np.round(W[0,0].numpy(), 2))

#Softmax
W = tf.nn.softmax(W, axis=-1)
print(np.round(W[0,0].numpy(), 2))

#Multiply with V
xout = tf.einsum('bhtm, bhtd -> bhtd', W, V) # B, H, T, m
print(np.round(xout[0,0].numpy().T, 2))
print(np.round(xout[0,1].numpy().T, 2))

#Concactinate the heads (not the fastest solution)
xout_list = []
# Loop over the attention heads (axis=1 of xout)
for i in range(H):
    xout_head = xout[:, i, :, :]  # Shape: (B, T, m)
    xout_list.append(xout_head)
# Concatenate along the last axis (axis=-1)
xout = tf.concat(xout_list, axis=-1)
print(np.round(xout[0].numpy().T, 2))

# A linear layer
xout = tf.keras.layers.Dense(C)(xout)

# A residual connection
x = x + xout

# A layer normalization
x = LayerNormalization()(x) #np.round(tf.reduce_mean(x, axis=-1),2)

# A feed forward layer (along the C axis)
xout = Dense(4*C, activation='relu')(x) #In GPT a GeLu activation is used
xout = Dense(C)(xout)
x = x + xout #Again Residual connection

# This attention block is repeated many times
x = LayerNormalization()(x) #Layer Normalization
xout = Dense(N_Token, activation='relu')(x) #From (B, T, C) to (B, T, N_Token)
pout = Softmax(axis=-1)(xout) #Softmax along the last axis


# The loss function
y_observed_one_hot = tf.one_hot(y_observed, depth=N_Token) # shape: (B, T, N_Token)
loss = -tf.reduce_sum(y_observed_one_hot * tf.math.log(pout + 1e-10)) / (B * T)  # Add a small constant for numerical stability

print("Loss:", loss.numpy())