# -----------------------------------------------------------------------------
# A Python/Numpy implementation of Hierarchical Temporal Memory
# -----------------------------------------------------------------------------
import numpy as np

n = 16      # Number of mini-columns within a HTM unit
m =  4      # Number of cells within a mini-column
d = 20      # Number of distal segments per cell
s = 16      # Number of synapses per segment
beta = 0.5  # Synapse connection threshold
            #  (permanence value has to be > beta for synapse to be active)
theta = 5   # Synapse activation threshold (theta < s)
            #  (theta synapses have to be co-activated to make the cell active)
LTP  = 0.10 # Long term potentiation (learning)
LTD  = 0.01 # Long term depression (forgetting)

# HTM unit actual activity (A)
#  → A[i,j] is the activity of the i'th cell in the j'th column
# A = np.zeros((n,m), dtype = int)
A = np.zeros((n,m), dtype = [("t", int),
                             ("t-1", int)])

# HTM unit predicted activity (Π)
#  → P[i,j] is the predictive state of the i'th cell in the j'th column
# P = np.zeros((n,m), dtype = int)
P = np.zeros((n,m), dtype = [("t", int),
                             ("t-1", int)])
            
# Segments & synapses (D)
#  → D[i,j,d] represents the d'th segment of the i'th cell in the j'th column
# →  Each distal segment contains a number of synapses (s)
D = np.zeros((n, m, d, n*m), dtype=int)
D[:,:,:,:s] = 1
for i,j,k in np.ndindex(n,m,d):
    np.random.shuffle(D[i,j,k])
    # No self connections
    while D[i,j,k,i*j] == 1:
        np.random.shuffle(D[i,j,k])

# Segment mask
Dm = D.reshape(n, m, d, n, m)
# Segment permanence values
D = Dm * np.random.uniform(0, 1, (n, m, d, n, m))
# Active synapses
Di = D > beta

# HTM unit feedforward weigth (Wᵗ)
W = np.zeros((n,m), dtype = [("t", int)])


# Stimuli sparse code
S = { "A" : np.array([1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]).reshape(n,1),
      "B" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]).reshape(n,1),
      "C" : np.array([0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0]).reshape(n,1) }


def _repr(Z):
    """
    ∙ : inactive
    ● : active
    """
    text = ""
    Z = np.atleast_2d(Z)
    for j in range(Z.shape[1]):
        for i in range(Z.shape[0]):
            v = Z[i,j]
            if v == 0: text += "∙"
            else:      text += "●"
        if j < Z.shape[1]-1:
            text += "\n"
    return text

def repr(Z1, Z2=None):
    """
    ∙ : inactive
    ● : active
    ○ : precicted 
    ⦿ : active and predicted
    """
    if Z2 is None:
        return _repr(Z1)
    R1, R2 = _repr(Z1), _repr(Z2)
    R = ""
    for c1,c2 in zip(R1,R2):
        c = c1
        if c1 == '●':
            if c2 == '●':
                c = '⦿'
        elif c2 == '●':
            c = '○'
        R += c
    return R
        

# np.random.seed(11)


# Step 1: initialization of the synapses
# --------------------------------------
# Current feed forward input pattern is W
W['t'] = S["A"]


# Step 2: computing cell states
# -----------------------------
# Computing activity A[t] (depends on prediction P[t-1])
WP = W['t']*P['t-1']
A['t'][:] = W
A['t'][WP.sum(-1).nonzero(),:] = 0
A['t'] += WP

# Computing prediction P[t] (depends on activity A[t])

# Segments activation
S = ((Di*A['t']).sum((-2,-1))) > theta

# Prediction is active if any of a cell segments is above threshold
P['t'] = (S.sum(-1) > 0).astype(int)


# Learning
# --------
# Segments activation
S = ((Di*A['t-1']).sum((-2,-1)))

# Index of most activated segment in a cell (shape is(m,n))
Si = np.argmax(S,-1)

# Index of most activated cell in a column (shape is (m,))
Ci = np.argmax(np.max(S,-1),-1)

# Active segments
S = (S > theta)

# Not finished...


# print(repr(A['t'],P['t']))
# print()
# print(repr(P))
# print()
