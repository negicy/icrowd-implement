import numpy as np
from scipy import linalg

d = np.dia(np.sum(sim_matrix, axis=0))

norm_sim_matrix = np.dot(np.sqrt(linalg.inv(d)), np.dot(sim_matrix, np.sqrt(linalg.inv(d))))