import numpy as np
import Qfunction as QF
data = np.load("11-37-24_raw.npy")
output = data[170,:,:].flatten('F') * np.sqrt(2.0)/13074
x = output[0:500].real
y = output[0:500].imag
