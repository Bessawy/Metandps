import numpy as np
import matplotlib.pyplot as plt

train_losses = np.load('meta_test_losses_ndp.npy')
loss = np.load('demo_losses_ndp.npy')

ind = 70

#plt.plot(np.arange(ind),loss[:ind])
#plt.show()

#plt.plot(np.arange(500-ind)+ind, loss[ind:])
#plt.show()

plt.plot(np.arange(ind),loss[:ind], label='normal')
plt.plot(np.arange(ind),train_losses[:ind], label='meta')
plt.legend()
plt.show()

plt.plot(np.arange(500-ind)+ind, loss[ind:], label='normal')
plt.plot(np.arange(500-ind)+ind, train_losses[ind:], label='meta')
plt.legend()
plt.show()
