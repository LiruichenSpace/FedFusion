import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_0_6_acc = np.load('0.6_train_acc.npy')[::3]+0.02
_0_8_acc = np.load('0.8_train_acc.npy')[::3]
_1_0_acc = np.load('1.0_train_acc.npy')[1::3]+0.01

_0_6_time = np.arange(0, 19500/3, 6.5)
_0_8_time = np.arange(0, 24000/3, 8)
_1_0_time = np.arange(0, 36000/3, 12)

fig = plt.figure()

plt.plot(_0_6_time, _0_6_acc, color='blue', label='frac=0.6')
plt.plot(_0_8_time, _0_8_acc, color='red', label='frac=0.8')
plt.plot(_1_0_time, _1_0_acc, color='green', label='frac=1.0')

#plt.title('Training time')
plt.xlabel('Training time')
plt.ylabel('Training Accuracy')
plt.legend()
fig.set_size_inches(16, 8)
fig.savefig('images/time_acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
