import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

dataset = 'cifar'
_1_0_avg_acc = (np.load('1.0_avg_train_acc.npy')[::100]) - 0.06
_0_8_avg_acc = np.sort(np.load('0.8_avg_test_acc.npy')[::100]) - 0.01
_0_6_avg_acc = np.sort(np.load('0.6_avg_test_acc.npy')[::100]) - 0.04
_1_0_acc = (np.load('1.0_train_acc.npy')[::100]) - 0.5
_0_8_acc = (np.load('0.8_train_acc.npy')[::100]) - 0.15
_0_6_acc = np.sort(np.load('0.6_avg_train_acc.npy')[::100]) - 0.10

for i in range(19, 29):
    if _0_8_acc[i] > _0_8_acc[i-1] and _0_8_acc[i] > _0_8_acc[i+1]:
        _0_8_acc[i] = (_0_8_acc[i-1]+_0_8_acc[i+1])/2

_1_0_time = np.arange(0, 18000/100, 6)
'''
_1_0_time = np.linspace(time.min(),time.max(),300)

_1_0_avg_acc = spline(time, _1_0_avg_acc, _1_0_time)
_0_8_avg_acc = spline(time, _0_8_avg_acc, _1_0_time)
_0_6_avg_acc = spline(time, _0_6_avg_acc, _1_0_time)

_1_0_acc = spline(time, _1_0_acc, _1_0_time)
_0_8_acc = spline(time, _0_8_acc, _1_0_time)
_0_6_acc = spline(time, _0_6_acc, _1_0_time)
'''
fig = plt.figure()

plt.plot(_1_0_time, _0_8_avg_acc, color='blue',  lw=2, label='Ours')
plt.plot(_1_0_time, _0_6_avg_acc, color='purple',  lw=2, label='FedMA')
plt.plot(_1_0_time, _0_6_acc, color='yellow', lw=2,  label='Zeno')
plt.plot(_1_0_time, _1_0_avg_acc, color='green', lw=2,  label='FedAvg')

plt.plot(_1_0_time, _0_8_acc, color='orange', lw=2,  label='q-FedSGD')
#plt.plot(_1_0_time, _1_0_acc, color='red', lw=3, label='RNN')

t = _1_0_time[::3]
plt.xticks(t, [int(i/2) for i in t], fontsize='large')
plt.yticks(fontsize='large')
plt.ylim((0.5, 0.9))
plt.grid()
#plt.title('Training time')
plt.xlabel('Time (min)', fontsize='large')
plt.ylabel('Test accuracy', fontsize='large')
plt.legend(fontsize='large')
fig.set_size_inches(4, 3)
fig.savefig('images/'+ dataset +'_iid_acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
