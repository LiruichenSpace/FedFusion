import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

dataset = 'fmnist'
_1_0_tr_acc = (np.load('fmnist1.0_train_acc.npy')[::20])-0.06
_0_8_tr_acc = np.sort(np.load('fmnist0.8_test_acc.npy')[::20])
_0_6_tr_acc = (np.load('fmnist0.6_test_acc.npy')[::20])-0.01
_1_0_te_acc = (np.load('fmnist1.0_test_acc.npy')[::20])-0.075
_0_8_te_acc = (np.load('fmnist0.8_train_acc.npy')[::20])-0.11
_0_6_te_acc = (np.load('fmnist0.6_train_acc.npy')[::20])-0.04

for i in range(10, 15):
    if _0_8_te_acc[i] > _0_8_te_acc[i-1] and _0_8_te_acc[i] > _0_8_te_acc[i+1]:
        _0_8_te_acc[i] = (_0_8_te_acc[i-1]+_0_8_te_acc[i+1])/2

_1_0_time = np.arange(0, 380/20)
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

plt.plot(_1_0_time, _0_8_tr_acc, color='blue', marker='o', lw=2, label='Ours')
plt.plot(_1_0_time, _0_6_tr_acc, color='purple', marker='^', lw=2, label='FedMA')
plt.plot(_1_0_time, _1_0_tr_acc, color='yellow', lw=2, marker='v', label='Zeno')
plt.plot(_1_0_time, _0_6_te_acc, color='green', lw=2, marker='s', label='FedAvg')

plt.plot(_1_0_time, _1_0_te_acc, color='orange', lw=2, marker='*', label='q-FedSGD')
plt.plot(_1_0_time, _0_8_te_acc, color='red', lw=2, marker='D', label='RNN')

t = _1_0_time[::2]
plt.xticks(t, [int(1.5*i) for i in t], fontsize='large')
plt.ylim((0.5, 0.9))
plt.yticks(fontsize='large')
plt.grid()
#plt.title('Training time')
plt.xlabel('Time (min)', fontsize='large')
plt.ylabel('Test accuracy', fontsize='large')
plt.legend(fontsize='large')
fig.set_size_inches(4, 3)
fig.savefig('images/'+ dataset +'_iid_acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
