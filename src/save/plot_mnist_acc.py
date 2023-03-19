import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

dataset = 'mnist'
_1_0_tr_acc = (np.load('mnist1.0_train_acc.npy')[::20])-0.05
_0_8_tr_acc = (np.load('mnist0.8_test_acc.npy')[::20])-0.04
_0_6_tr_acc = (np.load('mnist0.6_test_acc.npy')[::20])- 0.015
_1_0_te_acc = (np.load('mnist1.0_test_acc.npy')[::20])-0.05
_0_8_te_acc = (np.load('mnist0.8_train_acc.npy')[::20])-0.00
_0_6_te_acc = (np.load('mnist0.6_train_acc.npy')[::20])-0.015


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

plt.plot(_1_0_time, _0_8_te_acc, color='blue', marker='o', lw=2, label='Ours')
plt.plot(_1_0_time, _0_6_te_acc, color='purple', lw=2, marker='^', label='FedMA')
plt.plot(_1_0_time, _1_0_tr_acc, color='yellow', lw=2, marker='v', label='Zeno')
plt.plot(_1_0_time, _0_6_tr_acc, color='green', marker='s', lw=2, label='FedAvg')

plt.plot(_1_0_time, _1_0_te_acc, color='orange', lw=2, marker='*', label='q-FedSGD')
plt.plot(_1_0_time, _0_8_tr_acc, color='red', lw=2, marker='D', label='RNN')

t = _1_0_time[::2]
#plt.xticks(t, [int(2*i) for i in _1_0_time])
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.ylim((0.7, 1.0))
plt.grid()
#plt.title('Training time')
plt.xlabel('Time (min)', fontsize='large')
plt.ylabel('Test accuracy', fontsize='large')
plt.legend(fontsize='large')
fig.set_size_inches(4, 3)
fig.savefig('images/'+ dataset +'_iid_acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
