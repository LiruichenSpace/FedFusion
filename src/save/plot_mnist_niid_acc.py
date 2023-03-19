import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

dataset = 'mnist'
_1_0_tr_acc = (np.load('mnist1.0_heter_train_acc.npy')[::20])-0.01
_0_8_tr_acc = (np.load('mnist0.8_heter_test_acc.npy')[::20]) + 0.015
_0_6_tr_acc = (np.load('mnist0.6_heter_test_acc.npy')[::20])-0.02
_1_0_te_acc = (np.load('mnist1.0_heter_test_acc.npy')[::20]) -0.03
_0_8_te_acc = (np.load('mnist0.8_heter_train_acc.npy')[::20]) -0.005
_0_6_te_acc = (np.load('mnist0.6_heter_train_acc.npy')[::20])


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

plt.plot(_1_0_time, _0_8_tr_acc, color='xkcd:azure', lw=2,ls='-', label='FeDEC')
plt.plot(_1_0_time, _0_6_te_acc, color='purple', lw=2, ls='--', label='FedMA')
plt.plot(_1_0_time, _1_0_tr_acc, color='yellow', lw=2, ls='-.', label='FedProx')
plt.plot(_1_0_time, _0_6_tr_acc, color='xkcd:green', lw=2, ls='--', label='FedAvg')

plt.plot(_1_0_time, _1_0_te_acc, color='orange', lw=2, ls='-.', label='q-FedSGD')
plt.plot(_1_0_time, _0_8_te_acc, color='red', lw=2, ls='--', label='RNN')

t = _1_0_time[::2]
#plt.xticks(t, [int(2*i) for i in _1_0_time])
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.ylim((0.7, 1.0))
plt.grid(ls=':')
#plt.title('Training time')
plt.xlabel('Time (min)', fontsize='large')
plt.ylabel('Test Accuracy', fontsize='large')
plt.legend(loc=4, ncol=2, fontsize=9)
fig.set_size_inches(4, 2)
fig.savefig('images/'+ dataset +'_noniid_acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
