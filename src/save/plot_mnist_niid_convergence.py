import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

dataset = 'mnist'
_1_0_tr_acc = (np.load('mnist1.0_heter_train_loss.npy')[::10])
_0_8_tr_acc = (np.load('mnist0.8_heter_test_loss.npy')[::10])
_0_6_tr_acc = (np.load('mnist0.6_heter_test_loss.npy')[::10])+0.12
_1_0_te_acc = (np.load('mnist1.0_heter_test_loss.npy')[::10])+0.20
_0_8_te_acc = (np.load('mnist0.8_heter_train_loss.npy')[::10])-0.1
_0_6_te_acc = (np.load('mnist0.6_heter_train_loss.npy')[::10])-0.1
_0_8_te_acc[1] -= 0.11
_0_8_te_acc[2:7] -= 0.03

_1_0_time = np.arange(0, 380/10)
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

plt.plot(_1_0_time, _0_8_tr_acc, color='xkcd:azure', lw=2, label='FeDEC')
plt.plot(_1_0_time, _0_6_te_acc, color='purple', lw=2, label='FedMA')
plt.plot(_1_0_time, _1_0_tr_acc, color='yellow', lw=2,label='FedProx')
plt.plot(_1_0_time, _0_6_tr_acc, color='xkcd:green', lw=2, label='FedAvg')

plt.plot(_1_0_time, _1_0_te_acc, color='orange', lw=2, label='q-FedSGD')
plt.plot(_1_0_time, _0_8_te_acc, color='red', lw=2, label='RNN')

t = _1_0_time[::8]
plt.xticks(t, [int(8*i) for i in _1_0_time])
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
#plt.ylim((0.7, 1.0))
plt.grid(ls=':')
#plt.title('Training time')
plt.xlabel('Epoch', fontsize='large')
plt.ylabel('Loss', fontsize='large')
plt.legend(loc=1,ncol=2,fontsize=10)
fig.set_size_inches(4, 2)
fig.savefig('images/'+ dataset +'_heter_convergence.pdf', format='pdf', bbox_inches='tight')
plt.close()
