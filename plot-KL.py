import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import entropy


m_global = torch.load('global_model.pkl')

non_param_keys = ['running_mean', 'running_var']

def is_key(key):
    for c in non_param_keys:
        if c in key:
            return False
    return True

def COUNT(data):
    data_size=len(data)
    # Set bins edges
    data_set=sorted(set(data))
    bins= np.arange(-2, 2, 0.1)
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size
    return counts

# BN0 = torch.ones(0).cuda()
# for key in m_global.keys():
#     if is_key(key) is False:
#         BN0 = torch.cat([BN0, m_global[key].view(-1)], 0)
# bn0 = COUNT(BN0.cpu().numpy())


# fig = plt.figure()
# # spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[6, 2])
# # ax0 = fig.add_subplot(spec[0])
# bn1 = COUNT(np.load('bn1.npy'))
# bn5 = COUNT(np.load('bn5.npy'))
# bn10 = COUNT(np.load('bn10.npy'))
# bn15 = COUNT(np.load('bn15.npy'))
# bn20 = COUNT(np.load('bn20.npy'))

# kl0 = entropy(bn0, bn1)
# kl5 = entropy(bn5, bn1)
# kl10 = entropy(bn10, bn1)
# kl15 = entropy(bn15, bn1)
# kl20 = entropy(bn20, bn1)

# resnet densenet mobilenet lenet(FMNIST) lenet(MNIST)
FeDEC = [0.052, 0.079, 0.098, 0.034, 0.021]
FedMA = [0.26, 0.30, 0.36, 0.081, 0.043]
FedProx = [0.53, 0.77, 0.64, 0.11, 0.08]
FedAvg = [0.62, 0.80, 0.68, 0.13, 0.09]
qFedSGD = [0.43, 0.57, 0.61, 0.08, 0.08]
RNN = [0, 0, 0, 0.17, 0.12]
x = range(len(FeDEC))

plt.grid(ls=':')

rectsfedec = plt.bar(x=x, height=FeDEC, width=0.1, alpha=0.8, color='xkcd:azure', label="FeDEC")
rectsfedma = plt.bar(x=[i + 0.1 for i in x], height=FedMA, width=0.1, color='purple', label="FedMA")
rectsfeprox = plt.bar(x=[i + 0.2 for i in x], height=FedAvg, width=0.1, color='xkcd:green', label="FedAvg")
rectsfedavg = plt.bar(x=[i + 0.3 for i in x], height=FedProx, width=0.1, color='yellow', label="FedProx")
rectsqfedsgd = plt.bar(x=[i + 0.4 for i in x], height=qFedSGD, width=0.1, color='orange', label="q-FedSGD")
rectsrnn = plt.bar(x=[i + 0.5 for i in x], height=RNN, width=0.1, color='red', label="RNN")

# cdf(bn1, 'Central', 'r', 'solid')
# # cdf(bn0, 'FeDEC', 'b', 'dashed')
# cdf(bn5, '5 Clients', 'b', 'dotted')
# cdf(bn10, '10 Clients', 'g', 'dashed')
# cdf(bn15, '15 Clients', 'c', 'dashdot')
# cdf(bn20, '20 Clients', 'purple', 'dashdot')
plt.ylim((0.0, 1.0))
label_list = ['ResNet\n(CIFAR-10)', 'DenseNet121\n(CIFAR-10)', 'MobileNetV2\n(CIFAR-10)', 'LeNet\n(FMNIST)', 'LeNet\n(MNIST)']
plt.xticks([index + 0.25 for index in x], label_list, fontsize=9, rotation=30)
# plt.xlabel([])
#plt.xticks()
#ax0.set_yticks(fontsize='large')
#ax0.set_xticks()
plt.ylabel('KL-divergence', fontsize='large')
plt.legend(loc=0,frameon=False, ncol=3, fontsize=9, mode='expand')     # 设置题注 , mode='expand'
# plt.set_xlim((-1., 1.))

ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(2.5)
fig = plt.gcf()
fig.set_size_inches(4, 2)
fig.savefig('BN_KL.pdf', format='pdf',  bbox_inches='tight')


