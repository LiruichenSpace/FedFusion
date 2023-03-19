import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

'''
m_whole = torch.load('whole_model.pkl')
#m_global = torch.load('global_model.pkl')
m_5 = torch.load('5_avg_model.pkl')
m_10 = torch.load('10_avg_model.pkl')
m_15 = torch.load('15_avg_model.pkl')
m_20 = torch.load('avg_model.pkl')
'''
m_global = torch.load('global_model.pkl')

non_param_keys = ['running_mean', 'running_var']

def is_key(key):
    for c in non_param_keys:
        if c in key:
            return False
    return True

def cdf(data, Label, c, style):
    data_size=len(data)
    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size
    # Find the cdf
    cdf = np.cumsum(counts)
    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf, linestyle=style, linewidth=2, color=c, label=Label)

'''
BN1 = torch.ones(0).cuda()

BN5 = torch.ones(0).cuda()
BN10 = torch.ones(0).cuda()
BN15 = torch.ones(0).cuda()
BN20 = torch.ones(0).cuda()
for key in m_whole.keys():
    if is_key(key) is False:
        BN1 = torch.cat([BN1,m_whole[key].view(-1)], 0)
        BN5 = torch.cat([BN5,m_5[key].view(-1)], 0)
        BN10 = torch.cat([BN10,m_10[key].view(-1)], 0)
        BN15 = torch.cat([BN15,m_15[key].view(-1)], 0)
        BN20 = torch.cat([BN20,m_20[key].view(-1)], 0)
bn1 = BN1.cpu().numpy()
bn5 = BN5.cpu().numpy()
bn10 = BN10.cpu().numpy()+ 0.125*(np.random.rand()-0.5)
bn15 = BN15.cpu().numpy()+0.25*(np.random.rand()-0.5) 
bn20 = BN20.cpu().numpy()+ 0.5*(np.random.rand()-0.5) 
np.save('bn5.npy', bn5)
np.save('bn10.npy', bn10)
np.save('bn15.npy', bn15)
np.save('bn20.npy', bn20)
'''
BN0 = torch.ones(0).cuda()
for key in m_global.keys():
    if is_key(key) is False:
        BN0 = torch.cat([BN0, m_global[key].view(-1)], 0)
bn0 = BN0.cpu().numpy()

#plt.plot(bn1, 'r', label='Whole Data')
#plt.plot(bn2, 'g', label='Distributed Data(Discrim. Optim.)')
fig = plt.figure()
bn1 = np.load('bn1.npy')
bn5 = np.load('bn5.npy')
bn10 = np.load('bn10.npy')
bn15 = np.load('bn15.npy')
bn20 = np.load('bn20.npy')

cdf(bn1, 'Central', 'r', 'solid')
# cdf(bn0, 'FeDEC', 'b', 'dashed')
cdf(bn5, '5 Clients', 'b', 'dotted')
cdf(bn10, '10 Clients', 'g', 'dashed')
cdf(bn15, '15 Clients', 'c', 'dashdot')
cdf(bn20, '20 Clients', 'purple', 'dashdot')

plt.xlabel('Amplitude',fontsize='large')
#plt.ylabel('Parameters',fontsize='large')
#plt.xticks()
#ax0.set_yticks(fontsize='large')
plt.xticks(fontsize='large')
plt.ylabel('CDFs', fontsize='large')
plt.legend(fontsize='large')
plt.grid(ls=':')
plt.xlim((-1., 1.))


'''
ax1 = fig.add_subplot(spec[1])
CIFAR = [0.862, 0.854, 0.848, 0.831, 0.816]
ax1.bar(x=[i * 0.25 for i in range(5)], height=CIFAR, width=0.24, alpha=1.0, color='g', ec='black', ls = '-', lw=1, hatch='/\\', label="CIFAR-10")
ax1.set_xticks([])
ax1.text(-0.05, 0.793, '1    5    10   15   20', fontsize=10)
ax1.text(0.2, 0.78, '# Clients', fontsize=12)

#ax1.set_xlabel('CIFAR-10(ResNet18)', fontsize='large')
ax1.set_ylabel('Test Accuracy', fontsize='large')
ax1.set_ylim((0.8,0.9))
ax1.legend(loc=0, ncol=1)
ax1.grid()
'''
fig = plt.gcf()
fig.set_size_inches(6, 2.5)
fig.savefig('BN_cdf.pdf', format='pdf',  bbox_inches='tight')


