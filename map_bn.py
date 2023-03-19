import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

m_whole = torch.load('whole_model.pkl')
#m_global = torch.load('global_model.pkl')
m_global = torch.load('5_avg_model.pkl')

non_param_keys = ['running_mean', 'running_var']

def is_key(key):
    for c in non_param_keys:
        if c in key:
            return False
    return True

BN1 = torch.ones(0).cuda()
BN2 = torch.ones(0).cuda()
for key in m_whole.keys():
    if is_key(key) is False:
        BN1 = torch.cat([BN1,m_whole[key].view(-1)], 0)
        BN2 = torch.cat([BN2,m_global[key].view(-1)], 0)
bn1 = BN1.cpu().numpy()
bn2 = BN2.cpu().numpy() 

plt.plot(bn1, 'r', label='Whole Data')
plt.plot(bn2, 'g', label='Distributed Data(Discrim. Optim.)')
plt.ylabel('Amplitude',fontsize='large')
plt.xlabel('Parameters',fontsize='large')
plt.xticks()
plt.yticks(fontsize='large')

plt.legend(fontsize='large')
fig = plt.gcf()
fig.set_size_inches(6, 2.5)
fig.savefig('map_BN_avg.pdf', format='pdf',  bbox_inches='tight')


