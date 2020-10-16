#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = sm.datasets.anes96.load_pandas()
party_ID = np.arange(7)
labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
          "Independent-Independent", "Independent-Republican",
          "Weak Republican", "Strong Republican"]
          
plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # make plot larger in notebook
age = [data.exog['age'][data.endog == id] for id in party_ID]
fig = plt.figure()
ax = fig.add_subplot(111)
plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
                                'label_fontsize':'small',
                                'label_rotation':30}
sm.graphics.beanplot(age, ax=ax, labels=labels,
                     plot_opts=plot_opts)
ax.set_xlabel("Party identification of respondent.")
ax.set_ylabel("Age")
#plt.show()

def beanplot(data, plot_opts={}, jitter=False):
    """helper function to try out different plot options
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_opts_ = {'cutoff_val':5, 'cutoff_type':'abs',
                  'label_fontsize':'small',
                  'label_rotation':30}
    plot_opts_.update(plot_opts)
    sm.graphics.beanplot(data, ax=ax, labels=labels,
                         jitter=jitter, plot_opts=plot_opts_)
    ax.set_xlabel("Party identification of respondent.")
    ax.set_ylabel("Age")
    
fig = beanplot(age, jitter=True)


# In[ ]:




