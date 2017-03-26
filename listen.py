# %%
# %load_ext autoreload
# %autoreload 2

#%%

# Project level imports
from listen.data.an4.an4 import AN4
from listen.helpers.helpers import helper

# %%

# Use only the first file for demo purposes
an4data = AN4(debug=False, conversion=False)
# helpers.helper.save_data(dataset=an4data.trainset.data)
# helper.save_data(dataset=an4data.trainset.data)
