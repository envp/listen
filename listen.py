#%%

# Project level imports
from listen.data.an4 import an4
from listen.helpers.helpers import helper

# %%

# Use only the first file for demo purposes
# helpers.helper.save_data(dataset=an4data.trainset.data)
# helper.save_data(dataset=an4data.trainset.data)

an4data = an4.AN4()

helper.save_data(an4data.trainset)

