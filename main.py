#%%
# Project level imports
from listen.data.an4.an4 import AN4
from listen.helpers.helpers import helper

def main():
    # Set conversion = true to convert the raw files to wav
    an4data = AN4(conversion=False)

    # Uncomment this to run the MFCC computations
    # and save spectral data to disk
    helper.save_data(an4data.trainset)

if __name__ == '__main__':
    main()
