# %%
# Project level imports
from listen.data.an4.an4 import AN4
from tqdm import tqdm
import pprint


def main():
    # Set conversion = true to convert the raw files to wav
    # Set phones=True to numerize phones (including silence)
    an4data = AN4(conversion=False, phones=True)

if __name__ == '__main__':
    main()
