"""Reproduce decoding of spatial location from total alpha power using an IEM by
Foster and colleagues (https://pubmed.ncbi.nlm.nih.gov/26467522/)"""

# Import neccesary modules
from train_and_test_iem import train_and_test_all_subjs

if __name__ == '__main__':
    # Reproduce total alpha power IEM
    train_and_test_all_subjs('total_power')
