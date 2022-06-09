"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/"""

# Import neccesary modules
from train_and_test_iem import train_and_test_all_subjs

if __name__ == '__main__':
    # Decode spatial location from alpha oscillatory power
    train_and_test_all_subjs('PW')

    # Decode spatial location from aperiodic exponent
    train_and_test_all_subjs('exponent')
