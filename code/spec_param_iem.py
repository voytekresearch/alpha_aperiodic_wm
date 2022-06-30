"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/"""

# Import neccesary modules
from train_and_test_iem import train_and_test_all_subjs

if __name__ == '__main__':
    # Decode spatial location from alpha oscillatory power
    pw_channel_offsets, pw_ctf_slopes, t_arr = train_and_test_all_subjs('PW')

    # Decode spatial location from aperiodic exponent
    exp_channel_offsets, exp_ctf_slopes, _ = train_and_test_all_subjs(
        'exponent')

    # Decode spatial location from aperiodic offset
    offset_channel_offsets, offset_ctf_slopes, _ = train_and_test_all_subjs(
        'offset')

    # Put all CTF slopes into dictionary
    ctf_slopes = {
        'Power': pw_ctf_slopes, 'Exponent': exp_ctf_slopes,
        'Offset': offset_ctf_slopes}

