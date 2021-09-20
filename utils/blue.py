import math
from collections import Counter
import re

import numpy as np


def blue_stats(hypothesis, reference):
    """Compute statistics for BLUE

    Args:
        hypothesis ([type]): [description]
        reference ([type]): [description]
    """
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))

    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n,0]))

    return stats


def blue(stats):
    """Compute BLUE given n-gram statistics

    Args:
        stats ([type]): [description]
    """
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    
    (c, r) = stats[:2]
    log_blue_prec = sum([math.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.

    return math.exp(min([0,1-float(r)/c]) + log_blue_prec)


def get_blue(hypothesis, reference):
    """Get validation BLUE score for dev set

    Args:
        hypothesis ([type]): [description]
        reference ([type]): [description]
    """
    stats = np.zeros(10)
    
    for hyp, ref in zip(hypothesis, reference):
        stats += np.array(blue_stats(hyp, ref))
    
    return 100*blue(stats)


def idx_to_word(x, vocab):
    """Convert index number into word from vocabulary

    Args:
        x ([type]): [description]
        vocab ([type]): [description]

    Returns:
        str: sentence
    """    
    words = []
    
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    
    words = " ".join(words)
    
    return words
