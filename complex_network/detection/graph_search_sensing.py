"""This module implements the graph searching sensing algorithms, that takes peaks from an OLCR scan and then returns the possible
    position on the network that can explain that peak
    
    Assumptions.
        The perturbation is modelled as an additional node in between the link that scatters.
        Currently, we only support one perturbation within the network (i.e. one additional node), the idea is that faults are rare events
        The perturbation doesn't add any additional phase that can increase the optical path length of measured peaks
        We haven't considered the effects of dispersion which shifts OLCR peaks which is quite important"""


import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque, defaultdict


