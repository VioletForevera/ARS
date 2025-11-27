#!/usr/bin/env python3
"""
MountainCar训练和演示
"""

import gymnasium as gym
import time
import argparse
import collections
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from mountaincar_cl.environments import MountainCarCL, TaskScheduler, DynamicScenario
from mountaincar_cl.run_mountaincar import main
main()



