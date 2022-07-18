import pickle
from pyexpat import model
from re import X
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalerf

# Save the model
open('medicalcosts.pkl', 'wb')
pickle.dump(model, open('medicalcosts.pkl', 'wb'))


