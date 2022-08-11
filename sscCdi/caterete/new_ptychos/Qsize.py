#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:41:48 2022

@author: yurirt
"""

import sys
import numpy as np

number = 1+1j

size = sys.getsizeof(number)

print("Bytes:", size)




K = 400/4
m = 3072
n = 1000/4

print("K:", K)
print("m:", m)
print("n:", n)


total = size*K*m**2*n**2

print("Size kBs:",total/(1024))
print("Size MBs:",total/(1024**2))
print("Size GBs:",total/(1024**3))
print("Size TBs:",total/(1024**4))