#!/usr/bin/env python

from PIL import Image
import os
import glob
import sys

size_x = int(sys.argv[1])
size_y = int(sys.argv[2])
args = sys.argv[3:]
for i in args:
	img = Image.open(i)
	img = img.resize((size_x,size_y),Image.Resampling.LANCZOS)
	img.save(i,optimize=True,quality=80)
