#!/usr/bin/env python

from PIL import Image
import os
import glob
import sys

args = sys.argv[1:]
for i in args:
	img = Image.open(i)
	img.save(i,optimize=True,quality=100)
