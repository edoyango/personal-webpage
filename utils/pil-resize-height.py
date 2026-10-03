#!/usr/bin/env python

from PIL import Image, ImageOps
import sys

target_height = int(sys.argv[1])
args = sys.argv[2:]
for i in args:
	img = Image.open(i)
	img = ImageOps.exif_transpose(img)
	width, height = img.size
	if height <= target_height:
		continue
	new_width = round(width * target_height / height)
	img = img.resize((new_width, target_height), Image.LANCZOS)
	img.save(i, optimize=True, quality=80)
