from paraview.simple import *
import os

TIFF_ROOT = "/Karl-Johan"  # Just put the absolute path to Karl-Johan

reader = TIFFReader(FileName=f"{TIFF_ROOT}/temp_image.tif")
# help(reader)
Show()
Render()

