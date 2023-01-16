from paraview.simple import *
import os
import glob

TIFF_ROOT = "/Karl-Johan"

file_names = glob.glob(f"{TIFF_ROOT}/*.tif")
for file_name in file_names:
    reader = TIFFReader(FileName=file_name)
    disp = GetDisplayProperties()
    disp.Representation = "Volume"

    display = Show(reader)
    ColorBy(display, ('POINTS', 'Tiff Scalars'))  # Sets the correct data to be used for coloring
    disp.MultiComponentsMapping = True

HideAll()
Show(reader)
Render()
