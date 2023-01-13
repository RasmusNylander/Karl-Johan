from paraview.simple import *
import os

TIFF_ROOT = "/Karl-Johan"  # Just put the absolute path to Karl-Johan

reader = TIFFReader(FileName=f"{TIFF_ROOT}/temp_image.tif")
disp = GetDisplayProperties()
disp.Representation = "Volume"

display = Show(reader)
ColorBy(display, ('POINTS', 'Tiff Scalars'))  # Sets the correct data to be used for coloring

Render()
