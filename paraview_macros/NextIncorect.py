import os
import socket
from paraview.simple import *

TIFF_ROOT = "C:/Users/Nylan/projekter/Karl-Johan/combined/"

def read_volume(path):
    reader = TIFFReader(FileName=path)
    RenameSource(path.split(TIFF_ROOT)[1], reader)

    disp = GetDisplayProperties()
    disp.Representation = "Volume"

    display = Show(reader)
    ColorBy(display, ('POINTS', 'Tiff Scalars'))  # Sets the correct data to be used for coloring
    disp.MultiComponentsMapping = True
    return reader

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))
client_socket.send(b"\x11")

for source in GetSources().values():
    Delete(source)

bytes = client_socket.recv(10)
if bytes != b"No change":
    file_names = [f"{TIFF_ROOT}/{f}" for f in os.listdir(TIFF_ROOT) if f.endswith(".tif")]
    reader = read_volume(file_names[0])

    for file_name in file_names[1:]:
        reader = read_volume(file_name)
        Hide(reader)

ResetCamera()
Render()
