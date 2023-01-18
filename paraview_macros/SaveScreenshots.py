import os
import socket
from paraview.simple import *

TIFF_ROOT = "C:/Users/oisin/Documents/Uni/DTU/02507/repo/Karl-Johan/combined/"
IMAGE_ROOT = "C:/Users/oisin/Documents/Uni/DTU/02507/repo/Karl-Johan/screenshots/"


def read_volume(path):
    reader = TIFFReader(FileName=path)
    RenameSource(path.split(TIFF_ROOT)[1], reader)

    disp = GetDisplayProperties()
    disp.Representation = "Volume"

    display = Show(reader)
    ColorBy(display, ('POINTS', 'Tiff Scalars'))  # Sets the correct data to be used for coloring
    disp.MultiComponentsMapping = True
    return reader


def Save():
    filename = list(GetSources().keys())[list(GetSources().values()).index(GetActiveSource())][0]
    filename = filename[:-4]
    print(filename)

    SaveScreenshot(f"{IMAGE_ROOT}{filename[:6]}/{filename}.png")


def Next():

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    client_socket.send(b"\x0B")

    for source in GetSources().values():
        Delete(source)

    bytes = client_socket.recv(10)
    if bytes != b"No change":
        file_names = [f"{TIFF_ROOT}{f}" for f in os.listdir(TIFF_ROOT) if f.endswith(".tif")]
        reader = read_volume(file_names[0])

        for file_name in file_names[1:]:
            reader = read_volume(file_name)
            Hide(reader)

    ResetCamera()
    Render()


def Layer(weirdservercode):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    client_socket.send(weirdservercode)

    bytes = client_socket.recv(10)
    if bytes != b"No change":
        for source in GetSources().values():
            Delete(source)

        file_names = [f"{TIFF_ROOT}{f}" for f in os.listdir(TIFF_ROOT) if f.endswith(".tif")]
        reader = read_volume(file_names[0])

        for file_name in file_names[1:]:
            reader = read_volume(file_name)
            Hide(reader)

    ResetCamera()
    Render()


for i in range(9):
    Layer(b"\x0D")
    Save()
    Layer(b"\x0E")
    Save()
    Layer(b"\x0F")
    Save()
    Layer(b"\x10")
    Save()
    Next()

