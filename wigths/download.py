import gdown
import os

d = os.path.dirname(__file__)

with open(f"{d}/links.txt") as links:
    links = links.read()
    for string in links.split("\n"):
        name, link = string.split()
        gdown.download(link, f"{d}/{name}.pth")