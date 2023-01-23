import gdown

with open("wigths/links.txt") as links:
    links = links.read()
    for string in links.split("\n"):
        name, link = string.split()
        gdown.download(link, f"/wigths/{name}.pth")

