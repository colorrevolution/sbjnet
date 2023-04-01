from PIL import Image

path = "label3/008_i_t.png"
image = Image.open(path).convert("RGB")
image.save(path)