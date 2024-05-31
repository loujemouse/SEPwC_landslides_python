import rasterio
dataset = rasterio.open('test/data/raster_template.tif')
print(dataset.count)
number1 = dataset.transform * (0,0)
print(number1)
number2 = dataset.transform * (dataset.width, dataset.height)
print(number2)


dataset.indexes
band1 = dataset.read(dataset.indexes)
print("\n")
print(band1)

x, y = (dataset.bounds.left + 100000, dataset.bounds.top - 50000)

row, col = dataset.index(x, y)
print("\n")
print("\n")
print(row, col)

print("\n")
print("\n")
print(dataset.xy(dataset.height // 2, dataset.width // 2))
