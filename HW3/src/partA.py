from PIL import Image
from custom_utils import *
from KMeans import *
import time


image = Image.open('arc.jpg')

data = get_data_from_image(image)
d_mean = data.mean()
d_std = data.std()
data = (data - data.mean()) / data.std()
data = unique_rows(data)
ts_1 = time.perf_counter()
model = K_Means(k=8)
model.fit(data)

ts_2 = time.perf_counter()

labels, distances = model.predict(data)

ts_3 = time.perf_counter()

clustering_error = calculate_clustering_error(distances)

print('Clustering error (Avg over all pixels):', clustering_error)
print('Time taken to train:', ts_2 - ts_1)
print('Time taken to predict:', ts_3 - ts_2)

new_img = generate_clustered_image(image,model,d_mean,d_std)
new_img.show()
new_img.save("clustered_image.jpg")


