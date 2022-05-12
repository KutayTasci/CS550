from PIL import Image
import numpy as np

def unique_rows(a):

    return np.unique(a, axis=0)

def get_data_from_image(image):
    """
    Get the data from an image.
    """
    image_size = image.size
    arr = []
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            arr.append(image.getpixel((i, j)))
    return np.array(arr)

def calculate_clustering_error(distances):
    total_distance = 0
    for i in range(len(distances)):
        total_distance += min(distances[i])

    return total_distance / len(distances)

def generate_clustered_image(image, model, d_mean, d_std):
    im_shape = image.size
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            val = np.array([image.getpixel((i, j))])
            val = (val - d_mean) / d_std
            temp, _ = model.predict(val)
            new_val = model.centroids[np.argmax(temp)]
            new_val = new_val * d_std + d_mean
            image.putpixel((i, j), tuple(new_val.astype(int)))
    return image