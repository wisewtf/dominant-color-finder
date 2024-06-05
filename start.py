import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import argparse

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def get_dominant_color(image_path, k=4):
    image = Image.open(image_path)
    image_array = np.array(image)

    pixels = image_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    cluster_centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    label_counts = Counter(labels)

    dominant_label = label_counts.most_common(1)[0][0]
    dominant_color = cluster_centers[dominant_label]

    dominant_color = tuple(map(int, dominant_color))
    dominant_color_hex = rgb_to_hex(dominant_color)

    return dominant_color, dominant_color_hex

def main():

    parser = argparse.ArgumentParser(description="Find the dominant color in a JPEG image.")
    parser.add_argument('image_path', type=str, help='Path to the JPEG image file')
    
    args = parser.parse_args()
    
    dominant_color, dominant_color_hex = get_dominant_color(args.image_path)
    print(f"Dominant color (RGB): {dominant_color}")
    print(f"Dominant color (HEX): {dominant_color_hex}")

if __name__ == "__main__":
    main()
