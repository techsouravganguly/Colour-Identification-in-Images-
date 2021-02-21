from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab,deltaE_cie76
import os
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]),int(color[1]),int(color[2]))
def get_image(image_path):
    image = cv2.imread(image_path)
    image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
def get_colors(image,number_of_colors,show_chart):
    modified_image = cv2.resize(image,(600,400),interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1],3)
    clf  = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    if(show_chart):
        plt.figure(figsize = (8,6))
        plt.pie(counts.values(), labels = hex_colors,colors = hex_colors)
    return rgb_colors

def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    selected =[]
    for i in range(len(images)):
        selected.append(match_image_by_color(images[i], color, threshold, colors_to_match))
    return selected
def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image
image_directory = 'images'
COLORS = {
    'GREEN':[0,128,0],
    'YELLOW':[255,255,0],
    'BROWN': [165,42,42]
    }
images = []
image = os.listdir(image_directory)
for file in image:
    if not file.startswith('.'):
        images.append(get_image(os.path.join(image_directory,file)))
for i in COLORS:
    path = os.path.join(image_directory,i)
    os.mkdir(path)
    selected = show_selected_images(images, COLORS[i], 60, 5)
    for i in range(0,len(image)):
        if selected[i]:
            print(image[i])
            image1  = cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path,image[i]
                                     ),image1)
