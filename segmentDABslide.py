import matplotlib.pyplot as plt
from omero.gateway import BlitzGateway, MapAnnotationWrapper
import numpy as np
from lavlab.omero.tiles import create_tile_list_from_image, get_tiles, create_tile_list_2d
from lavlab.python_util import create_array
from lavlab.imsuite import imresize
import lavlab
from scipy.ndimage import gaussian_filter
from skimage import filters, measure, color, morphology
from skimage.color import rgb2hsv
from tqdm import tqdm
#
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
import skimage as ski

from skimage.measure import block_reduce
from skimage.morphology import closing, square
from skimage.measure import label
import tqdm
import lavlab
from lavlab.omero.images import mask_omero_tissue_loosely
lavlab.ctx.resources.io_max_threads = 2

import omero.gateway
import numpy as np
import threading
from typing import Generator, Optional
import lavlab
from skimage.feature import blob_dog
import get_rgb_tiles

def segmentDABslide(image_id: int, output_folder: str, creds = ("bchao","100Freestyle!")):
    #INPUTS
    #creds: should be a tuple containing two strings corresponding to username and password of omero
    #image_id: should be an int of the id of the iamge you are analyzing
    #output_folder: should be a string containing the output folder for the resulting file

    #OUTPUT
    #publishes a three-prong file, which the lab has been naming i2m. Although this isn't actually in MRI space, this file contains immunohistochemistry heat map.
    #the three prongs are 1: dab stain count, 2: nuclei count, 3: ratio of dab stain count to nuclei + dab stain count.

    conn = BlitzGateway(creds[0], creds[1], host="wss://wsi.lavlab.mcw.edu/omero-wss", port=443, secure=True)
    conn.connect()
    conn.c.enableKeepAlive(60000000)

    image = conn.getObject("Image", image_id)
    if image is None:
        print(f"Image with ID {image_id} not found.")
        conn.close()
        exit()

    tile_list = create_tile_list_from_image(image, rgb=True)
    # Get image dimensions from metadata
    image_width = image.getSizeX()
    image_height = image.getSizeY()
    num_z = image.getSizeZ()
    num_c = image.getSizeC()
    num_t = image.getSizeT()

    heat_dab_count = np.zeros((int(np.ceil(image_width/1024)), int(np.ceil(image_height/1024))))
    heat_nuc_count = np.zeros((int(np.ceil(image_width/1024)), int(np.ceil(image_height/1024))))
    heat_ratio = np.zeros((int(np.ceil(image_width/1024)), int(np.ceil(image_height/1024))))

    print("Everything's looking good so far, lets segment!")
    for tile, coord in tqdm.tqdm(get_rgb_tiles.get_rgb_tiles(image, tile_list)):
        z,c,t,(x,y,width,height) = coord
        position_x, position_y, size = x//1024, y//1024, int(width * height)
        block_size = (8,8)
        down_image_8 = np.zeros((int(np.ceil(tile.shape[0]/block_size[0])), int(np.ceil(tile.shape[1]/block_size[0])),3),dtype=np.uint8)
        for i in range(3):
            down_image_8[:,:,i] = ski.measure.block_reduce(tile[:,:,i], block_size = 8, func=np.mean)
        hed_down_image_8 = rgb2hed(down_image_8)
        dab = hed_down_image_8[:,:,2]
        hem = hed_down_image_8[:,:,0]
        thresh_otsu_dab = ski.filters.threshold_otsu(dab)
        binary_otsu_dab = (dab > thresh_otsu_dab) & (dab > .1)
        thresh_otsu_hem = ski.filters.threshold_otsu(hem)
        binary_otsu_hem = hem > thresh_otsu_hem
        binary_otsu_edit = morphology.remove_small_objects(binary_otsu_dab, min_size=4)
        blobs_log = ski.feature.blob_log(hem, min_sigma=2.125, max_sigma=5, threshold=0.004)
        blobs_log[:,2] *= np.sqrt(2)
        labeled_dab = ski.measure.label(binary_otsu_edit)
        dab_count = labeled_dab.max()
        heat_dab_count[position_x,position_y] = dab_count
        heat_nuc_count[position_x, position_y] = len(blobs_log)
        heat_ratio[position_x,position_y] = dab_count/(len(blobs_log)+dab_count) if len(blobs_log)+dab_count != 0 else 0

    np.savez(output_folder, dab=heat_dab_count, nuc=heat_nuc_count, ratio=heat_ratio)

# Create the heatmap using imshow
#plt.imshow(heat_ratio, cmap='viridis')
#plt.colorbar()  # Add a colorbar for reference
#plt.title("Heatmap of DAB density Array")
#plt.show()