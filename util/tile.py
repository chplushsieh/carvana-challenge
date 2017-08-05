
import numpy as np
import math


def get_tile_overlap(img_dim_length, tile_dim_length, num_tiles_in_dim):
    '''
    input:
      img_dim_length:  int representing either image height or width
      tile_dim_length: int representing either tile height or width
      num_tiles_in_dim: int representing how many tiles are along this dimension
    output:
      tile_dim_overlap: int representing how much tiles overlap with each other in this dimension
    '''

    dim_overlap = img_dim_length - num_tiles_in_dim * tile_dim_length
    tile_dim_overlap = dim_overlap / (num_tiles_in_dim - 1)
    assert isinstance( tile_dim_overlap, int )

    return tile_dim_overlap

def get_tile_layout(tile_size, img_size):
    '''
    input:
      tile_size: a tuple of ints (height, width) representing the size of a tile
      img_size:  a tuple of ints (height, width) representing the size of a whole image
    output:
      tile_layout:  a tuple of ints (num_of_rows, num_of_cols)
      tile_overlap: a tuple of ints (height_overlap, width_overlap) representing
                    how much neighboring tiles overlap with each other
    '''
    tile_height, tile_width = tile_size
    img_height,  img_width  = img_size

    num_of_rows = math.ceil(tile_height / img_height)
    num_of_cols = math.ceil(tile_width  / img_width)
    tile_layout = (num_of_rows, num_of_cols)

    height_overlap = get_tile_overlap(img_height, tile_height, num_of_rows)
    width_overlap  = get_tile_overlap(img_width,  tile_width,  num_of_cols)
    tile_overlap = (height_overlap, tile_overlap)

    return tile_layout, tile_overlap

def generate_tile_names(img_names, tile_size, img_size):
    '''
    input:
      img_names: a list of strings consisting of all image names
      tile_size: a tuple of ints (height, width) representing the size of a tile
      img_size:  a tuple of ints (height, width) representing the size of a whole image
    output:
      tile_names:  a list of strings consisting of all image tile names in img_name-<row_idx>-<col_idx> format
    '''
    tile_layout, _ = get_tile_layout(tile_size, img_size)
    num_of_rows, num_of_cols = tile_layout

    tile_names = []
    for img_name in img_names:
        tile_names_in_img = []
        for row_idx in range(1, num_of_rows+1):
            for col_idx in range(1, num_of_cols + 1):
                tile_name = img_name + '-' + str(row_idx) + '-' + str(col_idx)
                tile_names_in_img.append(tile_name)

        tile_names += tile_names_in_img

    return tile_names

def get_img_name(tile_name):
    '''
    get whole image name from tile name
    '''
    return tile_name.split('-')[0]

def get_tile(img, tile_name, tile_layout, tile_border): # TODO
    '''
    input:
      img:            of shape (3, height, width)
      tile_name:  <whole_img_name>_<row_idx>_<col_idx>
      tile_layout:     (num_of_rows, num_of_cols)
    '''
    tile_index = get_tile_index(tile_name)
    tile_shape = get_tile_shape(img.shape[1:], tile_layout)

    tile = crop_tile(img, tile_index, tile_shape, tile_layout, tile_border)

    return tile

########

def get_padding_shape(img_shape, zpad_shape):
    im_h,   im_w = img_shape
    z_h,   z_w = zpad_shape

    height_padding  = (math.ceil((z_h - im_h)/2), math.floor((z_h - im_h)/2))
    width_padding   = (math.ceil((z_w - im_w)/2), math.floor((z_w - im_w)/2))

    return height_padding, width_padding

def coords_in_zpad(coords, img_shape, zpad_shape):
    y, x = coords
    height_padding, width_padding = get_padding_shape(img_shape, zpad_shape)

    padded_y = y + height_padding[0]
    padded_x = x + width_padding[0]

    return padded_y, padded_x

def coords_in_which_tile(coords, tile_shape):
    tile_height, tile_width = tile_shape
    y, x = coords

    # get tile index
    row_index = math.ceil(y / tile_height)
    col_index = math.ceil(x / tile_width)
    tile_index = (row_index, col_index)
    return tile_index

def get_tile_index(img_patch_name):
    tile_row_idx, tile_col_idx = img_patch_name.split('_')[1], img_patch_name.split('_')[2]
    tile_idx = int(tile_row_idx), int(tile_col_idx)
    return tile_idx

def get_tile_shape(img_shape, tile_layout):
    num_of_rows, num_of_cols = tile_layout
    img_height, img_width = img_shape

    tile_height = int(math.floor(img_height / num_of_rows))
    tile_width = int(math.floor(img_width / num_of_cols))
    return (tile_height, tile_width)

def pad_border(img, border):

    channel_padding = (0, 0)
    height_padding  = (border, border)
    width_padding   = (border, border)

    padded_img = np.lib.pad(img, (channel_padding, height_padding, width_padding), 'constant')

    return padded_img

def crop_tile(img, tile_index, tile_shape, tile_layout, border):
    num_of_rows, num_of_cols = tile_layout
    _, img_height, img_width = img.shape
    row_idx, col_idx = tile_index
    t_h, t_w = tile_shape
    # print('Before Border Added, Tile Shape: {}'.format(tile_shape))

    y = (row_idx - 1) * t_h
    x = (col_idx - 1) * t_w

    padded_img = pad_border(img, border)

    if row_idx == num_of_rows:
        # tile in last row
        # leave all remainder to the last tile
        crop_h = img_height + 2*border
    else:
        crop_h = y + t_h + 2*border


    if col_idx == num_of_cols:
        # tile in last col
        # leave all remainder to the last tile
        crop_w = img_width + 2*border
    else:
        crop_w = x + t_w + 2*border

    cropped_img   =   padded_img[ :, y:crop_h, x:crop_w ]

    return cropped_img

def get_tile(img, img_tile_name, tile_layout, tile_border):
    '''
    Input:
    img:            of shape (3, height, width)
    img_tile_name:  <whole_img_name>_<row_idx>_<col_idx>
    tile_layout:     (num_of_rows, num_of_cols)
    '''
    tile_index = get_tile_index(img_tile_name)
    tile_shape = get_tile_shape(img.shape[1:], tile_layout)

    tile = crop_tile(img, tile_index, tile_shape, tile_layout, tile_border)

    return tile
