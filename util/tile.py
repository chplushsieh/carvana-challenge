
import numpy as np
import math

__all__ = [ 'pad_image', 'generate_tile_names', 'get_img_name', 'get_tile', 'stitch_predictions' ]

def get_tile_border(img_length, tile_length, num_tiles):
    '''
    input:
      img_length:  int representing either image height or width
      tile_length: int representing either tile height or width
      num_tiles: int representing how many tiles are along this dimension
    output:
      tile_border: int

    Note that tile_border can be 0 when img_length == tile_length

    Their relationships are:
    Eq. 1
      padded_img_length == img_length + 2 * tile_border
    Eq. 2
      padded_img_length == num_tiles * tile_length - (num_tiles - 1) * 2 * tile_border

    Eq. 3
      tile_body_length == tile_length - 2 * tile_border
    Eq. 4
      padded_img_length == num_tiles * tile_body_length + 2 * tile_border
    '''

    # To solve for tile_border, combine and reorganize Eq. 1 and Eq. 2
    tile_border = (num_tiles * tile_length - img_length) / (num_tiles * 2)

    assert tile_border >= 0 and tile_border % 1 == 0
    tile_border = int(tile_border)

    # Verify tile_border we got is right
    padded_img_length = img_length + 2 * tile_border # Eq. 1
    tile_body_length = tile_length - 2 * tile_border # Eq. 3
    assert padded_img_length == num_tiles * tile_body_length + 2 * tile_border # Eq. 4

    return tile_border

def get_tile_layout(tile_size, img_size):
    '''
    input:
      tile_size: a tuple of ints (height, width) representing the size of a tile
      img_size:  a tuple of ints (height, width) representing the size of a whole image
    output:
      tile_layout:  a tuple of ints (num_of_rows, num_of_cols)
      tile_border: a tuple of ints (height_border, width_border)
    '''
    tile_height, tile_width = tile_size
    img_height,  img_width  = img_size

    num_of_rows = math.ceil(img_height / tile_height )
    num_of_cols = math.ceil(img_width  / tile_width  )
    tile_layout = (num_of_rows, num_of_cols)

    height_border = get_tile_border(img_height, tile_height, num_of_rows)
    width_border  = get_tile_border(img_width,  tile_width,  num_of_cols)
    tile_border = (height_border, width_border)

    return tile_layout, tile_border

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

def get_tile_pos(tile_name):
    tile_row_idx, tile_col_idx = tile_name.split('-')[1], tile_name.split('-')[2]
    tile_pos = int(tile_row_idx), int(tile_col_idx)
    return tile_pos

def get_tile(img, tile_name, tile_size):
    '''
    get tile from image
    '''
    img_size = img.shape[1:]
    tile_layout, tile_border = get_tile_layout(tile_size, img_size)

    tile_pos = get_tile_pos(tile_name)

    tile = crop_tile(img, tile_pos, tile_size, tile_layout, tile_border)

    return tile

def pad_image(img, paddings):
    height_padding, width_padding = paddings

    channel_padding = (0, 0)
    height_padding  = (height_padding, height_padding)
    width_padding   = (width_padding, width_padding)
    padded_img = np.lib.pad(img, (channel_padding, height_padding, width_padding), 'constant')

    return padded_img

def crop_tile(img, tile_pos, tile_size, tile_layout, tile_border):
    '''
    crop from a image
    '''

    padded_img = pad_image(img, tile_border)

    # unpack inputs
    num_of_rows, num_of_cols = tile_layout
    _, img_height, img_width = img.shape
    row_idx, col_idx = tile_pos
    tile_h, tile_w = tile_size
    height_border, width_border = tile_border

    t_body_h, t_body_w = tile_h - 2 * height_border, tile_w - 2 * width_border

    # print('Tile Size: {}, {}'.format(tile_h, tile_w))
    # print('Tile Border: {}, {}'.format(height_border, width_border))
    # print('Tile Body: {}, {}'.format(t_body_h, t_body_w))

    crop_y_start = (row_idx - 1) * t_body_h
    crop_x_start = (col_idx - 1) * t_body_w

    if row_idx == num_of_rows: # if the tile is in last row
        # leave all remainder to the last tile
        crop_y_end = img_height + 2*height_border
    else:
        crop_y_end = crop_y_start + t_body_h + 2*height_border


    if col_idx == num_of_cols: # if the tile is in last col
        # leave all remainder to the last tile
        crop_x_end = img_width + 2*width_border
    else:
        crop_x_end = crop_x_start + t_body_w + 2*width_border

    cropped_img   =   padded_img[ :, crop_y_start:crop_y_end, crop_x_start:crop_x_end ]

    return cropped_img

def stitch_predictions(tile_preds):
    '''
    input:
      tile_preds: a dict of numpy arrays, with image tile names as keys and predicted masks as values

    output:
      img_preds: a dict of numpy arrays, with image names as keys and predicted masks as values
    '''
    img_preds = None
    # TODO
    return img_preds
