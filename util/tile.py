import torch

import numpy as np
import math

import util.const as const

__all__ = [ 'pad_image', 'generate_tile_names', 'get_tile_layout', 'get_img_name', 'get_tile', 'stitch_predictions', 'merge_preds_if_possible' ]

def remove_tile_borders(image, tile_borders):
    '''
    input:
      image: a Pytorch Variable of size (batch_size, num_channels, height, width)
      tile_border: a tuple of ints (height_border, width_border)
    output:
      image: a Pytorch Variable of size (batch_size, num_channels, height - 2 * tile_height_border, width - 2 * tile_width_border)
    '''
    tile_height_border, tile_width_border = tile_borders

    assert tile_height_border >= 0
    if tile_height_border > 0: # No need to remove border if it's 0
        image = image[:, :, tile_height_border:-tile_height_border, :]

    assert tile_width_border >= 0
    if tile_width_border > 0: # No need to remove border if it's 0
        image  =  image[:, :, :, tile_width_border:-tile_width_border]

    image = image.contiguous()
    return image

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

def remove_paddings(mask, paddings):
    '''
    input:
      mask: numpy array of shape (height, width)
      paddings: tuple of ints, (height_padding, width_padding)
    '''
    height_padding, width_padding = paddings

    assert height_padding >= 0
    if height_padding > 0: # No need to remove padding if it's 0
        mask  =  mask[height_padding:-height_padding, :]

    assert width_padding >= 0
    if width_padding > 0: # No need to remove padding if it's 0
        mask  =  mask[:, width_padding:-width_padding]

    return mask

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

def merge_preds_if_possible(tile_masks, img_rles, paddings):
    '''
    input:
      tile_masks: a dict of numpy arrays, with image tile names as keys and predicted masks as values
      img_rles: a dict of strings, with image names as keys and predicted run-length-encoded masks as values
    '''
    # TODO

    if len(tile_masks) == 0:
        return

    tile_size = tile_masks[0].shape
    padded_img_size = np.add(const.img_size, paddings)

    tile_layout, _ = get_tile_layout(tile_size, padded_img_size)
    num_of_rows, num_of_cols = tile_layout
    num_tiles = num_of_rows * num_of_cols

    # merge into whole image with shape: (1280, 1920)

    # remove paddings
    # img_mask = remove_paddings(img_mask, paddings)

    # image shape: (1280, 1918)
    # assert img_mask.shape == const.img_size

    # employ Run Length Encoding
    # preds['rle_mask']=preds['rle_mask'].apply(lambda x: run_length.encode(x))
    return

# def stitch_predictions(tile_preds):
#     '''
#     input:
#       tile_preds: a dict of numpy arrays, with image tile names as keys and predicted masks as values
#
#     output:
#       img_preds: a dict of numpy arrays, with image names as keys and predicted masks as values
#     '''
#     tile_names = tile_preds.keys()
#     tiles_by_imgs, max_tile_pos = organize_tiles(tile_names)
#
#     img_preds = {}
#     img_names = tiles_by_imgs.keys()
#     for img_name in img_names:
#         cur_img_tiles = tiles_by_imgs[img_name]
#         cur_tile_preds = create_dict_from_dict(cur_img_tiles, tile_preds)
#         img_preds[img_name] = merge_tiles(cur_tile_preds, max_tile_pos)
#
#     return img_preds

# def organize_tiles(tile_names):
#     '''
#     input:
#       tile_names: a list of strings, tile names of all images
#     output:
#       tiles_by_imgs: a dict with image names as keys and tile names of a image as values
#       max_tile_pos: a tuple of ints, which should be equivalent to tile_layout
#     '''
#
#     max_tile_pos = (-1, -1)
#     tiles_by_imgs = {}
#
#     for tile_name in tile_names:
#         img_name = get_img_name(tile_name)
#         tile_pos = get_tile_pos(tile_name)
#
#         max_tile_pos = max(tile_pos, max_tile_pos)
#
#         if img_name not in tiles_by_imgs:
#             tiles_by_imgs[img_name] = [tile_name]
#         else:
#             tiles_by_imgs[img_name].append(tile_name)
#
#     print("Max Tile Position: {}".format(max_tile_pos))
#     return tiles_by_imgs, max_tile_pos

# def create_dict_from_dict(some_keys, large_dict):
#     '''
#     from:
#     https://stackoverflow.com/questions/3420122/filter-dict-to-contain-only-certain-keys
#     '''
#     small_dict = { a_key: large_dict[a_key] for a_key in some_keys }
#     return small_dict

def merge_tiles(tile_masks, tile_layout):
    '''
    input:
      tile_masks:  a dict of numpy arrays, with tile names of a certain image as keys and their predicted masks as values
      tile_layout: a tuple of ints

    output:
      img_mask: a numpy array

    '''

    tile_names = list(tile_masks.keys())
    num_of_rows, num_of_cols = tile_layout

    assert len(tile_names) == num_of_rows * num_of_cols

    _, tile_height, tile_width = tile_masks[tile_names[0]].shape
    img_mask = np.zeros((num_of_rows * tile_height, num_of_cols * tile_width))

    for tile_name in tile_names:
        tile_row_idx, tile_col_idx = get_tile_pos(tile_name)

        start_y = (tile_row_idx - 1) * tile_height
        start_x = (tile_col_idx - 1) * tile_width

        end_y = tile_row_idx * tile_height
        end_x = tile_col_idx * tile_width

        img_mask[start_y:end_y, start_x:end_x] = tile_masks[tile_name]

    return img_mask
