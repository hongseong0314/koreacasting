import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import sys
from glob import glob
import tensorflow as tf

files = glob(os.path.join(os.getcwd(), 'data/mask/*.png'))

def full_mask(verbose=False):
    """
    mask 합
    """
    global files
    mask_list = [np.array(Image.open(path), dtype=np.uint8) for path in files]

    for i, mask in enumerate(mask_list):
        if i == 0:
            full_masks = mask
        else:
            full_masks = np.logical_or(full_masks, mask)

    if verbose:
        plt.figure(figsize=(6,6))
        plt.imshow(full_masks)
        plt.show()
    
    del mask_list
    return full_masks


def sort_files(files):
    """
    날짜 순으로 정렬
    """
    df = pd.DataFrame()
    df['path'] = files
    
    df['y'] = df['path'].apply(lambda x : x.split("\\")[-1][4:8]) # year
    df['m'] = df['path'].apply(lambda x : x.split("\\")[-1][8 :+ 8 + int(len(files[0].split("\\")[-1][8:-8]) / 2.)]) # month
    df['d'] = df['path'].apply(lambda x : x.split("\\")[-1][8 + int(len(files[0].split("\\")[-1][8:-8]) / 2.):-8]) # day
    df['hour'] = df['path'].apply(lambda x : x.split("\\")[-1][-8:-6]) # hour
    df['min'] = df['path'].apply(lambda x : x.split("\\")[-1][-6:-4]) # min
    
    df = df.sort_values(by=['y', 'm', 'd', 'hour', 'min']).reset_index(drop=True)
    return df.path.tolist()


def files_detail(files):
    """
    날짜 순으로 정렬
    """
    import pandas as pd
    df = pd.DataFrame()
    df['path'] = files
    
    df['y'] = df['path'].apply(lambda x : x.split("\\")[-1][4:8]) # year
    df['m'] = df['path'].apply(lambda x : x.split("\\")[-1][8 :+ 8 + int(len(files[0].split("\\")[-1][8:-8]) / 2.)]) # month
    df['d'] = df['path'].apply(lambda x : x.split("\\")[-1][8 + int(len(files[0].split("\\")[-1][8:-8]) / 2.):-8]) # day
    df['hour'] = df['path'].apply(lambda x : x.split("\\")[-1][-8:-6]) # hour
    df['min'] = df['path'].apply(lambda x : x.split("\\")[-1][-6:-4]) # min
    
    df = df.sort_values(by=['y', 'm', 'd', 'hour', 'min']).reset_index(drop=True)
    df['d_min'] = df['min'].shift(periods=1, axis=0, fill_value=0)
    
    df['sub'] = df['min'].astype(np.int64) - df['d_min'].astype(np.int64)
    df['sub'][0] = 5
    
    dup_index = df.loc[df['sub'] == -60].index.tolist()
    df = df.drop(dup_index, axis=0).reset_index(drop=True)
    df['delay'] = df['sub'].apply(lambda x : 1 if x == 5 or x == -55 else 0)
    
    # 빈공간 샘플 버리기
    w, h = (675, 660)
    comparison_img = np.array(Image.open(os.path.join(os.getcwd(), 'data/train/test2022431445.png')).crop((0, 30, w, h-30)))
    df['empty_ch'] = [1 if np.sum(np.abs(np.array(Image.open(path).crop((0, 30, w, h-30))) - comparison_img)) >= 14000 else 0 for path in df['path']]
    return df

def make_windowed_dataset(dataset, start_index, end_index, time_step, step):
    data = []
    start_index = start_index + time_step
    
    if end_index is None:
        end_index = len(dataset) 
    
    for i in range(start_index, end_index):
        indexs = range(i - time_step, i, step)
        sub_data = dataset.loc[indexs]
        empty_sum = np.sum(sub_data['empty_ch'])
        delay_sum = np.sum(sub_data['delay'])
        if empty_sum > 14 and delay_sum == time_step:
            data.append(sub_data.loc[indexs, 'path'].tolist())
    return data


def path2img(df):
    """
    image load 후 위성 관찰 지역만 mask 한 이미지와 map img 반환
    """
    masks = full_mask()
    sample_data = np.array([np.array(Image.open(path).resize((256,256))) * np.array(Image.fromarray(masks).resize((256,256))) for path in df])

    # map 삭제
    imgs = [img * np.array(img != 34, np.float32) for img in sample_data]

    map_point = np.array(sample_data[0] == 34, np.float32).nonzero()
    map_img = np.zeros((256,256))
    for i, pos in enumerate(zip(map_point[0], map_point[1])):
        map_img[pos[0], pos[1]] += 255.

    return tf.convert_to_tensor(imgs, dtype=tf.float32, dtype_hint=None, name=None), map_img

def decodermap(result, map_img):
    """
    img + map_img 
    """
    return np.array([G_img + map_img[..., np.newaxis] for G_img in result.numpy()])


def error_l2(result, target):
    from sklearn.metrics import mean_squared_error
    """
    예측한 이미지와 real이미지를 L2로 비교한다.
    """
    # 감가 배열 뒤에 이미지의 에러가 더 높은 오차를 갖는다.    
    time_step_decay = np.logspace(-0.5, 0, int(90 * 0.1))
    error = [mean_squared_error(result[i, ..., 0], target[i, ..., 0]) for i in range(9)]
    return np.array(error), np.array(error) * time_step_decay

def cosin_metric(x1, x2):
  return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosin_acc(result, target):

    # 감가 배열 뒤에 이미지의 에러가 더 높은 오차를 갖는다.    
    time_step_decay = np.logspace(-0.5, 0, int(90 * 0.1))
    acc = [cosin_metric(result[i].numpy().flatten(), 
                                           target[i].numpy().flatten()) for i in range(9)]

    return np.array(acc), np.array(acc) * time_step_decay


import matplotlib
from matplotlib import animation

matplotlib.rc('animation', html='jshtml')

def plot_animation(field, figsize=None,
                   vmin=0, vmax=10, cmap="jet", **imshow_args):
  fig = plt.figure(figsize=figsize)
  ax = plt.axes()
  ax.set_axis_off()
  plt.close() # Prevents extra axes being plotted below animation
  img = ax.imshow(field[0, ..., 0], vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)

  def animate(i):
    img.set_data(field[i, ..., 0])
    return (img,)

  return animation.FuncAnimation(
      fig, animate, frames=field.shape[0], interval=24, blit=False)


