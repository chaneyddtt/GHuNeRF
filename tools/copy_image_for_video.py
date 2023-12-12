import numpy as np
import argparse
import os
import shutil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)
parser.add_argument('--epoch', type=str)
parser.add_argument('--exp_folder_name', type=str)

args = parser.parse_args()

human_info_train = {
              'CoreView_313': {'begin_i': 550, 'i_intv': 1, 'ni': 250},
              'CoreView_377': {'begin_i': 0, 'i_intv': 1,
                               'ni': 500},
              'CoreView_386': {'begin_i': 80, 'i_intv': 1,
                               'ni': 450},
              'CoreView_390': {'begin_i': 400, 'i_intv': 1,
                               'ni': 500},
              'CoreView_392': {'begin_i': 0, 'i_intv': 1,
                               'ni': 400}
              }

human_info_test = {
    'CoreView_387': {'begin_i': 0, 'i_intv': 1, 'ni': 654},
    'CoreView_393': {'begin_i': 0, 'i_intv': 1, 'ni': 658},
    'CoreView_394': {'begin_i': 0, 'i_intv': 1, 'ni': 859}}

if 'testoo' in args.exp_folder_name:
    human_info = human_info_train
elif 'testxx' in args.exp_folder_name:
    human_info = human_info_test
else:
    human_info = {}

def copy_image():
    data_root = 'data/mesh/{}/epoch_{}/{}/renderings'.format(args.exp_name, args.epoch,args.exp_folder_name)
    humanid = 0
    for human in human_info:
        data_path = os.path.join(data_root, str(humanid))
        video_savepath = os.path.join(data_path, 'video_image')
        if not os.path.exists(video_savepath):
            os.makedirs(video_savepath)
        start = human_info[human]['begin_i']
        end = start + human_info[human]['ni']

        cnt = 0
        times = 0
        for poseid in np.arange(start, end, 1):
            if cnt > 90:
                cnt = 0
                times += 1
            source_image_path = os.path.join(data_path, str(poseid), str(cnt)+'.png')
            target_image_path = os.path.join(video_savepath, str(cnt + 91 * times) + '.png')
            shutil.copy(source_image_path, target_image_path)
            cnt += 1

        humanid +=1


def make_video():
    humanid = 0
    data_root = 'data/mesh/{}/epoch_{}/{}/renderings'.format(args.exp_name, args.epoch,args.exp_folder_name)
    for human in human_info:
        data_path = os.path.join(data_root, str(humanid), 'video_image')

        start = 0
        end = start + human_info[human]['ni']
        speed = 30
        img_array = []
        for poseid in np.arange(start, end, 1):
            img = cv2.imread(os.path.join(data_path, str(poseid) + '.png'))
            # img = img[::-1, ::-1]
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        humanid += 1

        out = cv2.VideoWriter(os.path.join(data_path, 'video.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), speed, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def make_video_image():
    for human in human_info:
        if human in ['CoreView_313', 'CoreView_315']:
            data_path = os.path.join('data/zju_mocap', human, 'Camera (8)')
        else:
            data_path = os.path.join('data/zju_mocap', human, 'Camera_B8')
        start = human_info[human]['begin_i']
        end = start + human_info[human]['ni']
        speed = 30
        img_array = []
        data_list = os.listdir(data_path)
        data_any = data_list[0]
        frame = data_any[:-4]
        zfill = len(frame)
        for poseid in np.arange(start, end, 1):
            frame_id = str(poseid).zfill(zfill)
            img = cv2.imread(os.path.join(data_path,  frame_id + '.jpg'))
            img = cv2.resize(img, (512, 512))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(os.path.join('data/input_videos', human + '.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), speed, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()



if __name__ == '__main__':
    # copy_image()
    make_video()
    # make_video_image()