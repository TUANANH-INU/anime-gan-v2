
import time
import cv2,os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from model import Generator


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_dir', type=str, default= 'weights/basic/face_paint_512_v2_0.pt',help='Directory name to save the checkpoints')
    parser.add_argument('--output_video_dir', type=str, default='output_video',help='Directory name of output video')
    parser.add_argument('--device', type=str.lower, default='gpu', choices=['cpu', 'gpu'], help='Running device of AnimeGANv2')
    return parser.parse_args()

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

if __name__ == '__main__':
    arg = parse_args()
    print(f"AnimeGANv2 model:  {arg.checkpoint_dir}")
    out_path = './samples/video'
    video_in = cv2.VideoCapture(0)
    # total = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = 400
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_out = cv2.VideoWriter(os.path.join(out_path, os.path.basename(arg.checkpoint_dir).rsplit('.',1)[0] + str(width) + ".avi"), fourcc, fps, (width, width))
    device = 'cuda:0' if 'gpu' == arg.device else 'cpu'

    if video_in.get(cv2.CAP_PROP_FRAME_WIDTH) >= video_in.get(cv2.CAP_PROP_FRAME_HEIGHT):
        dim_crop = (video_in.get(cv2.CAP_PROP_FRAME_HEIGHT), video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        dim_crop = (video_in.get(cv2.CAP_PROP_FRAME_WIDTH), video_in.get(cv2.CAP_PROP_FRAME_WIDTH))

    net = Generator()
    net.load_state_dict(torch.load(arg.checkpoint_dir, map_location='cpu'))
    net.to(device).eval()

    list_fps = []
    while True:
        time1 = time.time()
        ret, frame = video_in.read()
        if not ret:
            break
        cv2.imshow('img', frame)
        frame = center_crop(frame, dim_crop)
        time2 = time.time()
        frame = cv2.resize(frame, (width, width))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
        frame = torch.from_numpy(frame).cuda() 
        frame = frame.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            out = net(frame.to(torch.float16), False)
            
        out = out.squeeze(0).permute(1, 2, 0)
        out = ((out + 1.0) * 127.5).to(torch.float16).cpu()
        out = out.numpy().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        time5 = time.time()

        fps = 1/(time5-time1)
        list_fps.append(fps)
        print('fps', int(sum(list_fps)/len(list_fps)))
        
        cv2.imshow('img', out)
        video_out.write(out)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()