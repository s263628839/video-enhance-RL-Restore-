import random
import tensorflow as tf
from dqn.agent import Agent
from dqn.environment import MyEnvironment
from config import get_config
import sys
import numpy as np
import cv2


import time

from ffpyplayer.pic import Image
from ffpyplayer.player import MediaPlayer
from ffpyplayer.writer import MediaWriter


# Parameters
flags = tf.app.flags
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_boolean('is_train', False, 'Whether to do training or testing')
# test
flags.DEFINE_boolean('is_save', True, 'Whether to save results')
flags.DEFINE_string('dataset', 'mine', 'Select a dataset from mild/moderate/severe')
flags.DEFINE_string('play_model', 'models/', 'Path for testing model')
# training
flags.DEFINE_string('save_dir', 'models/save/', 'Path for saving models')
flags.DEFINE_string('log_dir', 'logs/', 'Path for logs')
flags.DEFINE_string('input_name', 'input/input.mp4', 'Input video filename')
flags.DEFINE_string('output_name', 'results/output.mp4', 'Output video filename')
FLAGS = flags.FLAGS



def main(_):
    with tf.Session() as sess:
        config = get_config(FLAGS)
        env = MyEnvironment(config)
        agent = Agent(config, env, sess)

        scale = 1
        # 1. first probe file, get metadata
        in_file = config.input_name
        out_file = config.output_name
        
        convert_num = -1
        ff_opts = {
            'out_fmt': 'yuv444p',
            'framedrop': False,
            'an': True,
            'sn': True,
        }
        player = MediaPlayer(in_file, ff_opts=ff_opts)
        # must wait for probe result, strange
        while player.get_metadata()['src_vid_size'] == (0, 0):
            time.sleep(0.01)
        meta = player.get_metadata()
        width = meta['src_vid_size'][0]
        height = meta['src_vid_size'][1]
        width_out = width * scale
        height_out = height * scale
        
        out_opts = {
            'pix_fmt_in': 'yuv444p',
            'pix_fmt_out': 'yuv420p',
            'width_in': width_out,
            'height_in': height_out,
            'frame_rate': meta['frame_rate'],
            'codec': 'libx264',
            #'acpect': '4:3',
        }
        lib_opts = {
            # config for BT.2020 HDR10
            # 'x265-params': 'range=pc:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:crf=15',

            # config for x264 to encode video
            'x264-params': 'crf=15',
        }
        writer = MediaWriter(out_file, [out_opts], lib_opts=lib_opts, overwrite=True)

        frame_count = 0
        start_timestamp = 0
        while True:
            
            frame, val = player.get_frame()
            if val == 'eof':
                print('end of video')
                break
            elif frame is None:
                time.sleep(0.01)
            else:
                t1 = time.time()*1000
                img, t = frame
                if frame_count == 0:
                    start_timestamp = t
                bufs = img.to_bytearray()
                assert len(bufs) >= 3
                
                Y = np.frombuffer(bufs[0], dtype=np.uint8)
                U = np.frombuffer(bufs[1], dtype=np.uint8)
                V = np.frombuffer(bufs[2], dtype=np.uint8)

                input_YUV = cv2.merge([Y, U, V])
                img = cv2.cvtColor(input_YUV, cv2.COLOR_YUV2RGB)
                img = np.array(img).reshape(height, width, 3)
                
                outputImg = agent.test_video(img)
                
                out = np.array(outputImg).reshape(height_out*width_out, 1, 3)
                YUV = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)

                (Y, U, V) = cv2.split(YUV)

                bufs = []
                bufs.append(Y.tobytes())
                bufs.append(U.tobytes())
                bufs.append(V.tobytes())
                outputImg = Image(plane_buffers=bufs, pix_fmt='yuv444p', size=(width_out, height_out))
                t = t - start_timestamp
                writer.write_frame(img=outputImg, pts=t, stream=0)

                t2 = time.time()*1000
                frame_count += 1
                if(frame_count % 30 == 0):
                    print('convert frame # ', frame_count)
                #print('--pts:', t)
                if frame_count >= convert_num > 0:
                    break
                # if frame_count >= 1800:
                #     break
                # print("time: ", time.time()*1000-tt)

        player.close_player()
        writer.close()


if __name__ == '__main__':
    tf.app.run()