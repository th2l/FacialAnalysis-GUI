# Author: Van-Thong Huynh
# Dept. of AIC, Chonnam National University
# Last modified: Nov 2021

import datetime
import time
from collections import deque
import os, sys, threading, typing

import pandas as pd
import validators
from PyQt6.QtGui import QIcon
from PyQt6.QtWebEngineCore import QWebEngineDownloadRequest
from dash.exceptions import PreventUpdate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dash components
import cv2
import insightface.model_zoo
import numpy as np
from dash import Dash, dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from flask import Response, stream_with_context
import dash_bootstrap_components as dbc

from PyQt6 import QtCore, QtWidgets, QtWebEngineWidgets

# Python Requests with REST APIs, download files
import base64
import requests

from pandas import DataFrame

import insightface
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob2
from tqdm import tqdm
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage import transform

from multiprocessing import Process, Pipe
import onnxruntime as ort

print('ONNX devices: ', ort.get_device())
# Zoom API Information
zoom_client_ID = os.getenv('ZOOM_CLIENT_ID')
zoom_client_secret = os.getenv('ZOOM_CLIENT_SECRET')
zoom_oauth_redirect_uri = os.getenv('ZOOM_REDIRECT_URI')
zoom_userID = 'me'

print(zoom_client_ID, zoom_client_secret, zoom_oauth_redirect_uri)
## Diskcache
import diskcache

MAX_SIZE_QUEUE = None


class VideoCamera(QtCore.QObject):
    # Based on https://community.plotly.com/t/does-dash-support-opencv-video-from-webcam/11012/2
    def __init__(self, parent=None):
        super(VideoCamera, self).__init__(parent)
        self.video = None
        self.running = False
        self.capture_save = ''

    def __del__(self):
        self.running = False
        self.video.release()

    @QtCore.pyqtSlot(str)
    def capture_current_img(self, img_id):
        # print('Receive signal capture ', img_id)
        self.capture_save = img_id

    @QtCore.pyqtSlot(bool)
    def trigger_camera(self, run_camera):
        if run_camera:
            print('Turn on camera')
            self.running = True
        else:
            print('Turn off camera')
            self.running = False

    def get_frame(self):
        if not self.running:
            if self.video is not None:
                self.video.release()
                self.video = None
            return None
        else:
            if self.video is None:
                self.video = cv2.VideoCapture(0)

        success, image = self.video.read()
        if self.capture_save != '':
            # Save image
            save_path = './assets/students/{}/{}.jpg'.format(self.capture_save,
                                                             datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
            # print(self.capture_save, save_path)
            cv2.imwrite(save_path, image)
            self.capture_save = ''
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()


def gen_camera(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class VideoCaptureThread:
    def __init__(self, src='0', threaded=True):
        if src == '0':
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        print('Video src: ', src, self.cap.isOpened())
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.stopped = False
        self.frame_queue = deque(maxlen=MAX_SIZE_QUEUE)
        self.threaded = threaded

    def start(self):
        threading.Thread(target=self.get_frame, args=()).start()

    def get_frame(self):

        while not self.stopped:
            try:
                self.cont, frame = self.cap.read()
                if not self.cont:
                    self.stop()
                    break
                else:
                    self.frame_queue.append(frame)
            except:
                break

    def stop(self):
        self.stopped = True
        self.cap.release()


class VideoWriterThread:
    def __init__(self, src, codec, fps=30, size=(640, 480)):
        self.capW = cv2.VideoWriter(src, codec, fps, size)
        self.stopped = False
        self.frame_queue = deque(maxlen=MAX_SIZE_QUEUE)
        self.student_queue = []
        self.thread = threading.Thread(target=self.write_frame, args=())
        self.count_num = 0

    @staticmethod
    def draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh):
        cur_students = []
        for dist, label, bb, cc in zip(rec_dist, rec_class, bbs, ccs):
            # Red color for unknown, green for Recognized
            color = (0, 0, 255) if dist < dist_thresh else (0, 255, 0)
            label = "Unknown" if dist < dist_thresh else label

            left, up, right, down = bb
            cv2.line(frame, (left, up), (right, up), color, 3, cv2.LINE_AA)
            cv2.line(frame, (right, up), (right, down), color, 3, cv2.LINE_AA)
            cv2.line(frame, (right, down), (left, down), color, 3, cv2.LINE_AA)
            cv2.line(frame, (left, down), (left, up), color, 3, cv2.LINE_AA)

            xx, yy = np.max([bb[0] - 10, 10]), np.max([bb[1] - 10, 10])
            # cv2.putText(frame, "Name: {}, dist: {:.4f}".format(label, dist), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #             color, 2)
            cv2.putText(frame, "{}".format(label), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cur_students.append(label)

        return frame, np.unique(cur_students)

    def write_frame(self):

        while True:
            while True:
                try:
                    st_wr = time.time()
                    frame, rec_dist, rec_class, bbs, ccs, dist_thresh = self.frame_queue.popleft()
                    _, cur_students = self.draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh)
                    self.capW.write(frame)

                    # print('Time in write: ', time.time() - st_wr)
                    self.student_queue.append(cur_students)
                    self.count_num += 1
                    # if self.count_num % 300 == 0:
                    #     print('Current write index ', self.count_num, time.time() - st_wr)
                except:
                    if len(self.frame_queue) == 0 and self.stopped:
                        self.stop()
                        break

            if len(self.frame_queue) == 0 and self.stopped:
                self.stop()
                break

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.capW.release()


class FaceAnalysis(QtCore.QObject):
    # Face recognition model code are adapted from
    # https://github.com/leondgarse/Keras_insightface/blob/master/video_test.py
    finished = QtCore.pyqtSignal(str)
    trigger_video_process_signal = QtCore.pyqtSignal(str, int, float)

    def __init__(self, parent=None):
        super(FaceAnalysis, self).__init__(parent)

        self.init_gpu = False
        self.det = None
        self.face_reg = None
        self.embeddings = None
        self.image_classes = None

        self.video_src = None
        self.second_per_detect = None
        self.dist_thresh = None
        self.do_video_analysis = False

        self.shared_data_parent, self.shared_data_child = Pipe()
        self.is_running_parent, self.is_running_child = Pipe()
        self.stop_running_parent, self.stop_running_child = Pipe()
        self.trigger_init_embedding_parent, self.trigger_init_embedding_child = Pipe()
        self.shared_data_result_parent, self.shared_data_result_child = Pipe()
        self.trigger_stop_running_parent, self.trigger_stop_running_child = Pipe()
        self.is_running = False

    def gen_frame_webcam(self):
        while True:
            if not self.shared_data_result_parent.poll(0.0002):
                continue
            frame_ret = self.shared_data_result_parent.recv()
            ret, jpeg = cv2.imencode('.jpg', frame_ret)
            jpeg_send = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_send + b'\r\n\r\n')

    def init_face_engine(self):
        if not self.init_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
            self.init_gpu = True

        if self.det is None:
            print('Re-initialize')
            self.det = insightface.model_zoo.SCRFD('./assets/face_analysis/face_reg/scrfd_10g_bnkps.onnx',
                                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.det.prepare(0)

        if self.face_reg is None:
            self.face_reg = load_model('./assets/face_analysis/face_reg/mobilenet_emb256.h5', compile=False)

    @QtCore.pyqtSlot()
    def re_init_face_embedding(self):
        self.trigger_init_embedding_parent.send('True')

    @QtCore.pyqtSlot()
    def stop_running(self):
        self.trigger_stop_running_parent.send('True')

    def start_video_analysis(self, vd_src, second_per_detect=5, dist_thresh=0.6, in_minutes=0, cls_name_students=()):
        print('Receive signals: ', vd_src, second_per_detect, dist_thresh)
        if self.is_running:
            raise ValueError('Current analysis is in progress')
        else:
            self.shared_data_parent.send([vd_src, second_per_detect, dist_thresh, in_minutes, cls_name_students])
            print('Send signals to child process')
            self.is_running = True
            while not self.stop_running_parent.poll(0.1):
                continue
            attendance_df = self.stop_running_parent.recv()
            self.is_running = False
            # print('**** Recieved finish signal. Finished running. ', attendance_df)
            self.stop_running_parent.send('OK')
            return attendance_df

    def init_embedding_images(self, re_run=False):
        emb_path = './assets/students/students_emb.npz'
        batch_size = 32
        if os.path.isfile(emb_path) and not re_run:
            npz = np.load(emb_path)
            self.image_classes, self.embeddings = npz['image_classes'], npz['embeddings']
        else:
            image_names = sorted(glob2.glob(os.path.join("./assets/students/*/*.jpg")) + glob2.glob(
                os.path.join("./assets/students/*/*.png")) + glob2.glob(os.path.join("./assets/students/*/*.jpeg")))

            """ Detect faces in images, keep only those have exactly one face. """
            nimgs, image_classes = [], []
            for image_name in tqdm(image_names, "Detect"):
                img = imread(image_name)
                nimg = self.do_detect_in_image(img, image_format="RGB")[-1]
                if nimg.shape[0] > 0:
                    nimgs.append(nimg[0])
                    image_classes.append(os.path.basename(os.path.dirname(image_name)))

            """ Extract embedding info from aligned face images """
            steps = int(np.ceil(len(image_classes) / batch_size))
            nimgs = (np.array(nimgs) - 127.5) * 0.0078125
            embeddings = [self.face_reg(nimgs[ii * batch_size: (ii + 1) * batch_size]) for ii in
                          tqdm(range(steps), "Embedding")]

            self.embeddings = normalize(np.concatenate(embeddings, axis=0))
            self.image_classes = np.array(image_classes)
            np.savez_compressed(emb_path, embeddings=self.embeddings, image_classes=self.image_classes)

        print(">>>> image_classes info:")
        print(pd.value_counts(self.image_classes))

    def do_detect_in_image(self, image, image_format="BGR"):
        imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
        imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
        bboxes, pps = self.det.detect(imm_BGR, (640, 640))
        nimgs = self.face_align_landmarks_sk(imm_RGB, pps)
        bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
        return bbs, ccs, nimgs

    def face_align_landmarks_sk(self, img, landmarks, image_size=(112, 112), method="similar"):
        tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
            dtype=np.float32)
        ret = []
        for landmark in landmarks:
            # landmark = np.array(landmark).reshape(2, 5)[::-1].T
            tform.estimate(landmark, src)
            ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
        return (np.array(ret) * 255).astype(np.uint8)

    def image_recognize(self, frame, image_format='BGR'):
        bbs, ccs, nimgs = self.do_detect_in_image(frame, image_format=image_format)
        if len(bbs) == 0:
            return [], [], [], []

        emb_unk = self.face_reg((nimgs - 127.5) * 0.0078125).numpy()
        emb_unk = normalize(emb_unk)
        dists = np.dot(self.embeddings, emb_unk.T).T
        rec_idx = dists.argmax(-1)
        rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
        rec_class = [self.image_classes[ii] for ii in rec_idx]

        return rec_dist, rec_class, bbs, ccs

    def video_recognize(self, video_src='0', second_per_detect=5, dist_thresh=0.6):
        st_vr = time.time()

        if isinstance(video_src, tuple):
            write_video_src = video_src[1]
            video_src = int(video_src[0])
        else:
            write_video_src = video_src.replace('.mp4', '_result.mp4')

        cap_thread = VideoCaptureThread(video_src)

        width = cap_thread.width
        height = cap_thread.height

        cap_result_thread = VideoWriterThread(write_video_src, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

        cur_frame_idx = 0
        cap_thread.start()
        # cap_result_thread.start()

        print('Video src: ', video_src, cap_thread.cap.isOpened())
        frames_per_detect = int(second_per_detect * cap_thread.fps)  # cap_thread.fps
        print('Frame per detect: ', frames_per_detect, cap_thread.fps)
        while True:
            if isinstance(video_src, int) and self.trigger_stop_running_child.poll(0.0001):
                cap_thread.stop()
                self.trigger_stop_running_child.recv()
                break
            try:
                frame = cap_thread.frame_queue.popleft()

                if cur_frame_idx % frames_per_detect == 0:
                    st_deep = time.time()
                    rec_dist, rec_class, bbs, ccs = self.image_recognize(frame)
                    print('Time in deep: ', time.time() - st_deep)
                _, cur_students = cap_result_thread.draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh)
                cap_result_thread.student_queue.append(cur_students)
                if isinstance(video_src, int):
                    self.shared_data_result_child.send(frame)
                cap_result_thread.capW.write(frame)

                cur_frame_idx += 1
            except:
                if len(cap_thread.frame_queue) == 0 and cap_thread.stopped:
                    break

        cap_result_thread.stop()
        cap_thread.stop()
        cap_result_thread.capW.release()
        cap_thread.cap.release()

        os.system("ffmpeg -y -i {} -vcodec libx264 {}".format(write_video_src,
                                                              write_video_src.replace('_result.mp4', '_analyzed.mp4')))
        os.system("rm {}".format(write_video_src))

        cv2.destroyAllWindows()
        print('Time in while: ', time.time() - st_vr)

        use_index = np.arange(0, cur_frame_idx, frames_per_detect)
        student_attendance = np.array(cap_result_thread.student_queue, dtype=object)[use_index]

        return list(student_attendance), use_index // cap_thread.fps

    def start(self):
        p = Process(target=self.run, args=(), daemon=True)
        p.start()

    def run(self):

        print('Running face analysis engine')
        while True:
            self.init_face_engine()
            if self.embeddings is None and self.image_classes is None:
                self.init_embedding_images(re_run=False)

            if self.trigger_init_embedding_child.poll(0.0002):
                # Re-init embedding
                tmp = self.trigger_init_embedding_child.recv()
                self.init_embedding_images(re_run=True)

            if not self.shared_data_child.poll(0.0002):
                continue

            video_src, second_per_detect, dist_thresh, in_minutes, students_list = self.shared_data_child.recv()
            print('Receive data: ', video_src, second_per_detect, dist_thresh, in_minutes)
            self.do_video_analysis = False

            st = time.time()
            results, use_index = self.video_recognize(video_src, second_per_detect, dist_thresh)
            print('Total time: ', time.time() - st)

            if in_minutes:
                post = '(sec)'
            else:
                post = '(min)'
            attendance_df = {'Timestamp {}'.format(post): use_index // (in_minutes * 59 + 1)}
            attendance_df.update({x: np.zeros(len(results), dtype=int) for x in students_list})
            for timestamp in range(len(results)):
                cur_students = results[timestamp]
                for stu in cur_students:
                    if stu in students_list:
                        attendance_df[stu][timestamp] = 1

            attendance_df = DataFrame.from_dict(attendance_df)

            self.stop_running_child.send(attendance_df)
            # print('Send results signal to parent')


class QDash(QtCore.QObject):
    zoom_auth_signal = QtCore.pyqtSignal()
    zoom_off_signal = QtCore.pyqtSignal()
    zoom_uuid_analyze_signal = QtCore.pyqtSignal(list)
    zoom_process_video_signal = QtCore.pyqtSignal(str, int, float)
    capture_webcam_signal = QtCore.pyqtSignal(str)
    trigger_webcam_signal = QtCore.pyqtSignal(bool)
    trigger_re_init_embedding_signal = QtCore.pyqtSignal()
    zoom_stop_analyze_signal = QtCore.pyqtSignal()

    def __init__(self, parent: typing.Optional['QObject'] = None) -> None:
        super(QDash, self).__init__(parent=parent)
        # launch_uid = uuid4()
        cache = diskcache.Cache("./cache")
        long_callback_manager = DiskcacheLongCallbackManager(cache, expire=3600)

        self.__app = Dash(suppress_callback_exceptions=True, assets_folder='assets',
                          long_callback_manager=long_callback_manager,
                          external_stylesheets=[dbc.themes.BOOTSTRAP],
                          title='PRLAB - Facial Analysis', )

        self.oauth_code = ''
        self.access_token = ''
        self.face_engine = FaceAnalysis()
        self.zoom_process_video_signal.connect(self.face_engine.start_video_analysis)
        self.trigger_re_init_embedding_signal.connect(self.face_engine.re_init_face_embedding)
        self.zoom_stop_analyze_signal.connect(self.face_engine.stop_running)

        self.webcam_object = VideoCamera()
        self.capture_webcam_signal.connect(self.webcam_object.capture_current_img)
        self.trigger_webcam_signal.connect(self.webcam_object.trigger_camera)

        self.face_engine.start()
        self.update_database = False
        self.zoom_webcam = False

        select_record_dropdown = dbc.DropdownMenu([
            dbc.DropdownMenuItem("Zoom Cloud", id='zoom-cloud', n_clicks=0, external_link=True,
                                 href="https://zoom.us/oauth/authorize?response_type=code&client_id={}&redirect_uri={}".format(
                                     zoom_client_ID, zoom_oauth_redirect_uri)),
            dbc.DropdownMenuItem(dcc.Upload('Local Recorded', id='zoom-local', multiple=True), n_clicks=0),
            dbc.DropdownMenuItem('Webcam', id='zoom-webcam', n_clicks=0),
        ],
            label="Select Records")

        zoom_webcam_analyzed_modal = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Zoom webcam analyze'), close_button=False),
            dbc.ModalBody([dbc.Container(
                dbc.Row([dbc.Col(html.Img(src="/video_feed_analyzed", id='webcam-analyzed'), align='center'), ],
                        style={'textAlign': 'center'}), )]),
            dbc.ModalFooter(
                dbc.Button("Finish", id='do-zoom-webcam-finish', class_name='ms-auto', n_clicks=0),
                class_name='align-self-center'
            )
        ], backdrop="static", keyboard=False, id='zoom-webcam-analyzed-modal', is_open=False, size='xl', centered=True)

        analyze_button = dbc.Button('Analyze', id="analyze-btn", className="me-1", n_clicks=0),
        analyze_options = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Analyze options')),
            # Body
            dbc.ModalBody([
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.Span("Interval"), width=2, align='center'),
                        dbc.Col(dbc.Input(type="number", min=0.1, step=0.05, value=1, id='analyze-interval'), width=3),
                        dbc.Col(dbc.Select(id='interval-type',
                                           options=[{'label': 'seconds', 'value': 0}, {'label': 'minutes', 'value': 1}],
                                           value=0), width='auto')
                    ]),

                    dbc.Row([
                        dbc.Col(html.Span("Recognition Sensitive"), width='auto', align='center'),
                        dbc.Col(dbc.Input(type="number", min=0.1, max=0.99, value=0.5, step=0.05,
                                          id='recognition-sensitive'),
                                width=3),
                    ]),

                    dbc.Row([
                        dbc.Col(html.Span("Face analysis options"), width='auto', align='center'),
                    ]),

                    dbc.Row([
                        dbc.Col(dbc.Checklist(options=[{'label': 'Attendance', 'value': 1},
                                                       {'label': 'Emotion', 'value': 2, 'disabled': False},
                                                       {'label': 'Engagement', 'value': 3, 'disabled': False}],
                                              id='face-analyze-options', value=[1], switch=True),
                                width={'offset': 1})
                    ]),

                    dbc.Row([
                        dbc.Col(html.Span("Select class"), width='auto', align='center'),
                        dbc.Col(dbc.Select(id='classes-list', options=[], ), width='auto')
                    ])
                ],
                    class_name='container-fluid overflow-hidden d-grid gap-3',
                )
            ]),
            # Footer
            dbc.ModalFooter(
                dbc.Button("OK", id='do-analyze-btn', class_name='ms-auto', n_clicks=0)
            )
        ],
            id='analyze-options',
            scrollable=True,
            is_open=False
        )
        meeting_list = dash_table.DataTable(id='meeting-lists',
                                            columns=[
                                                {"name": i, "id": i} for i in
                                                ['ID', 'Topics', 'Start time', 'DL Link', 'Play Link']
                                            ],
                                            data=[],
                                            row_selectable='single',
                                            fill_width=True,
                                            style_cell={
                                                'textAlign': 'left', 'height': 'auto',
                                            },
                                            style_cell_conditional=[
                                                {'if': {'column_id': 'ID'}, 'display': 'none'},
                                                {'if': {'column_id': 'DL Link'}, 'display': 'none'},
                                                {'if': {'column_id': 'Play Link'}, 'display': 'none'},
                                            ],
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgb(220, 220, 220)',
                                                }
                                            ],
                                            style_table={'overflowX': 'auto'}
                                            )
        meeting_player = html.Video(id='meeting-player', controls=True, autoPlay=True, src='', width='100%',
                                    height='auto')

        attendance_check_results = dash_table.DataTable(id='attendance-results',
                                                        columns=[
                                                            {"name": i, "id": i} for i in
                                                            ['Timestamp (sec)', 'STUD 1', 'STUD 2', 'STUD 3', 'STUD 4']
                                                        ],
                                                        page_size=15,
                                                        data=[
                                                            {'Timestamp (sec)': 'NaN', 'STUD 1': 'NaN', 'STUD 2': 'NaN',
                                                             'STUD 3': 'NaN', 'STUD 4': 'NaN'}],
                                                        row_selectable='multi',
                                                        style_cell={
                                                            'textAlign': 'center', 'height': 'auto',
                                                            'minWidth': '80px', 'width': '110px', 'maxWidth': '150px',
                                                            'whiteSpace': 'normal'
                                                        },
                                                        style_cell_conditional=[
                                                            {'if': {'column_id': 'Emotions'}, 'display': 'none'},
                                                            {'if': {'column_id': 'Attention'}, 'display': 'none'}],
                                                        style_data_conditional=[
                                                            {
                                                                'if': {'row_index': 'odd'},
                                                                'backgroundColor': 'rgb(220, 220, 220)',
                                                            }
                                                        ],
                                                        style_header={
                                                            'fontWeight': 'bold', 'height': '50px',
                                                        },
                                                        fixed_rows={'headers': True},
                                                        fixed_columns={'headers': True, 'data': 1},
                                                        style_table={'overflowX': 'auto', 'overflowY': 'auto',
                                                                     'height': 400, 'minWidth': '100%'},
                                                        )
        attendance_check_graphs = dcc.Graph(figure={
            'data': [],
            'layout': {
                'title': 'Data Visualization'
            }
        },
            id='attendance_graphs'
        )
        webcam_modal = dbc.Container([dbc.Row(
            [dbc.Col(html.Img(src="/video_feed", id='webcam-jpeg'), align='center'), ], style={'textAlign': 'center'}),
            dbc.Row([dbc.Col(dbc.Button('Take picture', id='take-picture', n_clicks=0),
                             width='auto'),
                     dbc.Col(dbc.Button('Close', id='close-webcam-take-picture-btn',
                                        class_name='ms-auto', n_clicks=0), width='auto')],
                    justify='center')],
            fluid=True, class_name='justify-content-center d-grid gap-4')
        # webcam_modal = dbc.Col(html.Img(src="/video_feed"), align='center', style={'textAlign': 'center'})

        student_database_modal_body = dbc.Container([dbc.Row([
            dbc.Col(html.Span('Student ID'), width='auto', align='center'),
            dbc.Col(dcc.Dropdown(id='student-list-db',
                                 placeholder='Select an ID'),
                    width='3')]),
            dbc.Row([dbc.DropdownMenu([
                dbc.DropdownMenuItem("From Webcam", id='add-photo-webcam', n_clicks=0),
                dbc.DropdownMenuItem(dcc.Upload('Upload', id='add-photo-upload', multiple=True), n_clicks=0),
            ],
                label="Add Photo", id='add-photo-btn')]),
            dbc.Row(webcam_modal, justify='center', align='center', class_name='g-2', id='webcam-modal',
                    style={'display': 'none'}),
            dbc.Row(id='student-photos', ),

        ], class_name='d-grid gap-4')

        student_database_modal = dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Student Database", class_name='ms-auto')),
                                            dbc.ModalBody(student_database_modal_body),
                                            dbc.ModalFooter(
                                                dbc.Button('Close', id='close-std-db-modal-btn', class_name='ms-auto',
                                                           n_clicks=0))
                                            ], id='student-database-modal', is_open=False, fullscreen=True)

        self.app.layout = dbc.Container([
            dbc.Row(style={'height': 2}),
            dbc.Row([
                dbc.Col(dbc.Row([dbc.Col(select_record_dropdown, width='auto'),
                                 dbc.Col(dbc.Button('Student Database', id='std-db-btn', n_clicks=0), align='center',
                                         width='auto'), dbc.Col(student_database_modal, width='auto')]), width=6),
                dbc.Col(dbc.Row([dbc.Col(dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Log out", id='zoom-cloud-logout')
                ],
                    label='Not Sign in', id='user_btn',
                ), width='auto')], justify='end'), width=6, align='center'), ],
                class_name='g-1',
                style={'border-bottom': '1px solid #ccc', 'margin': '2'}
            ),

            dbc.Row([
                dbc.Col(dbc.Row([
                    dbc.Col(dcc.DatePickerRange(id='date-picker-range', min_date_allowed=datetime.date(2019, 8, 1),
                                                max_date_allowed=datetime.date.today(), end_date=datetime.date.today(),
                                                start_date=datetime.date.today()),
                            width='auto'),
                    dbc.Col(dbc.Button('Get records', className="me-1", n_clicks=0, id='get-records-btn'),
                            align='center',
                            width='auto'),
                    dbc.Col(analyze_button, width='auto', align='center'),
                    dbc.Col(analyze_options, width='auto', align='center'),
                    dbc.Col(zoom_webcam_analyzed_modal, width='auto', align='center')]
                )),
                dbc.Col([dbc.Row(dbc.Col(dbc.Select(id='video-viewer-options',
                                                    options=[{'label': 'Zoom Recorded', 'value': '1'},
                                                             {'label': 'Analyzed Video', 'value': '2'}],
                                                    value=1,
                                                    ), width={'size': 'auto'}, align='center'
                                         ), justify='end')]),
            ],
                class_name='g-1', align='center'
            ),

            dbc.Row([
                dbc.Col(html.Div(meeting_list), width=6),
                dbc.Col(html.Div(meeting_player), width=6)
            ],
                class_name='g-1 h-50',
                justify='between',
            ),

            dbc.Row([
                dbc.Col(
                    [dbc.Row([dbc.Col(dcc.Download(id='dl-atr-dcc')),
                              dbc.Col(dbc.Button('Export Excel', id='dl-attendance-results'), width='auto',
                                      align='end')]),
                     dbc.Row(attendance_check_results)], width=6),
                # dbc.Col(attendance_check_graphs, width=6)
                dbc.Col([
                    dbc.Row([dbc.Col(html.Span("Visualization"), width={'offset': 1, 'size': 'auto'}, align='center', ),
                             dbc.Col(dbc.Select(id='viz_select',
                                                options=[{'label': 'Attendance (%)', 'value': '1'}],
                                                value=1
                                                ), width='auto'
                                     ),
                             dbc.Col(dcc.Download(id='dl-viz-dcc'), style={'display': 'none'}),
                             dbc.Col(dbc.Button('Export Excel', id='dl-viz-results'), width='auto',
                                     )
                             ], align='start'),
                    dbc.Row(attendance_check_graphs),
                ], width=6)
            ],
                className='g-1',
                # justify='start'
            ),
            # @self.app.callback(Output('check-url', 'children'), Input('current-url', 'pathname'))
            dbc.Row([dbc.Col(dbc.Label(id='access-token', children=''), width=4)], class_name='g-1', justify='between',
                    style={'display': 'none'}),
            dbc.Row([html.P(id='check-meeting-player', style={'display': 'none'})]),
            dbc.Row([html.P(id='check-url', style={'display': 'none'}), dcc.Location(id='current-url', refresh=True),
                     dcc.Store('zoom_webcam_store'), dcc.Store('video-src-webcam')]),

            # dbc.Row([dbc.Col(html.Img(src="/video_feed"), width=3)])
        ],
            class_name='overflow-hidden d-grid gap-4',
            fluid=True,
            style={'width': '95%'}
        )

        # Server route
        @self.app.server.route('/video_feed')
        def video_feed():
            return Response(gen_camera(self.webcam_object), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.server.route('/video_feed_analyzed')
        def video_feed_analyzed():
            return Response(stream_with_context(self.face_engine.gen_frame_webcam()),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # Callbacks
        @self.app.callback([Output('webcam-modal', 'style'), Output('add-photo-btn', 'disabled')],
                           [Input('add-photo-webcam', 'n_clicks'), Input('close-webcam-take-picture-btn', 'n_clicks')],
                           prevent_initial_call=True)
        def toggle_webcam_modal(n_click, close_n_click):

            change_id = [p['prop_id'] for p in callback_context.triggered][0]
            if 'add-photo-webcam' in change_id:
                # Take photos from webcam
                self.trigger_webcam_signal.emit(True)
                print('Add photo webcam clicked')
                return {}, True
            elif 'close-webcam-take-picture-btn' in change_id:
                # Close webcam
                print('Close webcam clicked')
                self.trigger_webcam_signal.emit(False)
                return {'display': 'none'}, False
            else:
                raise PreventUpdate

        @self.app.callback(Output('student-database-modal', 'is_open'),
                           [Input('std-db-btn', 'n_clicks'), Input('close-std-db-modal-btn', 'n_clicks')],
                           State('student-database-modal', 'is_open'), prevent_initial_call=True)
        def toggle_student_database_modal(n_click, close_n_click, is_open):
            if n_click or close_n_click:
                return not is_open

        @self.app.callback(Output('student-list-db', 'options'), Input("student-list-db", "search_value"),
                           running=[(Output('add-photo-btn', 'disabled'), True, False)])
        def update_options(search_value):
            cur_std_db = [{'label': std_id, 'value': std_id} for std_id in os.listdir('./assets/students/') if
                          os.path.isdir('./assets/students/{}'.format(std_id))]

            if not search_value:
                raise PreventUpdate
                # return cur_std_db

            search_results = [o for o in cur_std_db if search_value in o['label']]
            if len(search_results) > 0:
                return search_results
            else:
                return [{'label': 'Add {} to database'.format(search_value), 'value': search_value}]

        @self.app.callback(Output('student-photos', 'children'),
                           [Input('student-list-db', 'value'), Input('take-picture', 'n_clicks'),
                            Input('add-photo-upload', 'contents')],
                           State('webcam-jpeg', 'src'), State('current-url', 'href'),
                           State('add-photo-upload', 'filename'),
                           prevent_initial_call=True)
        def selected_student_db(value, n_clicks, photo_contents, webcam_jpg, current_url, photo_filenames):

            if not value:
                print('No ID selected')
                raise PreventUpdate

            change_id = [p['prop_id'] for p in callback_context.triggered][0]

            if not os.path.isdir('./assets/students/{}'.format(value)):
                os.makedirs('./assets/students/{}'.format(value))

            if 'take-picture' in change_id:
                # Take picture
                # print('Emit capture webcam signal')
                self.capture_webcam_signal.emit(value)
                self.update_database = True
                time.sleep(0.2)
            elif 'add-photo-upload' in change_id:
                print('Add photo upload ', photo_filenames)

                # Process uploaded file
                if not isinstance(photo_filenames, list):
                    photo_contents = [photo_contents]
                    photo_filenames = [photo_filenames]

                print('Number of photo: ', len(photo_filenames))
                for file_idx in range(len(photo_contents)):
                    content_type, content_string = photo_contents[file_idx].split(',')
                    photo_name = photo_filenames[file_idx]

                    if 'jpeg' not in content_type and 'png' not in content_type:
                        continue
                    decoded = base64.b64decode(content_string)
                    print('Write path: ', './assets/students/{}/up_{}'.format(value, photo_name))
                    with open('./assets/students/{}/{}'.format(value, photo_name), 'wb') as upl_writer:
                        upl_writer.write(decoded)
                    self.update_database = True

            list_photos = sorted(glob2.glob('./assets/students/{}/*.jpg'.format(value)) + glob2.glob(
                './assets/students/{}/*.png'.format(value)) + glob2.glob('./assets/students/{}/*.jpeg'.format(value)))

            if len(list_photos) > 0:
                card = [dbc.Card(children=[dbc.CardImg(src=x[1:], id=x, class_name='img-fluid pb-3 m-0', alt=x)],
                                 class_name='col-3', style={'justifyContent': 'center', 'border': 'none'}) for x in
                        list_photos]
            else:
                card = html.Span('There is no photo of this student (ID: {}) in the database'.format(value))
            return card

        @self.app.callback(
            [Output("analyze-options", "is_open"), Output('classes-list', 'options'), Output('classes-list', 'value'),
             Output('zoom_webcam_store', 'data')],
            [Input('analyze-btn', 'n_clicks'), Input('do-analyze-btn', 'n_clicks'), Input('zoom-webcam', 'n_clicks')],
            [State('analyze-options', 'is_open'), State('classes-list', 'options'), State('zoom_webcam_store', 'data'),
             State('classes-list', 'value')], prevent_initial_call=True)
        def toggle_analyze_options(analyze_click, do_analyze_click, zoom_webcam_clicks, is_open, student_list_options,
                                   student_list_value, zoom_webcam_store):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]
            if 'zoom-webcam' in change_id:
                self.zoom_webcam = True
            elif 'analyze-btn' in change_id:
                self.zoom_webcam = False
            else:
                self.zoom_webcam = zoom_webcam_store

            if analyze_click or do_analyze_click or zoom_webcam_clicks:
                cur_sel_idx = student_list_value
                class_name = student_list_options
                if not is_open:
                    # Will open, update student lists
                    # student-list
                    class_list_files = sorted(glob2.glob('./assets/classes_list/*.txt'))
                    # options = [{'label': 'seconds', 'value': 0}, {'label': 'minutes', 'value': 1}]
                    class_name = [
                        {'label': class_list_files[cl_idx].split('/')[-1].replace('.txt', ''), 'value': cl_idx} for
                        cl_idx in range(len(class_list_files))]

                    print('Current list value: ', type(student_list_value))
                    if student_list_value is not None:
                        cur_sel = student_list_options[int(student_list_value)]['label']
                        for idx in range(len(class_name)):
                            print(class_name[idx]['label'], cur_sel)
                            if class_name[idx]['label'] == cur_sel:
                                cur_sel_idx = idx
                                print(cur_sel_idx)
                                break
                    else:
                        cur_sel_idx = class_name[0]['value']

                return not is_open, class_name, cur_sel_idx, self.zoom_webcam

            return is_open, student_list_options, student_list_value, self.zoom_webcam

        @self.app.callback([Output('meeting-lists', 'data')],
                           [Input('get-records-btn', 'n_clicks'), Input('zoom-local', 'contents')],
                           State('date-picker-range', 'start_date'), State('date-picker-range', 'end_date'),
                           State('zoom-local', 'filename'), State('zoom-local', 'last_modified'),
                           running=[(Output('get-records-btn', 'disabled'), True, False)],
                           prevent_initial_call=True)
        def get_meeting_lists(grbtn_clicks, zoom_local_contents, start_date, end_date, zoom_local_names,
                              zoom_local_last_modified):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]

            meeting_lists_data = DataFrame(columns=['ID', 'Topics', 'Start time'])

            if 'get-records-btn' in change_id and self.access_token != '':
                # Get list of meeting
                requests_address = 'https://api.zoom.us/v2/users/{}/recordings'.format(zoom_userID)
                requests_headers = {
                    "Authorization": "Bearer " + self.access_token,
                }
                resp_get = requests.get(url=requests_address, headers=requests_headers,
                                        params={'from': start_date, 'to': end_date}
                                        )
                print(resp_get.status_code, resp_get.json())
                resp = resp_get.json()

                if resp_get.status_code == 200:
                    if resp['total_records'] > 0:
                        total_records = resp['total_records']
                        list_meetings = resp['meetings']

                        disp_meetings = []
                        for idx in range(total_records):
                            cur_uuid = list_meetings[idx]['uuid']
                            curr_topic = list_meetings[idx]['topic']
                            curr_start_time = list_meetings[idx]['start_time']
                            curr_recording_files = list_meetings[idx]['recording_files']
                            cur_gal_view = False
                            cur_dl_link = ''
                            cur_play_link = ''
                            for rec_files in curr_recording_files:
                                if rec_files['recording_type'] == 'gallery_view':
                                    cur_gal_view = True
                                    cur_dl_link = rec_files['download_url']
                                    cur_play_link = rec_files['play_url']

                            disp_meetings.append([cur_uuid, curr_topic, curr_start_time, cur_dl_link, cur_play_link])

                        df = DataFrame(data=disp_meetings, index=None,
                                       columns=['ID', 'Topics', 'Start time', 'DL Link', 'Play Link'])
                        return (df.to_dict('records'),)
            elif 'zoom-local' in change_id:
                # Open and Select file with QFileDialog
                # TODO:
                if zoom_local_contents is not None:
                    print(zoom_local_names, zoom_local_last_modified)
                    if not isinstance(zoom_local_contents, list):
                        zoom_local_contents = [zoom_local_contents]
                        zoom_local_names = [zoom_local_names]
                        zoom_local_last_modified = [zoom_local_last_modified]

                    disp_meetings = []
                    for file_idx in range(len(zoom_local_contents)):
                        # Assume all video are mp4 file
                        if not zoom_local_names[file_idx].endswith('.mp4'):
                            print("It's not mp4 file")
                            continue
                        curr_uuid = zoom_local_names[file_idx].replace('.mp4', '')
                        curr_topic = curr_uuid
                        curr_start_time = datetime.datetime.fromtimestamp(zoom_local_last_modified[file_idx])
                        cur_dl_link = './assets/recordings/{}_local.mp4'.format(curr_uuid)
                        cur_play_link = './assets/recordings/{}_local.mp4'.format(curr_uuid)

                        disp_meetings.append([curr_uuid, curr_topic, curr_start_time, cur_dl_link, cur_play_link])

                        content_type, content_string = zoom_local_contents[file_idx].split(',')
                        decoded_content = base64.b64decode(content_string)

                        # Write uploaded file to assets/recordings/
                        with open('./assets/recordings/{}_local.mp4'.format(curr_uuid), 'wb') as upl_writer:
                            upl_writer.write(decoded_content)

                    df = DataFrame(data=disp_meetings, index=None,
                                   columns=['ID', 'Topics', 'Start time', 'DL Link', 'Play Link'])
                    print('Upload files\n', df)
                    return (df.to_dict('records'),)
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate
            return (meeting_lists_data.to_dict('records'),)

        @self.app.callback(Output('check-meeting-player', 'children'), Input("meeting-player", "src"),
                           prevent_initial_call=True)
        def check_meeting_player(meeting_src):
            print('Meeting src changed')
            if meeting_src is not None:
                print('Meeting src changed ' + meeting_src)
                return 'Meeting src changed ' + meeting_src
            else:
                raise PreventUpdate

        @self.app.callback([Output("meeting-player", "src")],
                           [Input('meeting-lists', 'selected_rows'), Input('video-viewer-options', 'value')],
                           State('meeting-lists', 'data'), State('zoom_webcam_store', 'data'),
                           State('video-src-webcam', 'data'),
                           prevent_initial_call=True)
        def update_player_src(selected_rows, viewer_value, meeting_lists_data, zoom_webcam_store, zoom_video_src):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]
            print(change_id, callback_context.triggered)
            if 'meeting-lists' in change_id:
                meeting_uuid = meeting_lists_data[selected_rows[0]]['ID']
                curr_dl_url = meeting_lists_data[selected_rows[0]]['DL Link']

                if not os.path.isdir('./assets/recordings'):
                    os.makedirs('./assets/recordings')

                # Download meeting
                if validators.url(curr_dl_url):
                    if not os.path.isfile('./assets/recordings/{}.mp4'.format(meeting_uuid)):
                        resp_dl = requests.get(curr_dl_url).content
                        with open('./assets/recordings/{}.mp4'.format(meeting_uuid), 'wb') as fd:
                            fd.write(resp_dl)
                    postfix = ''
                else:
                    # It's a local file in assets
                    postfix = '_local'

                play_url = '/assets/recordings/{}{}.mp4'.format(meeting_uuid, postfix)

            elif 'video-viewer-options' in change_id:
                try:
                    if selected_rows is not None and not zoom_webcam_store:
                        meeting_uuid = meeting_lists_data[selected_rows[0]]['ID']
                        if validators.url(meeting_lists_data[selected_rows[0]]['DL Link']):
                            postfix = ''
                        else:
                            postfix = '_local'
                        if int(viewer_value) == 1:
                            play_url = '/assets/recordings/{}{}.mp4'.format(meeting_uuid, postfix)
                        else:
                            play_url = '/assets/recordings/{}{}.mp4'.format(meeting_uuid, postfix).replace('.mp4',
                                                                                                           '_analyzed.mp4')
                    else:
                        if int(viewer_value) == 1:
                            play_url = zoom_video_src
                        else:
                            play_url = zoom_video_src.replace('.mp4', '_analyzed.mp4')
                except:
                    play_url = ''
            else:
                play_url = ''

            print(viewer_value, play_url)
            return (play_url,)

        @self.app.callback(Output('zoom-webcam-analyzed-modal', 'is_open'),
                           [Input('do-analyze-btn', 'n_clicks'), Input('do-zoom-webcam-finish', 'n_clicks')],
                           State('zoom-webcam-analyzed-modal', 'is_open'), State('zoom_webcam_store', 'data'))
        def open_webcam_analyzing(n_clicks, do_zoom_webcam_finish_click, is_open, zoom_webcam_store):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]

            print('Triggered open_zoom_webcam_analyzing', 'do-analyze-btn' in change_id, self.zoom_webcam)
            if 'do-analyze-btn' in change_id and zoom_webcam_store:
                print('Triggered, ', is_open, not is_open)
                return not is_open
            elif 'do-zoom-webcam-finish' in change_id:
                # Close zoom analyzing
                self.zoom_stop_analyze_signal.emit()
                return False
            else:
                raise PreventUpdate

        @self.app.callback(
            [Output("attendance-results", "data"), Output('attendance-results', 'columns'),
             Output('video-src-webcam', 'data')],
            [Input('do-analyze-btn', 'n_clicks'), State("attendance-results", "data"),
             State('analyze-interval', 'value'), State('interval-type', 'value'),
             State('face-analyze-options', 'value'), State('recognition-sensitive', 'value'),
             State('attendance-results', 'columns'), State('meeting-lists', 'data'),
             State('meeting-lists', 'selected_rows'),
             State('access-token', 'children'),
             State('classes-list', 'options'), State('classes-list', 'value'),
             State('zoom_webcam_store', 'data'), State('video-src-webcam', 'data'), ],
            running=[(Output('analyze-btn', 'disabled'), True, False),
                     (Output('analyze-btn', 'children'), [dbc.Spinner(size="sm"), " Running..."], 'Analyze'),
                     ],
            # interval=10*1800,
            prevent_initial_call=True)
        def run_face_analysis(do_analyze_btn_clicks, attendance_data, analyze_interval, analyze_interval_type,
                              analyze_options, analyze_senstive,
                              columns, meeting_lists_data, selected_rows, access_token,
                              classes_name, classes_idx, zoom_webcam_store, zoom_video_src,
                              ):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]

            if 'do-analyze-btn' not in change_id:
                raise PreventUpdate

            #######
            self.zoom_webcam = zoom_webcam_store
            analyze_interval_type = int(analyze_interval_type)
            if analyze_interval_type == 1:
                second_per_detect = 60 * analyze_interval
            else:
                second_per_detect = analyze_interval

            if self.update_database:
                self.trigger_re_init_embedding_signal.emit()
                self.update_database = False

            cls_name_students = np.loadtxt(
                './assets/classes_list/{}.txt'.format(classes_name[int(classes_idx)]['label']), dtype=str).tolist()
            print('List students: ', cls_name_students, self.zoom_webcam)

            if self.zoom_webcam or selected_rows is None:
                video_src = ('0', './assets/recordings/zoom_webcam_{}.mp4'.format(
                    datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")))
                play_video_src = video_src[1][1:]
            else:
                meeting_uuid = meeting_lists_data[selected_rows[0]]['ID']
                if validators.url(meeting_lists_data[selected_rows[0]]['DL Link']):
                    postfix = ''
                else:
                    postfix = '_local'
                video_src = './assets/recordings/{}{}.mp4'.format(meeting_uuid, postfix)
                play_video_src = video_src[1:]

            attendance_df = self.face_engine.start_video_analysis(video_src,
                                                                  second_per_detect=second_per_detect,
                                                                  dist_thresh=analyze_senstive,
                                                                  in_minutes=analyze_interval_type,
                                                                  cls_name_students=cls_name_students)

            columns = [{"name": i, "id": i} for i in list(attendance_df.keys())]
            ## TODO:
            # print('Finished face engine ', attendance_data)
            attendance_data = attendance_df.to_dict('records')
            return attendance_data, columns, play_video_src

        @self.app.callback(Output('dl-atr-dcc', 'data'), [Input('dl-attendance-results', 'n_clicks')],
                           State('attendance-results', 'data'), prevent_initial_call=True)
        def download_attendance_results_table(n_clicks, data):
            df = DataFrame.from_records(data)
            return dcc.send_data_frame(df.to_excel, "attendance_results.xlsx", index=False,
                                       sheet_name='attendance results')

        @self.app.callback(Output('dl-viz-dcc', 'data'), [Input('dl-viz-results', 'n_clicks')],
                           State('attendance_graphs', 'figure'), prevent_initial_call=True)
        def download_attendance_results_table(n_clicks, figure_data):
            print(figure_data['data'])
            data = figure_data['data']
            cols = ['Student ID', data[0]['name']]
            x = data[0]['x']
            y = data[0]['y']
            df = DataFrame(data=np.array([x, y]).transpose(), columns=cols)

            return dcc.send_data_frame(df.to_excel, "attendance_viz_download.xlsx", index=False,
                                       sheet_name='attendance viz')

        @self.app.callback(Output('attendance_graphs', 'figure'),
                           [Input('viz_select', 'value'), Input('attendance-results', 'data')],
                           prevent_initial_call=True)
        def update_attendance_graph(viz_value, attendance_results_data):
            change_id = [p['prop_id'] for p in callback_context.triggered][0]
            if 'viz_select' not in change_id and 'attendance-results' not in change_id:
                raise PreventUpdate
            else:
                print(change_id)
            if int(viz_value) == 1 and attendance_results_data is not None:
                # Attendance with percent for each students
                print(attendance_results_data)
                if len(attendance_results_data) > 0:
                    df = DataFrame.from_records(data=attendance_results_data)
                    stu_perc = {}
                    for stu in df.columns:
                        if 'Timestamp' not in stu:
                            stu_perc[stu] = df[stu].sum() * 100 / len(df)

                    print(stu_perc, stu_perc.keys(), stu_perc.values())
                    figure = {'data': [{'x': list(stu_perc.keys()), 'y': list(stu_perc.values()), 'type': 'bar',
                                        'name': 'Attendance (%)'}],
                              'layout': {
                                  'title': 'Attendance (%) visualization'
                              }}
                    # figure = px.bar(x=stu_perc.keys(), y=stu_perc.values())
                    return figure

            raise PreventUpdate

        @self.app.callback(
            [Output('check-url', 'children'), Output('current-url', 'href'), Output('user_btn', 'label'),
             Output('current-url', 'refresh')],
            Input('current-url', 'search'), State('current-url', 'href'), State('user_btn', 'label'),
            prevent_initial_call=True)
        def url_updating(value, cur_href, cur_usr_btn_label):
            usr_btn_label = cur_usr_btn_label
            if value.startswith('?code='):
                self.oauth_code = value.split('&')[0].replace('?code=', '')
                print(self.oauth_code, value)

                print('Get access token')
                success = self.get_access_token()

                if success:
                    # Get user name
                    requests_address = 'https://api.zoom.us/v2/users/{}'.format(zoom_userID)
                    requests_headers = {
                        "Authorization": "Bearer " + self.access_token,
                    }
                    resp_get = requests.get(url=requests_address, headers=requests_headers,
                                            params={'from': '2021-10-25', 'to': '2021-11-24'}
                                            )
                    print(resp_get.status_code, resp_get.json())
                    resp = resp_get.json()
                    usr_btn_label = '{} {}'.format(resp['last_name'], resp['first_name'])
                    print('Success: ', usr_btn_label)
                    href_oauth = zoom_oauth_redirect_uri
                    return '', href_oauth, usr_btn_label, False
                else:
                    href_oauth = "https://zoom.us/oauth/authorize?response_type=code&client_id={}&redirect_uri={}".format(
                        zoom_client_ID, zoom_oauth_redirect_uri)
                    return '', href_oauth, usr_btn_label, True
            # print('URL changed ', value)
            return 'URL changed '.format(value), cur_href, usr_btn_label, False

    @property
    def app(self):
        return self.__app

    def get_access_token(self):
        num_count = 0
        while True:
            if num_count == 2:
                self.access_token = ''
                return False

            is_refresh = self.access_token != ''
            authorize_str = '{}:{}'.format(zoom_client_ID, zoom_client_secret)
            authorize_str_base64 = base64.b64encode(authorize_str.encode('ascii'))
            grant_type = 'authorization_code' if not is_refresh else 'refresh_token'
            refresh_token = 'refresh_token={}&'.format(self.access_token) if is_refresh else ''

            resp = requests.post(
                url='https://zoom.us/oauth/token?{}grant_type={}&code={}&redirect_uri={}'.format(refresh_token,
                                                                                                 grant_type,
                                                                                                 self.oauth_code,
                                                                                                 zoom_oauth_redirect_uri),
                headers={'Authorization': 'Basic ' + authorize_str_base64.decode('ascii'),
                         'Content-Type': 'application/x-www-form-urlencoded'},
            ).json()
            print(resp)

            if 'access_token' in resp:
                self.access_token = resp['access_token']
                return True
            num_count += 1

    def run(self, **kwargs):
        threading.Thread(target=self.app.run_server, kwargs=kwargs, daemon=True).start()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setWindowIcon(QIcon('./assets/favicon.ico'))
        self.setWindowTitle("PRLAB - Facial Analysis")
        self.browser = QtWebEngineWidgets.QWebEngineView()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        lay = QtWidgets.QVBoxLayout(central_widget)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self.browser, stretch=1)
        self.resize(1200, 1200)
        # Dash
        self.qdash = QDash()
        self.qdash.run(debug=True, use_reloader=False, dev_tools_hot_reload=False, host='0.0.0.0')
        self.browser.load(QtCore.QUrl("http://127.0.0.1:8050"))

        # self.browser.urlChanged.connect(self.get_oauth_code)
        self.browser.page().profile().downloadRequested.connect(self.on_downloadRequested)

        self.oauth_code = ''
        self.access_token = ''

    @QtCore.pyqtSlot(QtCore.QUrl)
    def get_oauth_code(self, url_: QtCore.QUrl):
        url_str = url_.toString()
        self.access_token = ''
        if url_str.startswith(zoom_oauth_redirect_uri):
            self.oauth_code = url_str.replace('{}/?code='.format(zoom_oauth_redirect_uri), '')
            print(self.oauth_code, url_str)
            self.browser.setUrl(QtCore.QUrl("http://127.0.0.1:8050"))
            self.get_access_token()

    def get_access_token(self):
        num_count = 0
        while True:
            if num_count == 2:
                self.browser.load(QtCore.QUrl(
                    "https://zoom.us/oauth/authorize?response_type=code&client_id={}&redirect_uri={}".format(
                        zoom_client_ID, zoom_oauth_redirect_uri)))
                break
            is_refresh = self.access_token != ''
            authorize_str = '{}:{}'.format(zoom_client_ID, zoom_client_secret)
            authorize_str_base64 = base64.b64encode(authorize_str.encode('ascii'))
            grant_type = 'authorization_code' if not is_refresh else 'refresh_token'
            refresh_token = 'refresh_token={}&'.format(self.access_token) if is_refresh else ''

            resp = requests.post(
                url='https://zoom.us/oauth/token?{}grant_type={}&code={}&redirect_uri={}'.format(refresh_token,
                                                                                                 grant_type,
                                                                                                 self.oauth_code,
                                                                                                 zoom_oauth_redirect_uri),
                headers={'Authorization': 'Basic ' + authorize_str_base64.decode('ascii'),
                         'Content-Type': 'application/x-www-form-urlencoded'},
            ).json()
            print(resp)
            if 'error' in resp.keys():
                num_count += 1
                print('Still in here')
                continue
            else:
                self.access_token = resp['access_token']

                # Get user name
                requests_address = 'https://api.zoom.us/v2/users/{}'.format(zoom_userID)
                requests_headers = {
                    "Authorization": "Bearer " + self.access_token,
                }
                resp_get = requests.get(url=requests_address, headers=requests_headers,
                                        params={'from': '2021-10-25', 'to': '2021-11-24'}
                                        )
                print(resp_get.status_code, resp_get.json())
                resp = resp_get.json()

                # Update oauth and access_key, user_name to dash
                update_values = {'oauth-code': self.oauth_code, 'access-token': self.access_token}
                for idx in range(len(self.qdash.app.layout.children)):

                    if hasattr(self.qdash.app.layout.children[idx], 'children') and isinstance(
                            self.qdash.app.layout.children[idx].children, list):
                        for idx_child in range(len(self.qdash.app.layout.children[idx].children)):
                            if hasattr(self.qdash.app.layout.children[idx].children[idx_child], 'children') and hasattr(
                                    self.qdash.app.layout.children[idx].children[idx_child].children, 'id'):
                                if self.qdash.app.layout.children[idx].children[idx_child].children.id in ['user_btn']:
                                    self.qdash.app.layout.children[idx].children[
                                        idx_child].children.label = '{} {}'.format(resp['last_name'],
                                                                                   resp['first_name'])

                                elif self.qdash.app.layout.children[idx].children[idx_child].children.id in [
                                    'oauth-code', 'access-token']:
                                    self.qdash.app.layout.children[idx].children[idx_child].children.children = \
                                        update_values[
                                            self.qdash.app.layout.children[idx].children[idx_child].children.id]

                # self.browser.load(QtCore.QUrl("http://127.0.0.1:8050"))
                break

    @QtCore.pyqtSlot(QWebEngineDownloadRequest)
    def on_downloadRequested(self, download: QWebEngineDownloadRequest):
        old_path = download.downloadDirectory()
        file_name = download.downloadFileName()
        suffix = QtCore.QFileInfo(file_name).suffix()
        print(suffix, old_path, file_name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="Save File",
                                                        directory='{}/{}'.format(old_path, file_name))
        if path:
            download.setDownloadDirectory(QtCore.QFileInfo(path).absolutePath())
            download.setDownloadFileName(QtCore.QFileInfo(path).fileName())

            download.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    w = MainWindow()
    w.show()

    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
