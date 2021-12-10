

For **demo purpose** only.

1. Environments: Tested with Miniconda3 on Linux (Fedora 35, Ubuntu 21.10) with Python 3.8

2. To use *Cloud Recording*, we need Zoom API Client ID, Client Secret, Redirect URL for OAuth. To get that, follow instructions on [Create an OAuth App
](https://marketplace.zoom.us/docs/guides/build/oauth-app) page. After that, export 3 environment variables `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET`, `ZOOM_REDIRECT_URI`, or modify directly [Line 51-53](https://github.com/th2l/FacialAnalysis-GUI/blob/main/main.py#L51-L53) in ```main.py```

3. To run on GPU, we need to install CUDA and CUDNN for TensorFlow 2.6+.
4. If we do not want to run on GPU, or do not have GPU:
   * Changed [Line 297](https://github.com/th2l/FacialAnalysis-GUI/blob/main/main.py#L297) to `self.det.prepare(-1)`.
   * Install `onnxruntim` instead of `onnxruntime-gpu`.
   * Add `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` to [Line 15](https://github.com/th2l/FacialAnalysis-GUI/blob/main/main.py#L15).
5. Modify `scrfd.py` in `miniconda3/envs/projectai/lib/python3.8/site-packages/insightface/model_zoo/`: line 73, add `providers=None` in `__init__` function, and line 81, add `providers=providers` in `onnxruntime.InferenceSession`. This modification applied to `insightface==0.5` only, for other versions, it may be different, check it by yourself.
6. To run this code on Windows, we need to modify [Line 461](https://github.com/th2l/FacialAnalysis-GUI/blob/main/main.py#L461), change `p = Process(...)` to `p = threading.Thread(...)`, because Windows has some problems on `python multiprocessing` mechanism that I can not solve, so change it to *multithreading* is one of the solutions but it may slow down the application.
7. Default [PyQt6-WebEngine](https://pypi.org/project/PyQt6-WebEngine/) do not support to play mp4 file with H264 codec, we need to build from source Qt WebEngine with *Proprietary codecs* enabled, then copy corresponding *lib files* to installed PyQt6 folders. To do that, follow the instruction in https://doc.qt.io/qt-6/qtwebengine-features.html
8. This app can run in their UI or in localhost (default is http://localhost:8050/) with browsers (e.g. Chrome, Edge, etc.). When using localhost with Chrome or Edge, the mp4 files with H264 codec are normal play.
9. Link for video demonstration: https://1drv.ms/v/s!AoeAp4aV23Gqeq-5PvynHUotpn8?e=TegyU2 

If you find this repo helpful, please star it and credit below works.
## Credits

* User interface: [Dash 2.0](https://dash.plotly.com/), [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) and [PyQt6](https://riverbankcomputing.com/software/pyqt/intro)
* Face Recognition models: [Keras_insightface](https://github.com/leondgarse/Keras_insightface) and [deepinsight/insightface](https://github.com/deepinsight/insightface)
