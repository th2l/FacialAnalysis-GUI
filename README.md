

For demo purpose only.

To use *Cloud Recording*, we need Zoom API Client ID, Client Secret, Redirect URL for OAuth. To get that, follow instructions on [Create an OAuth App
](https://marketplace.zoom.us/docs/guides/build/oauth-app) page. After that, export 3 environment variables `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET`, `ZOOM_REDIRECT_URI`, or modify directly [Line 51-53](https://github.com/th2l/FacialAnalysis-GUI/blob/main/main.py#L51-L53) in ```main.py```

Link for video demonstration: https://1drv.ms/v/s!AoeAp4aV23Gqeq-5PvynHUotpn8?e=TegyU2 

## Credits

* Environments: Tested with Miniconda3 on Linux (Fedora 35, Ubuntu 21.10) with Python 3.8
* User interface: [Dash 2.0](https://dash.plotly.com/), [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) and [PyQt6](https://riverbankcomputing.com/software/pyqt/intro)
* Face Recognition models: [Keras_insightface](https://github.com/leondgarse/Keras_insightface) and [deepinsight/insightface](https://github.com/deepinsight/insightface)
