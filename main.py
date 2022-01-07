import os
import sys
from tkinter import Tk, messagebox
import eel
import base64
from coral_cam import CoralCam

# Coral Cam is a global singleton
coral_cam = CoralCam()


def show_error(title: str, msg: str):
    """Opens a tk message box and show a message on it.
    :param title: The title of the error.
    :param msg: The actual error message.
    :return: None
    """
    root = Tk()
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg)
    root.destroy()


@eel.expose
def video_feed():
    """ The main program loop that gets exposes to javascript and updates coral-cam-video-feed. Internally it calls
    coral_cam.get_frame() in order to get the frame that's already been preprocessed and then updates it to the image
    tag.
    :return: None
    """
    while True:
        frame = coral_cam.get_frame()
        if frame is not None:
            # Convert bytes to base64 encoded str, as we can only pass json to frontend
            blob = base64.b64encode(frame).decode('utf-8')
            eel.updateImageSrc(blob)()


@eel.expose
def set_engine(inference_type: str, model: str, edgetpu: bool):
    """ Switch inference mode, model and toggle the edgetpu on/off when the submit button is clicked.
    :param inference_type: The type of inference ['classification', 'detection', 'pose-estimation', 'segmentation']
    :param model: The name of the model that user selected.
    :param edgetpu: Where to toggle the edgetpu on or off.
    :return: None
    """
    coral_cam.set_engine(inference_type, model, edgetpu)


if __name__ == "__main__":
    try:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        eel.init(os.path.join(curr_path, 'web'))
        host = '0.0.0.0'
        port = 8888
        print(f'Starting CoralCam @ {host}:{port}')
        eel.start('index.html', size=(1282, 900), host=host, port=port)
    except Exception as e:
        show_error(title='Failed to initialise server', msg=f'Could not launch a local server, reason: {e}')
        sys.exit()
