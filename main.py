import os
import sys
from tkinter import Tk, messagebox
import eel
import base64
from coral_cam import CoralCam

# Coral Cam is a global singleton
coral_cam = CoralCam()


@eel.expose
def show_error(title, msg):
    root = Tk()
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg)
    root.destroy()


@eel.expose
def video_feed():
    while True:
        frame = coral_cam.get_frame()
        if frame is not None:
            # Convert bytes to base64 encoded str, as we can only pass json to frontend
            blob = base64.b64encode(frame).decode('utf-8')
            eel.updateImageSrc(blob)()


@eel.expose
def set_engine(inference_type, model, edgetpu):
    coral_cam.set_engine(inference_type, model, edgetpu)


if __name__ == "__main__":
    try:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        eel.init(os.path.join(curr_path, 'web'))
        eel.start('index.html', size=(1282, 900))
    except Exception as e:
        show_error(title='Failed to initialise server', msg=f'Could not launch a local server, reason: {e}')
        sys.exit()
