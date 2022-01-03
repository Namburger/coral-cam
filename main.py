import os
import sys
from tkinter import Tk, messagebox
import eel
import base64
from camera import VideoCamera


@eel.expose
def show_error(title, msg):
    root = Tk()
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg)
    root.destroy()


def gen_frame(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield frame


@eel.expose
def video_feed():
    camera = VideoCamera()
    for frame in gen_frame(camera):
        # Convert bytes to base64 encoded str, as we can only pass json to frontend
        blob = base64.b64encode(frame)
        blob = blob.decode("utf-8")
        eel.updateImageSrc(blob)()


# Start the server
def start_app():
    try:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        eel.init(os.path.join(curr_path, 'web'))
        eel.start('index.html', size=(1280, 770))
    except Exception as e:
        show_error(title='Failed to initialise server', msg=f'Could not launch a local server, reason: {e}')
        sys.exit()


if __name__ == "__main__":
    start_app()
