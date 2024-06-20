import sys
import argparse

from ionpy import Node, Builder, Buffer, Port, Param, Type, TypeCode
import numpy as np

from tkinter import *
from tkinter.ttk import *
import sv_ttk
from PIL import ImageTk, Image

class App(Frame):
    def __init__(self, window, args):
        super().__init__(window, padding=15)

        self.window = window

        # Model
        self.camera_width = int(args.resolution.split('x')[0])
        self.camera_height = int(args.resolution.split('x')[1])
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        self.prompt = np.zeros(1024, dtype=np.int8)
        self.response = np.zeros(1024, dtype=np.int8)

        # View model
        self.prompt_string = StringVar()
        self.prompt_string.set('Explain the image in a single sentence.')

        # Support variables
        self.live_mode = True
        self.advanced_mode = False
        self.analyze_in_progress = False
        self.last_response = np.copy(self.response)

        self.init_pipeline()
        self.init_layout()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_pipeline(self):

        self.b = Builder()
        self.b.set_target("host-cuda")
        self.b.with_bb_module("ion-bb")


        # U3V camera
        # params = [Param("num_devices", 1), Param("realtime_diaplay_mode", True)]
        # n_img_cwh = self.b.add("image_io_u3v_cameraN_u8x3").set_params(params)

        # UVC camera
        params = [Param("width", self.camera_width), Param("height", self.camera_height)]
        n_img_whc = self.b.add("image_io_camera").set_params(params)
        params = [Param("dim0", 2), Param("dim1", 0), Param("dim2", 1)]
        n_img_cwh = self.b.add("base_reorder_buffer_3d_uint8").set_iports([n_img_whc.get_port("output")]).set_params(params);

        self.prompt_buf = Buffer(array=self.prompt)
        prompt_port = Port(name="prompt", type=Type(TypeCode.Int, 8, 1), dim=1)
        prompt_port.bind(self.prompt_buf)

        params = [Param("width", self.camera_width), Param("height", self.camera_height)]
        n_txt = self.b.add("llm_llava").set_iports([n_img_cwh.get_port("output")[0], prompt_port]).set_params(params)

        for i in range(self.camera_height):
            for j in range(self.camera_width):
                self.img[i][j] = [i%256, i%256, i%256]

        self.img_buf = Buffer(array=self.img)
        n_img_cwh.get_port("output").bind(self.img_buf)

        self.response_buf = Buffer(array=self.response)
        n_txt.get_port("output").bind(self.response_buf)

    def init_layout(self):
        self.img_canvas = Canvas(self, width = self.screen_width, height = self.screen_height)
        self.img_canvas.pack()

        response_frame = Frame(self, padding=15, height=50)
        self.response_label = Label(response_frame, font=('Helvetica', 48), wraplength=self.screen_width-30, padding=15, anchor = 'nw', justify='left')
        self.response_label.pack()
        self.img_canvas.create_window(40, 40, window = response_frame, anchor = 'nw')

        self.update_prompt()
        self.update_response()
        self.update_periodic()

    def update_periodic(self):
        # Running pipeline
        self.b.run()

        img = Image.fromarray(self.img)
        cutoff = self.camera_height - (self.screen_height / self.screen_width) * self.camera_width
        img = img.crop((0, cutoff/2, self.camera_width, self.camera_height-cutoff/2))
        img = img.resize((self.screen_width, self.screen_height))
        self.photo = ImageTk.PhotoImage(image = img)
        self.img_canvas.create_image(0, 0, image = self.photo, anchor = 'nw')

        if (self.live_mode):
            self.update_response()

        self.window.after(30, self.update_periodic)

    def update_prompt(self):
        self.prompt.fill(0)
        i = 0
        if not self.advanced_mode:
            # Append image prompt marker
            for c in '<image>':
                self.prompt[i] = ord(c)
                i += 1
        offset = i
        for i, c in enumerate(self.prompt_string.get()):
            self.prompt[offset+i] = ord(c)

        # Clearing response make look & feel better
        self.response.fill(0)
        self.response_label.configure(text='')

    def update_response(self):
        # question = "Hey buddy, what's on your eyes?\n"
        response = ''.join([chr(v) for v in self.response])
        response = response.split('.')[0]
        self.response_label.configure(text=response)

    def toggle_live(self):
        self.live_mode = not self.live_mode
        self.analysis_button.configure(state='disabled' if self.live_mode else 'normal')

    def analyze(self):
        self.analyze_in_progress = True
        self.analysis_button.configure(text='Analyzing...')
        self.last_response = np.copy(self.response)
        self.window.after(100, self.wait_response)

    def wait_response(self):
        if np.array_equal(self.last_response, self.response):
            self.window.after(100, self.wait_response)
        else:
            self.update_response()
            self.analysis_button.configure(text='Analyze')
            self.analyze_in_progress = False

    def on_closing(self):
        del self.b
        self.window.destroy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', default='640x480', help='Camera resolution in "<width>x<height>" format. e.g. 640x480')

    args = parser.parse_args()

    root = Tk()
    root.wm_attributes('-type', 'splash')
    root.wm_attributes('-fullscreen', True)
    sv_ttk.set_theme("dark")
    App(root, args).pack(expand=True, fill='both')
    root.mainloop()
