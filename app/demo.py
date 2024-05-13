import sys
sys.path.insert(0, '/home/iitaku/Develop/ion-kit/python')
from ionpy import Node, Builder, Buffer, Port, Param, Type, TypeCode
import numpy as np

from tkinter import *
from tkinter.ttk import *
import sv_ttk
from PIL import ImageTk, Image

class App(Frame):
    def __init__(self, window):
        super().__init__(window, padding=15)

        # self.rowconfigure(0, weight=1)
        # self.rowconfigure(1, weight=1)
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=1)

        self.window = window

        # Model
        self.width = 1280
        self.height = 960
        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
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

        params = [Param("num_devices", 1), Param("realtime_diaplay_mode", True)]
        n_img_cwh = self.b.add("image_io_u3v_cameraN_u8x3").set_param(params)

        self.prompt_buf = Buffer(array=self.prompt)
        prompt_port = Port(name="prompt", type=Type(TypeCode.Int, 8, 1), dim=1)
        prompt_port.bind(self.prompt_buf)

        params = [Param("width", self.width), Param("height", self.height)]
        n_txt = self.b.add("llm_llava").set_iport([n_img_cwh.get_port("output")[0], prompt_port]).set_param(params)

        for i in range(self.height):
            for j in range(self.width):
                self.img[i][j] = [i%256, i%256, i%256]

        self.img_buf = Buffer(array=self.img)
        n_img_cwh.get_port("output").bind(self.img_buf)

        self.response_buf = Buffer(array=self.response)
        n_txt.get_port("output").bind(self.response_buf)

    def init_layout(self):
        xframe = Frame(self)
        xframe.place(relx=0.5, rely=0.5, anchor="c")

        img_frame = Frame(xframe, padding=15)
        # #img_frame.grid(row=2, column=0, columnspan=12, sticky='nsew')
        # img_frame.grid(row=2, sticky='nsew')
        #self.img_canvas = Canvas(img_frame, width = self.width, height = self.height)
        self.img_canvas = Canvas(xframe, width = 1920, height = 1080)
        self.img_canvas.pack()

        control_frame = Frame(xframe, padding=15)
        #control_frame.grid(row=0, column=0, sticky='nsew')
        # control_frame.grid(row=0, sticky='nsew')
        self.prompt_textbox = Entry(control_frame, textvariable=self.prompt_string, width=100, font=('Helvetica', 18))
        self.prompt_textbox.bind("<FocusOut>", lambda event: self.update_prompt())
        # self.prompt_textbox.grid(row=0, column=0, columnspan=10, padx=5)
        # self.prompt_label = Label(xframe, font=('Helvetica', 40), wraplength=1900, justify='left')
        # self.prompt_label.configure(text=self.prompt_string)
        # self.img_canvas.create_window(80, 40, window = self.prompt_label, anchor = NW)

        # self.live_checkbutton = Checkbutton(control_frame, text='Live', style='Switch.TCheckbutton', command=self.toggle_live)
        # self.live_checkbutton.grid(row=0, column=10, padx=5)

        # self.analysis_button = Button(control_frame, text='Analyze', command = self.analyze, width=10, style='Accent.TButton')
        # self.analysis_button.grid(row=0, column=11, padx=5)

        response_frame = Frame(xframe, padding=15, height=50)
        #response_frame.grid(row=1, columnspan=2, sticky='nsew')
        # response_frame.grid(row=1, sticky='nsew')
        # self.question_label = Label(response_frame, font=('Helvetica', 48), wraplength=1850, padding=15, anchor= 'nw', justify='left')
        # self.question_label.configure(text="Hey, what's on your eyes?")
        # self.question_label.pack()
        self.response_label = Label(response_frame, font=('Helvetica', 48), wraplength=1850, padding=15, anchor = 'nw', justify='left')
        self.response_label.pack()
        #eself.response_label.pack()
        self.img_canvas.create_window(40, 40, window = response_frame, anchor = 'nw')

        self.update_prompt()
        self.update_response()
        self.update_periodic()

    def update_periodic(self):
        # Running pipeline
        self.b.run()

        img = Image.fromarray(self.img)
        img = img.crop((0, 120, 1280, 120+720))
        img = img.resize((1920, 1080))
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

# GUIウィンドウの作成
root = Tk()  # create CTk window like you do with the Tk window
#root.title("Interactive LLAVA Demo")
root.geometry("1920x1280")
root.wm_attributes('-type', 'splash')
root.wm_attributes('-fullscreen', True)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
sv_ttk.set_theme("dark")
App(root).pack(expand=True, fill='both')
root.mainloop()
