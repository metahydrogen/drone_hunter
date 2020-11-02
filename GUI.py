import tkinter as tk
import os

# Create a window object
window = tk.Tk()
window.title('Drone Hunter Controls')
window.geometry('500x500+500+500')

# Create Frames
frame = tk.Frame(master=window)
frame.pack()

# Create function definitions
def detectON():
    print("Object Detection -> ON")
    os.system('python TFLite_detection_webcam.py --modeldir Sample_TFLite_model --threshold 0.65')
def detectOFF():
    print("Object Detection -> OFF")
    os.system(q)
# def rPiSSH():
#     print('Connecting via SSH')
#     ipaddress='raspberrypi.local'
#     hostname='pi'
#     os.system('ssh ' + hostname + '@' + ipaddress)
#     print('Connected Successfully')
def function4():
    print(self)

# Create Buttons
# button0 = tk.Button(
#     master=frame,
#     command=rPiSSH,
#     text='PiSSH',
#     fg='magenta',
#     bg='black',
#     width=15,
# 	height=3,
# )

button1 = tk.Button(
    master=frame,
    command=detectON,
    text='Detection ON',
    fg='green',
    bg='black',
    width=15,
	height=3,
)

button2 = tk.Button(
    master=frame,
    command=detectOFF,
    text='Detection OFF',
    fg='red',
    bg='black',
    width=15,
	height=3,
)

# Pack widgets into window
# button0.pack()
button1.pack()
button2.pack()

# main
window.mainloop()