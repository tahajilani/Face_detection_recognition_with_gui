import tkinter as tk
import cv2
from PIL import Image, ImageTk
from face_detection import set_GPU,get_boxes,draw_box,save_cropped,write_names,return_things
from train import train

#video_stream at 0,0
#instruction at 0,1
#entry form at 2,2
#button at 1,1
#saved picture at 0,3
cap = cv2.VideoCapture(0)
root = tk.Tk()
canvas=tk.Canvas(root,width=1200,height=500)
canvas.grid(columnspan=3)
root.bind('<Escape>', lambda e: root.quit())

# background=ImageTk.PhotoImage(file="background.png")
# canvas.create_image(0,0,image=background,anchor="nw")
device=set_GPU()
mtcnn,resnet=return_things(device)

def show_frame():
    _, frame = cap.read()
    count=0
    boxes=get_boxes(frame,device)
    universal_boxes=boxes
    names=write_names(boxes,frame,device,mtcnn,resnet)
    frame=draw_box(frame,boxes,names)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # percent by which the image is resized
    scale_percent = 100

    # calculate the 50 percent of original dimensions
    width = int(cv2image.shape[1] * scale_percent / 100)
    height = int(cv2image.shape[0] * scale_percent / 100)
    cv2image=cv2.resize(cv2image,(width,height))
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain = tk.Label(image=imgtk)
    lmain.image=imgtk
    lmain.grid(column=0,row=0,padx=5,pady=5,sticky='w')
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


#instructons
instruct='Press train button if you want to retrain ' \
         'data and update embeddings. Write your name' \
         'and press take pictures button to add yourself ' \
         'in our database. Sit infront of the camera such ' \
         'that atleast one quarter of the screen is covered' \
         'by your face and you can see a bounding box around' \
         'your face.While Pictures are being taken' \
         'Tilt to your left,right,up and down. '
instructions=tk.Text(root,height=10,width=50,padx=10,pady=10)
instructions.insert(1.0,instruct)
instructions.tag_configure("center",justify='center')
instructions.grid(column=1,row=0)
#entry widget
e=tk.Entry(root)
e.grid(column=1,row=1,ipadx=30,ipady=15)

e.insert(0,"Enter Name here: ")

def my_click():
    something=tk.Label()
    for count in range(100):
        _, frame = cap.read()
        boxes=get_boxes(frame,device,max_box=True)
        save_cropped(frame, boxes, id,count)


#take pictures button
take_pic_btn=tk.Button(root,bg='#8225A0',fg='#BD70FF',width=40,height=2,text='take pictures',command=my_click,font="Ralewway")
take_pic_btn.grid(column=0,row=2)


#display_image to be saved
train_btn=tk.Button(root,bg='#8225A0', fg='#BD70FF',width=40,height=2,text='train',command=train,font="Raleway")
train_btn.grid(column=0,row=1)
show_frame()
root.mainloop()
