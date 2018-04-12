#importing the libraries to be used later
#VAraible fro torch to do backprop with descent and one tensor as argument in there
#cv2 for putting the boxes over the image 
#basetransform for transforming the input as per the requirement of neural net and voc classes to load up the labels
# and classes
#imageio to take input (PIL can also be used)

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#defining the function with 3 arguments frame, net, trnsform which would be handeled later 

def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

#creating the neural net
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

#doing some transformations to make the video compatible with the pretrained neural net
transform = BaseTransform(net.size, (104/256.0 , 117/256.0, 123/256.0))

#getting the frames of the videos applying detect on them and appending the resultant frames into the ouput frame
reader = imageio.get_reader('epic-horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('object_detections.mp4', fps = fps)
for i,frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()



    