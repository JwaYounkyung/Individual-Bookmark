# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

import time
import pafy
import cv2
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

url = 'https://www.youtube.com/watch?v=P8jOQUsTU9o'
video = pafy.new(url)
best = video.getbest(preftype='mp4') # Selects the stream with the highest resolution

dist_thresh = 1.2

def collate_fn(x):
    return x[0]

# initilaize the embedding list of the comparative group
def initialize(mtcnn, resnet):

    dataset=datasets.ImageFolder('photos') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names
    # {0: 'angelina_jolie', 1: 'bradley_cooper', 2: 'kate_siegel', 3: 'paul_rudd', 4: 'shea_whigham', 5: 'taylor_swift'}
    
    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        faces, prob = mtcnn(img, return_prob=True) 

        for face in faces:
            if face is not None and prob>0.90: # if face detected and porbability > 90%
                emb = resnet(face.unsqueeze(0).to(device)) # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
                name_list.append(idx_to_class[idx]) # names are stored in a list

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file


# draw the rectangular and name in video
def extract_face_info(img, box, name, min_dist):
    (x1, y1, x2, y2) = box.tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2) # draw face rectangle

    if min_dist < dist_thresh:
        cv2.putText(img, "Face : " + name, (x1, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img, "Dist : " + str(min_dist), (x1, y2 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, 'No matching faces', (x1, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


# face verification
def recognize_face(img, face, database, network):

    embedding_list = database[0] # getting embedding data
    name_list = database[1] # getting list of names
    
    dist_list = [] # list of matched distances, minimum distance is used to identify the person

    emb = network(face.unsqueeze(0).to(device)).detach() # detech is to make required gradient false

    # photos에 있는 사진 중에 가장 비슷한 이미지가 무엇인지 판별
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    
    name = name_list[idx_min]
    min_dist = min(dist_list)

    return name, min_dist

real_sec = [8,9,10,16,17,18,19,20,21,22,23,24,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,116,117,118,119,126,127,128,129,130,131,132,133,149,150,151,152,153,154,155,156,164,174,175,176,177,178,179,195,196,215,216,217,218,219,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,264,265,266,265,268,269]
real_frame = list()
captured_frame = list()

def recognize():
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20, keep_all=True, device=device) # initializing mtcnn for face detection
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # initializing resnet for face img to embeding conversion

    initialize(mtcnn, resnet)
    cap=cv2.VideoCapture(best.url) # use youtube

    saved_data = torch.load('data.pt') # loading data.pt file
    
    frame = 0
    # frame 단위
    while True:
        frame += 1
        # face detection
        ret, img = cap.read() # ret 정상적으로 읽어 왔는지
        
        if ret: 
            faces, boxes = mtcnn(img) # returns cropped face and bounding box
            
            # 한 frame 당 잡힌 얼굴 단위
            if (faces != None): # 영상에서 얼굴이 잡히지 않을 수도 있음
                for i, face in enumerate(faces):
                    name, min_dist = recognize_face(img, face, saved_data, resnet)
                    # extract_face_info(img, boxes[i], name, min_dist)

                    if(name == 'chris_martin'):
                        captured_frame.append(frame)

            '''
            cv2.imshow('Recognizing faces', img) 
            
            if cv2.waitKey(1) == ord('q'):
                break
            '''
        else:
            break


    
    for i in real_sec: #초 프레임으로 변경
        for j in range(30):
            real_frame.append(i*30+j)
                
    print(real_frame)
    print(captured_frame)
    sum = 0
        
    for i in real_frame:
        if i in captured_frame:
            sum += 1  
    print("Accuracy: {:.2f} % \n".format(((float(sum))/(len(real_frame)))*100))   
    
    cap.release()
    # cv2.destroyAllWindows()

    
    

recognize()