# Face verification

## Requirements
``` 
Python >= 3.7, PyTorch >= 1.7.0, torchvision >= 0.8.1, facenet-pytorch==2.5.2, pafy==0.5.5, youtube-dl==2021.5.16
```

#### before run this code, you need to fix mtcnn.py in facenet-pytorch package
#### line 270 'return faces' -> 'return faces, batch_boxes'

## Run
```
python face_verification.py
```

## Process 
1. Put the input 
    + Youtube URL, 타임라인을 얻길 원하는 인물
2. Comparative group embedding matrix 형성
    + photos에 있는 모든 사진에 대한 각각의 embedding matrix 저장
    + embedding matrix shape : torch.Size([1, 512])
3. 프레임 별로 face가 각각 어떤 사람인지 판별
    + 위에 형성한 embedding matrix와 영상에 나오는 인물의 embedding matrix를 비교해 현재 이미지와 가장 비슷한 이미지가 무엇인지 판별
4. 특정 인물이 나오는 타임라인 형성
    + 몇 번째 프레임에 특정 인물이 나오는지 print

## Algorithm 
+ Face Detection
    + MTCNN
+ Face Verification
    + InceptionResnet v1

## Files
+ face_verification.py 
    + 주요 코드
+ photos
    + target datasets
+ dataset
    + photos에 넣고 사용한 데이터 셋
+ facenet_trainig
    + 자세한 내용 해당 폴더 안 README 참조



## Result
+ 브레이브 걸스 1
    + 영상 : https://www.youtube.com/watch?v=GoBLWClS4ts 
    + 대상 : 은지
    + real_sec [22,23,24,25,26,27,28,29,54,55,56,57,58,59,60,85,86,87,88,89,90,91,92,116,117,118,119,120,121,122,148,149,150,151,152,153,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199]
    + 정확도
        + 데이터셋 1 정확도 : 40.20%
        + 데이터셋 2 정확도 : 25.62 % 

+ 브레이브 걸스 2
    + 영상 : https://www.youtube.com/watch?v=155cI2v1l-s
    + 대상 : 은지
    + real_sec [8,9,10,11,17,18,19,20,21,24,28,29,30,31,32,33,34,35,36,37,38,41,42,43,44,45,46,59,60,61,67,68,69,72,73,76,77,111,112,118,119,120,124,125,130,131,136,137,138,139,141,142,143,153,154,156,157,158,159,160,177,178,188,189,190]
    + 정확도
        + 데이터셋 1 정확도 : 66.26%
        + 데이터셋 2 정확도 : 64.10%

+ Coldplay
    + 영상 : https://www.youtube.com/watch?v=P8jOQUsTU9o
    + 대상 : chris martin
    + real_sec [8,9,10,16,17,18,19,20,21,22,23,24,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,116,117,118,119,126,127,128,129,130,131,132,133,149,150,151,152,153,154,155,156,164,174,175,176,177,178,179,195,196,215,216,217,218,219,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,264,265,266,265,268,269]
    + 정확도 : 70.57

## References

1. facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

