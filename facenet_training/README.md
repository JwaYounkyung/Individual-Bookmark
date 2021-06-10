# Facenet Training

## Requirements
requirements.txt

## Experiments
1. Training Network with own data and Save
```
python src/classifier.py TRAIN datasets/own/own_mtcnnalign_160/train/ src/models/facenet/20180402-114759/20180402-114759.pb src/models/my_classifier.pkl --batch_size 100 
```
+ training한 모델 my_classifier.pkl로 저장 
2. Test Network with own data
```
python src/classifier.py CLASSIFY datasets/own/own_mtcnnalign_160/test/ src/models/facenet/20180402-114759/20180402-114759.pb src/models/my_classifier.pkl --batch_size 100 
```
+ test 결과 src/models/my_classifier_test.txt에 저장 되어 있음
3. Validate the Network with LFW dataset
```
python src/validate_on_lfw.py datasets/lfw/lfw_mtcnnpy_160 src/models/facenet/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization
```
+ 결과 


## References

1. facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)


