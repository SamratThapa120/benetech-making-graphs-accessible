# benetech-making-graphs-accessible
This is my code for training a two-stage (YOLOv7 Object detector+ OCR model) for the "Benetech - Making Graphs Accessible" kaggle competition. I got 20th position in the competition.

# Using ML to create tabular data from graphs
My approach to the problem involves two main steps: object detection and Optical Character Recognition(OCR)

#### Object Detection:
I trained a yolov7 model to detect the x-axis labels, y-axis labels, the chart bounding box, and the data points on the chart. I was able to accurately compute the position of the datapoints on the chart images by linear interpolation of the x-axis and y-axis tick coordinates, and the x-axis/y-axis labels. You can find the notebook I used to create the dataset to train the object detection model here. Here are some examples of the object detection dataset.

This approach also works well for scatter plots, where some models like Donut fail.
Some notes:
*This object detection model also works as the chart classification model. 
*There are overlapping bounxing boxes for some x-axis labels. However, my OCR model was able to extract the correct text despite the input image including text from neiboring bboxes. 

#### OCR model:
Using the EasyOCR libary, I trained a ResNet(feature extractor)+BidirectionalLSTM model with Connectionist Temporal Classification(CTC) loss. I would like to thank ____ for this notebook, I had to make several modifications to the inference and training pipeline, but it was a good starting point. The additional ICSR dataset improved the accuracy of the OCR model by about 5% from 84% to 89%.


I participated in this competition only for the last 4 weeks. So, due to lack of time, I wasnt able to try out other end-to-end approaches. I think there is a lot of room for improvement for this model. For example,about 25% of the predictions made by the model get automatically scored 0, because of mismatching number of predictions. This mismatch is due to only 1 or 2 points for charts besides scatter plot. 

# Steps to reproduce results
## Download competition dataset and additional dataset
First, download the following three datasets:
Competition dataset
ISCR 2023 training dataset
ISCR 2023 test dataset
## Create datasets for Detection model and OCR
Run the following three notebooks to create the datasets for both detection model, and OCR model
OCR dataset:
create_OCR_dataset_COMPETION+ADDITIONAL_DATA.ipynb

Yolov7 dataset with additional data
create_yolo_dataset_ADDITIONAL_DATA.ipynb
Yolov7 dataset with competition data
create_yolo_dataset_COMPETITION_DATA.ipynb
## Run training scripts for Detection model and OCR
Train yolov7 model
    cd yolov7
    bash train.sh
Train OCR model
    cd OCR
    bash train.sh
# Model-checkpoints
The checkpoints of the model that I trained are listed in the following directories:s
yolov7:
    yolov7/runs/train/yolov7-custom-sgd

OCR:
    OCR/saved_models/en_filtered_old