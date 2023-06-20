# benetech-making-graphs-accessible
This is my code for training a two-stage system ([YOLOv7](https://github.com/WongKinYiu/yolov7) Object detector+ [OCR model](https://github.com/JaidedAI/EasyOCR)) for the "[Benetech - Making Graphs Accessible](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview "Benetech - Making Graphs Accessible")" kaggle competition. I got 20th place in the competition.

# Using ML to create tabular data from graphs
My approach to the problem involves two main steps: object detection and Optical Character Recognition(OCR)
<img width="1026" alt="image" src="https://github.com/SamratThapa120/benetech-making-graphs-accessible/assets/38401989/f349dd62-6dbb-4792-8654-65d870c1b6f1">
#### Object Detection(Yolov7):
I trained a yolov7 model to detect the x-axis labels, y-axis labels, the chart bounding box, and the data points on the chart. The coordinates of the data-points were not provided in the dataset. I was able to accurately compute the position of the datapoints on the chart images by linear interpolation of the x-axis and y-axis tick coordinates with respect to the x-axis/y-axis labels (values). During inference, I inverted this process by calulating the data-series from the linear interpolation of the values of x-axis and y-axis labels with respect to the co-ordinates of the data points

This approach also works relatively well for scatter plots, compared to other approaches like Donut.

Some notes:
- This object detection model was also used as the chart-type classification model.
- There are overlapping bounxing boxes for some x-axis labels like the image below. However, my OCR model was able to extract the correct text despite the input image including text from neighbour bboxes. 
<img width="849" alt="image" src="https://github.com/SamratThapa120/benetech-making-graphs-accessible/assets/38401989/1ab49a89-1db7-48a1-9b3e-429217142b45">

#### OCR model:
Using the EasyOCR libary, I trained a ResNet(feature extractor)+BidirectionalLSTM model with Connectionist Temporal Classification(CTC) loss. The additional  dataset improved the accuracy of the OCR model by about 5% from 84% to 89%.

#### Post processing:
After receiving the bounding boxes from the model, I performed some post-processing based on some simple heuristics like: removing the data points that lie outside the chart bbox, restricting x-labels(y-labels for horizontal-bar) to lie under the chart bbox, and restricting y-labels(x-labels for horizontal-bar) to the left side of the chart bbox. 
Also, the x/y axis tick coordinates are calculated using the x/y-axis bbox, and the chart bbox. I use the nearest point that lies on the chart bbox from the center of the x/y label bbox as the respective x/y tick coordinate. I chose this approach because the precision and recall of the x/y labels was higher than the x/y axis ticks in an older version of the model. 


I participated in this competition only for the last 4 weeks. So, due to lack of time, I wasnt able to try out other approaches like Donut . I think there is a lot of room for improvement for this model. For example,about 25% of the predictions made by the model automatically get scored 0, because of mismatching number of predictions. This mismatch is due to only 1 or 2 points for charts besides scatter plot.

# Steps to reproduce results
## Download competition dataset and additional dataset
First, download the following three datasets:
[Competition dataset](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data "Competition dataset")
[ICPR 2022 CHART-Infographics UB PMC Training Dataset](https://chartinfo.github.io/toolsanddata.html "ICPR 2022 CHART-Infographics UB PMC Training Dataset")
[ICPR 2022 CHART-Infographics UB-Unitec PMC Testing Dataset](https://chartinfo.github.io/toolsanddata.html "ICPR 2022 CHART-Infographics UB-Unitec PMC Testing Dataset")
## Create datasets for Detection model and OCR
Run the following three notebooks to create the datasets for both detection model, and OCR model

OCR dataset:
`create_OCR_dataset_COMPETION+ADDITIONAL_DATA.ipynb`
Yolov7 dataset with additional data:
`create_yolo_dataset_ADDITIONAL_DATA.ipynb`
Yolov7 dataset with competition data:
`create_yolo_dataset_COMPETITION_DATA.ipynb`
## Run training scripts for Detection model and OCR
Train yolov7 model
```bash
cd yolov7
bash train.sh
```
Train OCR model
```bash
cd OCR
bash train.sh
```
# Model-checkpoints
The checkpoints of the model that I trained are listed in the following directories:s
yolov7:
`    yolov7/runs/train/yolov7-custom-sgd`

OCR:
`    OCR/saved_models/en_filtered_old`
