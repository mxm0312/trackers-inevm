# Tracker Pipelines

### **SORT Tracker**

First of all you need to build an image for SORT Tracker:
```
cd sort
docker build -t sort-image .
```

#### 1. Evaluate / Markup mode

You need to run the container and mount all of the necessary directories
**For evaluation / Markup** You need to mount `dataset_dir` (with videos to markup) and `output_dir` (where to save markup in COCO format)
Also you need to mount folders with sort tracker code: `common` and `sort`:
```
docker run -it --rm -v $(pwd)/common:/tracker_demo/common \
                    -v $(pwd)/sort:/tracker_demo/sort \
                    -v $(pwd)/dataset_dir:/tracker_demo/dataset \
                    -v $(pwd)/output_dir:/tracker_demo/output sort-image bash
```
Then from the terminal:
```
cd sort
python3 evaluate.py ../common/yolov5s.pt ../dataset ../output
```
Finally markup jsons in COCO format will be saved in the `output_dir` folder

### ByteTracker

First of all you need to build an image for ByteTracker:
```
cd bytetracker
docker build -t bytetracker-image .
```

#### 1. Evaluate / Markup mode

You need to run the container and mount all of the necessary directories
**For evaluation / Markup** You need to mount `dataset_dir` (with videos to markup) and `output_dir` (where to save markup in COCO format)
Also you need to mount folders with sort tracker code: `common` and `bytetracker`:
```
docker run -it --rm -v $(pwd)/common:/tracker_demo/common \
                    -v $(pwd)/bytetracker:/tracker_demo/bytetracker \
                    -v $(pwd)/dataset_dir:/tracker_demo/dataset \
                    -v $(pwd)/output_dir:/tracker_demo/output sort-image bash
```
Then from the terminal:
```
cd bytetracker
python3 evaluate.py ../common/yolov5s.pt ../dataset ../output
```
Finally markup jsons in COCO format will be saved in the `output_dir` folder

## Visualization stage:
After we got jsons with markup from the previous steps, we can visualize Tracks by running this line of code:
```
cd common
python3 visualize.py {video_path} {markup_json_path} {output_path}
```