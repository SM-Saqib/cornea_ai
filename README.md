Start of a Vision project
Aptly named as Cornea

FOR LARGE SCALE OBJECT DETECTION

Structure of the project:

1. Data Preprocessing
2. Model FINE TUNING
3. Model Evaluation
4. Model Deployment


We are using a pipeline of multiple models to achieve the desired results. The models are as follows:
YoloV3
Resenet
MOG2 model (running in the background)

For data pipeline, we can use either of the following:
1. Gstreamer
2. OpenCV

For inference, we will use OpenCV

For model training, we will use Pytorch, and develop scripts to make this a recurrent and automated process, we will only train Resnet model, because this will be used to actually classify our object. Yolo will be used just for localizing and detecting the category of the object.

The  inference flow will look like this:

1. Capture the video feed -> Preprocess the video feed -> Run MOG2 model to see if there is any motion detected -> If yes, then extract a larger bounding box around the moving object -> then run the YoloV3 model to detect category/object and localize the object in the bounding box. -> Extract crops of the detected objects -> Optionally run the Resnet model for the relevant category on the crops to classify the object, if further focused detection is required -> Display the results on the screen with a bounding box around the object and the class of the object.

2. We may have some more models in the pipeline, for example, we will have OCR model if text is detected.

3. MOG2 model will be keeping  a reference background after some time, to make sure the moving object event is over. We dont want to make the moving object to disappear by becoming part of the background.


The above is to simulate a natural object detections by eye-sight.
A person is unfocused and just looking in the distance. A moving item captures his attention, and he focuses on it. If its something interesting, he focuses some more to accurately detect the object, and takes his time on that. Finally he then goes back to his unfocused state.

Key Aspects:

Efficiency: The pipeline prioritizes efficiency by using MOG2 for initial motion detection, avoiding unnecessary processing of every frame.
Hierarchical Detection: The system uses a hierarchical approach, starting with coarse detection (YOLO) and moving to finer classification (ResNet) only when necessary.
Adaptability: The inclusion of OCR and the potential for other models makes the system adaptable to different scenarios.
Human like simulation: The system is built to imitate how a human detects and focuses on objects.

project structure:
object_detection/
├── data/
│   ├── raw/           # Raw video and image data
│   ├── processed/     # Processed data for training/inference
│   ├── models/        # Trained models (YOLO, ResNet, OCR)
│   └── annotations/   # Annotation files for training
├── models/
│   ├── yolo/          # YOLOv3 related files
│   │   ├── weights/   # YOLO weights
│   │   ├── config/    # YOLO config files
│   │   └── yolo_detector.py #Yolo detection class
│   ├── resnet/        # ResNet related files
│   │   ├── resnet_classifier.py #Resnet classification class
│   │   └── weights/   # Resnet weights
│   ├── mog2/          # MOG2 related files
│   │   └── mog2_detector.py #MOG2 detection class
│   ├── ocr/           # OCR related files
│   │   └── ocr_detector.py #OCR detection class
├── scripts/
│   ├── train_resnet.py  # Script to train ResNet
│   ├── train_yolo.py    # Script to train YOLO
│   ├── process_video.py # Script to run the inference pipeline
├── utils/
│   ├── preprocessing.py # Image/video preprocessing functions
│   ├── visualization.py # Functions for displaying results
│   └── helper.py      # General helper functions
├── requirements.txt 

