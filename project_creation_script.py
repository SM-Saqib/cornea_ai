import os

project_name = "./"

# Create the project directory
# os.makedirs(project_name, exist_ok=True)

# Create the data directory and subdirectories
data_dir = os.path.join(project_name, "data")
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "annotations"), exist_ok=True)

# Create the models directory and subdirectories
models_dir = os.path.join(project_name, "models")
os.makedirs(os.path.join(models_dir, "yolo", "weights"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "yolo", "config"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "resnet", "weights"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "mog2"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "ocr"), exist_ok=True)

# Create the scripts directory
scripts_dir = os.path.join(project_name, "scripts")
os.makedirs(scripts_dir, exist_ok=True)

# Create the utils directory
utils_dir = os.path.join(project_name, "utils")
os.makedirs(utils_dir, exist_ok=True)

# Create the Python files and add comments with their filenames
def create_python_file(filepath, comment):
    with open(filepath, "w") as f:
        f.write(f"# {comment}\n")

# Models
create_python_file(os.path.join(models_dir, "yolo", "yolo_detector.py"), "yolo_detector.py")
create_python_file(os.path.join(models_dir, "resnet", "resnet_classifier.py"), "resnet_classifier.py")
create_python_file(os.path.join(models_dir, "mog2", "mog2_detector.py"), "mog2_detector.py")
create_python_file(os.path.join(models_dir, "ocr", "ocr_detector.py"), "ocr_detector.py")

# Scripts
create_python_file(os.path.join(scripts_dir, "train_resnet.py"), "train_resnet.py")
create_python_file(os.path.join(scripts_dir, "train_yolo.py"), "train_yolo.py")
create_python_file(os.path.join(scripts_dir, "process_video.py"), "process_video.py")

# Utils
create_python_file(os.path.join(utils_dir, "preprocessing.py"), "preprocessing.py")
create_python_file(os.path.join(utils_dir, "visualization.py"), "visualization.py")
create_python_file(os.path.join(utils_dir, "helper.py"), "helper.py")

# Create requirements.txt
with open(os.path.join(project_name, "requirements.txt"), "w") as f:
    f.write("# Add your project dependencies here\n")

print(f"Project structure '{project_name}' created successfully.")