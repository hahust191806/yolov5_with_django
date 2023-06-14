# Deploying YOLOV5 with Django Framework for Object Detection Task

This is a basic project that I just finished learning Django, this application allows user to upload images from their localhost, process them using YOLOV5, and return the result to the current web interface. Additionally, the processed image will be saved to the media folder. I will add more features, complete the app, and develop it into a product that everyone can use. 

### A project to demonstrate easy integration of YoloV5 in Django WebApp

Note: This is not a full-fledged production ready app though can be scaled to work as one.

### Features of the WebApp 

* Upload Image. 
* Convert uploaded image size to 640x640.
* Detect object on an image with YoloV5. 
* Some additional features will be added in the near future. 

## Steps to use locally 
```bash
clone the repo locally

# create virtual env 
python -m venv env_name

# install packages 
pip install django 
pip install yolov5 
pip install opencv-python 
pip install torch

# migrate 
python manage.py makemigrations 
python manage.py migrate 

# create super user 
python manage.py createsuperuser 

# run server 
python manage.py runserver
