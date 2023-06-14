from yolov5.utils.torch_utils import select_device 
from yolov5.utils.general import check_img_size
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (non_max_suppression, scale_boxes)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import LOGGER
import torch 
import cv2 
import numpy as np 
import time 
import os 

class Yolov5: 
    def __init__(self):
        self.device = "" # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.weights = "yolov5n.pt" # model path or triton URL
        self.conf = 0.25 # confidence threshold
        self.iou = 0.45 # NMS IOU threshold
        self.imgsz = (640,640) # inference size (height, width)
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.names = ""
        self.max_det = 1000  # maximum detections per image
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.half = False  # use FP16 half-precision inference
        self.data = "data/coco128.yaml" # dataset.yaml path
        self.agnostic_nms = False  # class-agnostic NMS
        self.model = None 
        self.agnostic_nms = False
        self.auto = True 
    
    # Load model     
    def set_up_model(self, weights):
        device = select_device(self.device)
        self.device = device
        self.weights = weights
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half) 
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)
        self.names = self.model.names
        self.model.eval() # chuyển mô hình sang chế độ evalution để inference
    
    # Convert img to torch 
    # Data Loader 
    def preprocess(self, img):
        # Resize and pad image while meeting stride-multiple constraints
        img = letterbox(img, self.imgsz, stride=self.model.stride, auto=self.auto)[0]
        # cv2.read() load an images with HWC, BGR layout (height, width, channels), while pytorch requires CHW, RGB layout--> .transpose((2, 0, 1))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        """
        Mảng liền kề là một mảng trong đó các phần tử được lưu trữ ở các vị trí bộ nhớ liền kề, giúp truy cập các phần tử trong mảng nhanh hơn. 
        Trong học sâu, nhiều thư viện và khung yêu cầu dữ liệu đầu vào ở định dạng liền kề, đặc biệt là khi sử dụng các tài nguyên điện toán hiệu suất cao như GPU hoặc TPU.
        Bằng cách sử dụng hàm ascontiguousarray để chuyển đổi hình ảnh đầu vào thành một mảng liền kề, bạn đảm bảo rằng dữ liệu hình ảnh được lưu trữ theo cách tối ưu cho mô hình học sâu đang được sử dụng. 
        Điều này có thể giúp cải thiện hiệu suất của mô hình và giảm thời gian cần thiết để đào tạo hoặc suy luận.
        """
        img = np.ascontiguousarray(img)  # contiguous
        # convert image from numpy to torch 
        img = torch.from_numpy(img).to(self.device)
        # là 1 biểu thức, chuyển đổi mảng numpy thành a half precision floating-point
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img
    
    # Hậu xử lý ảnh
    def postprocess(self, preds, img, orig_img):
        preds = non_max_suppression(preds, self.conf, self.iou, agnostic=self.agnostic_nms, max_det=self.max_det, classes=self.classes)
        
        results = []
        for i, det in enumerate(preds):
            if len(det):
                orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
                shape = orig_img.shape
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    results.append([x1, y1, x2, y2, conf, cls])
        return results
        
    # Quá trình inference
    def inference(self, img):
        result = []
        image_copy = img.copy()
        cv2.imwrite("test.jpg", img)
        img = self.preprocess(img)    
        pre_time = time.time()
        preds = self.model(img, augment=False, visualize=False)
        print("inference_time: ", int((time.time() - pre_time)*1000), "ms")
        results = self.postprocess(preds, img, image_copy)
        for result in results: 
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_copy, str(self.model.names[int(cls)]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
        return image_copy
    
    # Vẽ box và hiển thị score 
    def plot(self, results, img):
        for result in results: 
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, str(self.model.names[int(cls)]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img