from django.shortcuts import render, HttpResponse
from django.views import View
import cv2 
from .yolov5 import Yolov5
from .models import ResultModel
import numpy as np 
from django.conf import settings
import os

# Create your views here.

class Index(View):
    
    def get(self, request):
        return render(request, 'web_app/index.html')
    
    def post(self, request):
        # Khiz submit 1 form chứa các trường FileField hoặc ImageField, muốn lấy dữ liệu, phải thêm một tham số request.FILES để truy cập vào các tập tin được upload từ form 
        # Khởi tạo 1 đối tượng model và gán giá trị cho các trường thuộc tính của đối tượng đó nhằm lưu dl vào db
        form = ResultModel(request.POST, request.FILES)
        # Lấy file name từ trường 'name'
        name = request.POST.get('name')
        # Lấy file ảnh từ trường tên 'image' trong model, nó là 1 đối tượng UploadedFile, đại diện cho tập tin được upload từ form
        img = request.FILES.get('image', None)
        if img is not None: 
            # Đọc ảnh bằng opencv 
            image_data = img.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Load model
            model = Yolov5()
            model.set_up_model('weights/yolov5s.pt')
            model.model.warmup(imgsz=(1, 3, *model.imgsz))
            # inference 
            result = model.inference(img_cv2)
            # Tạo đường dẫn đến file kết quả 
            # MEDIA_ROOT là đường dẫn tuyệt đối đến thư mục lưu trữ tệp tin media của ứng dụng django, trỏ đến ổ đĩa trên máy chủ
            result_file = os.path.join(settings.MEDIA_ROOT, 'web_app', name + '.jpg')
            # MEDIA_URL là đường dẫn trên trang web để truy cập tới các tệp tin media trong MEDIA_ROOT. 
            result_url = os.path.join(settings.MEDIA_URL, 'web_app', name + '.jpg')
            cv2.imwrite(result_file, result)
            print(result_file)
            print(result_url)
            # truyền đường dẫn đến ảnh kết quả vào biến context
            return render(request, 'web_app/index.html', {'result_url' : result_url})
        else: 
            return HttpResponse('Không tìm thấy ảnh!')
            
def docs(request):
    return render(request, 'web_app/docs.html')