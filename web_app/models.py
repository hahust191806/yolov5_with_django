import os
from django.db import models

# Create your models here.

class ResultModel(models.Model):
    # Lưu trữ tên của ảnh trong model
    name = models.CharField(max_length=100)
    # Khởi tạo trường lưu trữ ảnh trên đĩa cụ thể là thư mục media, sau khi tải lên một hình ảnh, django sẽ lưu đường dẫn đến tệp hình ảnh trong cơ sở dữ liệu và lưu trữ tệp trên đĩa. 
    image = models.ImageField(upload_to='media/')
    def __str__(self):
        return self.title()