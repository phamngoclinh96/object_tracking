## Object Tracking

Object tracking sử dụng yolo và deep sort

Cài thư viện cần thiết
```sh
pip install -r requirements.txt
```
Tải file [yolo3.weights](https://drive.google.com/file/d/16V0rDO0tuv9p_E6jwp3Cdnir1pBr_gic/view?usp=sharing) và lưu vào thư mục yolo

Chạy test theo cú pháp
```
python3 test_on_video.py [path input] [path output]
```

Ví dụ
```
python test_on_video.py input.avi output.avi
```
