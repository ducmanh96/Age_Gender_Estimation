# Sử dụng hình ảnh chính thức của Python làm hình ảnh cha
FROM python:3.7-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép các nội dung thư mục hiện tại vào container tại /app
COPY . /app

# Sao chép OpenVINO vào container
COPY openvino /opt/openvino

# Cài đặt các gói cần thiết
RUN apt-get update && \
    apt-get install -y \
    wget \
    sudo \
    libtbb-dev \
    libtbbmalloc2 \
    libopencv-dev \
    cmake \
    libssl-dev \
    build-essential \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    pciutils && \
    rm -rf /var/lib/apt/lists/* 

# Thiết lập OpenVINO
ENV INTEL_OPENVINO_DIR /opt/openvino
ENV PATH="$INTEL_OPENVINO_DIR/bin:$PATH"
RUN /bin/bash -c "source /opt/openvino/setupvars.sh"

# Cài đặt các gói Python
RUN pip install --upgrade pip && \
    pip install \
    opencv-python-headless \
    flask \
    openvino \
    opencv-python \
    numpy 

# Mở cổng 5000 cho ứng dụng Flask
EXPOSE 5000

# Định nghĩa biến môi trường để Flask chạy trên localhost
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Thêm quyền truy cập vào thiết bị camera khi chạy container
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
