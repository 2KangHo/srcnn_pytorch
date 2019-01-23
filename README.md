# srcnn_pytorch
PyTorch Implementation

## Requirement
torch
torchvision
python-tk (or pyhton3-tk)

## Training

	> python main.py --upscale_factor 3 --batch_size 10 --cuda --test_batch_size 20 --epochs 100 --lr 0.01

or

	> python3 main.py --upscale_factor 3 --batch_size 10 --cuda --test_batch_size 20 --epochs 100 --lr 0.01

## Sample Usage

	> python run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_100.pth --cuda --output_filename <output_filename>

or

	> python3 run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_100.pth --cuda --output_filename <output_filename>

# 주의
sample에서 input image는 학습에 사용된 BSDS300 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.
