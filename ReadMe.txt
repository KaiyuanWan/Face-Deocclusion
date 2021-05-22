1.test.py : 
	This python file is for testing. You just need replace the 'opt.test_dir' with your own test data path or using dataset proivided by us and the test file will output three files in './result' which are detect result(detect_mask), face parsing result(parsing_mask) and face de-occlusion result(rec_face), respectively.
2.models
	This folder contains two python file: 'generator.py' and 'ops.py'. The 'generator.py' consists of four parts, which are detection part, parsing part, reconstruction part, discriminator part. And 'ops.py' contains the operators used in 'generator.py'.
3.trained_models
	This file contains two trained models:'detect_parsing.pth.tar' for detection and parsing, 'reconstruction.pth.tar' for reconstruction. We put the trained model in Google Drive, you can download the model by this url:
	(https://drive.google.com/drive/folders/1zNf6HATv3QtrKqGIQBq4af5qQAJ1RUcj?usp=sharing), and then put the trained model in this folder.
4.result
	The result will be saved in this folder.
5.test_image
	This is test images we provide. You can use them to verificate our method. Of course, you can also test on your own dataset.
6.enviroment
	numpy                     1.17.1 
	python                    3.6.8 
	torch                     1.0.1.post2 
	opencv-python             4.1.2.30
	Pillow                    6.1.0
	CUDA                      8.0.61
	CUDNN			  5.1.5
