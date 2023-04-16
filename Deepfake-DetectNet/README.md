# Exploring Deepfake Detection with DetectNet

"Deepfakes" refer to fake media, like pictures and videos, that are developed using deep neural networks. Unlike fake media created using Photoshop, these forgeries are _almost indistinguishable_ from the real thing.

In this project, I explore the world of Deepfakes using a pre-trained model engineered to detect them, known as **DetectNet**.

Dataset
==
Small dataset : </br>
Deepfakes photos : 2,845 frames</br>
Real  photos : 4,259 frames</br>
Total : 7,104</br>

</br>
</br>

Data augmentation
==
![image](readme_image/data_augmentation.png)


</br>
</br>

Predict
==
![image](readme_image/predict.png)

</br>
</br>

You will see the 4 states
==

Real
--
![image](readme_image/real.png)

Deepfakes
--
![image](readme_image/deepfakes.png)

Mistaken deepfakes (actually the image is real)
--
![image](readme_image/mistaken_deepfakes_real.png)


Mistaken real (actually the image is fake)
--
![image](readme_image/mistaken_real_fake.png)


Result
==
Real : 3748</br>
Mistaken for fake : 512 (誤認成為假的，其實是真的)</br>
DeepFake : 2563</br>
mistaken for real : 281 (誤認成為真的，其實是假的)</br>
total : 7104</br>
</br>
正確率 : 0.88</br>
錯誤率 : 0.12</br>
