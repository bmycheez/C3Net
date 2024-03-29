# C3Net
This is a PyTorch implementation of the [New Trends in Image Restoration and Enhancement workshop and challenges on image and video restoration and enhancement (NTIRE 2020 with CVPR 2020)](https://data.vision.ee.ethz.ch/cvl/ntire20/) paper, [C3Net: Demoireing Network Attentive in Channel, Color and Concatenation](http://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Kim_C3Net_Demoireing_Network_Attentive_in_Channel_Color_and_Concatenation_CVPRW_2020_paper.html).

If you find our project useful in your research, please consider citing:
~~~
@InProceedings{Kim_2020_CVPR_Workshops,
author = {Kim, Sangmin and Nam, Hyungjoon and Kim, Jisu and Jeong, Jechang},
title = {C3Net: Demoireing Network Attentive in Channel, Color and Concatenation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
~~~

# Dependencies
Python 3.6.9   
PyTorch 1.4.0 

# Data
[Reference](https://competitions.codalab.org/competitions/22223#participate-get_data)

You have to sign in Codalab and apply to **NTIRE 2020 Demoireing Challenge** before getting the data. 

# Proposed algorithm
![C3Net (Track 1: Single Image)](Figures/Figure_1.png)   
![AVC_Block](Figures/Figure_2.png)   
![AttBlock](Figures/Figure_3.png)   
![ResBlock](Figures/Figure_4.png)   
![C3Net-Burst (Track 2: Burst)](Figures/Figure_5.png)   
![AVC_Block-Burst](Figures/Figure_6.png)   

# Training
Use the following command to use our training codes
~~~
python train.py
~~~
There are other options you can choose.
Please refer to train.py.

# Test
Use the following command to use our test codes
~~~
python test.py
~~~
There are other options you can choose.
Please refer to test.py.  

# Performance (PSNR/SSIM)
To use heavier model, we also used numpy to read input data, not hdf5.
[Hyung-Joon](https://github.com/Hyung-Joon) and [jisukim](https://github.com/jisus189) helped it.  
**Our best records can be derived in [the code](https://github.com/Hyung-Joon/Demoire-Burst-single-master)** <u>by changing h5 into numpy and reducing GPU memory</u>.  

|Validation Server                                                                   |PSNR    |SSIM    |Rank    |
|:-----------------------------------------------------------------------------------|:-------|:-------|:-------|
|[Track 1: Single Image](https://competitions.codalab.org/competitions/22223#results)|41.30   |0.99    |9th     |
|[Track 2: Burst](https://competitions.codalab.org/competitions/22224#results)       |40.55   |0.99    |5th     |  

![Burst_Results_List](Figures/Burst_Results_List.PNG)
  
[Testing Server Reference](https://arxiv.org/pdf/2005.03155.pdf)
|Testing Server       |PSNR    |SSIM    |Rank   |
|:--------------------|:-------|:-------|:------|
|Track 1: Single Image|41.11   |0.99    |4th    |
|Track 2: Burst       |40.33   |0.99    |5th    |  

![Final_Results](Figures/Final_Results.PNG)  

![Honorable_Mention_Award](Figures/HMA.PNG)  

# Contact
If you have any question about **Demoireing** model and the CVPR2020 challenge paper, feel free to ask me to <ksmh1652@gmail.com>.  
If you have any question about **Deblurring** model, visit [here](https://github.com/Hyung-Joon/Deblur-mobile-RCAN-Master) and feel free to ask Hyung-Joon to <013107nam@gmail.com>.  
If you have any question about using **more heavier C3Net**, visit [here](https://github.com/Hyung-Joon/Demoire-Burst-single-master) and feel free to ask jisukim to <jisus.kim189@gmail.com>.  

# Acknowledgement
Thanks for [SaoYan](https://github.com/SaoYan/DnCNN-PyTorch) who gave the implementaion of DnCNN.  
Thanks for [yun_yang](https://github.com/jt827859032/DRRN-pytorch) who gave the implementation of DRRN.  
Thanks for [BumjunPark](https://github.com/BumjunPark/DHDN) who gave the implementation of DHDN.  

Hint of color loss from [Jorge Pessoa](https://github.com/jorge-pessoa/pytorch-colors).  
Hint of concatenation and residual learning from [RDN (informal implementation)](https://github.com/lingtengqiu/RDN-pytorch).  
Hint of U-net block from [DIDN (formal implementation)](https://github.com/SonghyunYu/DIDN).  

C3Net started from [RUN](https://github.com/bmycheez/RUN).  

# More Details
Also, we won 3rd Place in [**NTIRE 2020 Challenge on Image and Video Deblurring**](https://arxiv.org/pdf/2005.01244.pdf) thanks to [Hyung-Joon](https://github.com/Hyung-Joon) and [jisukim](https://github.com/jisus189).  
The code is available at [here](https://github.com/Hyung-Joon/Deblur-mobile-RCAN-Master).  

![3rd_Place](Figures/3RD.PNG)
