# 图像阈值处理（python-opencv）

## 一.全局阈值化处理

1.简单阈值当然是最简单，选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像了。函数为**cv2.threshold()** ，这个函数有四个参数，**第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数**，常用的有： 
• cv2.THRESH_BINARY（黑白二值） 
• cv2.THRESH_BINARY_INV（黑白二值反转） 
• cv2.THRESH_TRUNC （得到的图像为多像素值） 
• cv2.THRESH_TOZERO 

• cv2.THRESH_TOZERO_INV 

该函数有**两个返回值**，第一个retVal（**得到的阈值值**），第二个就是**阈值化后的图像**。

## 二.局部阈值化处理（自适应阈值化）

而自适应阈值可以看成一种局部性的阈值，通过规定一个区域大小，比较这个点与区域大小里面像素点的平均值（或者其他特征）的大小关系确定这个像素点是属于黑或者白（如果是二值情况）。使用的函数为：**cv2.adaptiveThreshold（）** 
该函数需要填6个参数：

- 第一个原始图像

- 第二个像素值上限

- 第三个自适应方法Adaptive Method: 
  — cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值 
  —cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权重为一个高斯窗口

- 第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV

- 第五个Block size:规定领域大小（一个正方形的领域）

- 第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） 

  ```pyhton3
  import cv2
  import matplotlib.pyplot as plt
  
  img = cv2.imread('flower.jpg',0) #直接读为灰度图像
  ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
  cv2.THRESH_BINARY,11,2) #换行符号 \
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  cv2.THRESH_BINARY,11,2) #换行符号 \
  images = [img,th1,th2,th3]
  plt.figure()
  for i in xrange(4):
      plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
  plt.show()
  
  ```

  

这种方法理论上得到的效果更好，相当于**在动态自适应的调整属于自己像素点的阈值**，而不是整幅图像都用一个阈值。

## 三、Otsu二值化

### 3.1原理

OTSU[算法](http://lib.csdn.net/base/datastructure)也称最大类间差法，有时也称之为大津算法，由大津于1979年提出，被认为是图像分割中阈值选取的最佳算法，计算简单，不受图像亮度和对比度的影响，因此在数字图像处理上得到了广泛的应用。它是按图像的灰度特性,将图像分成背景和前景两部分。因方差是灰度分布均匀性的一种度量,背景和前景之间的类间方差越大,说明构成图像的两部分的差别越大,当部分前景错分为背景或部分背景错分为前景都会导致两部分差别变小。因此,使类间方差最大的分割意味着错分概率最小。

对于图像I(x,y)，前景(即目标)和背景的分割阈值记作T，属于前景的像素点数占整幅图像的比例记为ω0，其平均灰度μ0；背景像素点数占整幅图像的比例为ω1，其平均灰度为μ1。图像的总平均灰度记为μ，类间方差记为g。

假设图像的背景较暗，并且图像的大小为M×N，图像中像素的灰度值小于阈值T的像素个数记作N0，像素灰度大于阈值T的像素个数记作N1，则有：
　　　　　　ω0=N0/ M×N (1)
　　　　　　ω1=N1/ M×N (2)
　　　　　　N0+N1=M×N (3)
　　　　　　ω0+ω1=1　　　 (4)
　　　　　　μ=ω0*μ0+ω1*μ1 (5)
　　　　　　g=ω0(μ0-μ)^2+ω1(μ1-μ)^2 (6)
将式(5)代入式(6),得到等价公式:
　　　　　　g=ω0ω1(μ0-μ1)^2 　　 (7)　这就是类间方差
采用遍历的方法得到使类间方差g最大的阈值T,即为所求。

### 3.2 Otsu算法图解

对于直方图有两个峰值的图像，大津法求得的Ｔ近似等于两个峰值之间的低谷。

`Otsu过程：` 

1. `计算图像直方图；` 
2. `设定一阈值，把直方图强度大于阈值的像素分成一组，把小于阈值的像素分成另外一组；` 
3. `分别计算两组内的偏移数，并把偏移数相加；` 
4. `把0~255依照顺序多为阈值，重复1-3的步骤，直到得到最小偏移数，其所对应的值即为结果阈值。`
---------------------


[OpenCV](http://lib.csdn.net/base/opencv)的二值化操作中，有一种“大津阈值处理”的方法，使用函数

cv2.threshold(img,0,255,**cv2.THRESH_BINARY+cv2.THRESH_OTSU**)

### 3.3 Otsu 代码

- ```python3
  import cv2
  import matplotlib.pyplot as plt
  
  img = cv2.imread('finger.jpg',0) #直接读为灰度图像
  #简单滤波
  ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  #Otsu 滤波
  ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  print ret2
  plt.figure()
  plt.subplot(221),plt.imshow(img,'gray')
  plt.subplot(222),plt.hist(img.ravel(),256)#.ravel方法将矩阵转化为一维
  plt.subplot(223),plt.imshow(th1,'gray')
  plt.subplot(224),plt.imshow(th2,'gray')
  
  ```

- ```C++
  int MyAutoFocusDll::otsuThreshold(IplImage *frame)
  {
      const int GrayScale = 256;
      int width = frame->width;
      int height = frame->height;
      int pixelCount[GrayScale];
      float pixelPro[GrayScale];
      int i, j, pixelSum = width * height, threshold = 0;
      uchar* data = (uchar*)frame->imageData;  //指向像素数据的指针
      for (i = 0; i < GrayScale; i++)
      {
          pixelCount[i] = 0;
          pixelPro[i] = 0;
      }
  
      //统计灰度级中每个像素在整幅图像中的个数  
      for (i = 0; i < height; i++)
      {
          for (j = 0; j < width; j++)
          {
              pixelCount[(int)data[i * width + j]]++;  //将像素值作为计数数组的下标
          }
      }
  
      //计算每个像素在整幅图像中的比例  
      float maxPro = 0.0;
      int kk = 0;
      for (i = 0; i < GrayScale; i++)
      {
          pixelPro[i] = (float)pixelCount[i] / pixelSum;
          if (pixelPro[i] > maxPro)
          {
              maxPro = pixelPro[i];
              kk = i;
          }
      }
  
      //遍历灰度级[0,255]  
      float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
      for (i = 0; i < GrayScale; i++)     // i作为阈值
      {
          w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
          for (j = 0; j < GrayScale; j++)
          {
              if (j <= i)   //背景部分  
              {
                  w0 += pixelPro[j];
                  u0tmp += j * pixelPro[j];
              }
              else   //前景部分  
              {
                  w1 += pixelPro[j];
                  u1tmp += j * pixelPro[j];
              }
          }
          u0 = u0tmp / w0;
          u1 = u1tmp / w1;
          u = u0tmp + u1tmp;
          deltaTmp = w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
          if (deltaTmp > deltaMax)
          {
              deltaMax = deltaTmp;
              threshold = i;
          }
      }
  
      return threshold;
  }
  ```

  



