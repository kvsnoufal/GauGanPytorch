# GauGan: Generating Photorealistic Images from Drawings
Paper To Code implementation of NVIDIA's GauGan on a custom Landscape 's Dataset. Generating photorealistic-ish:p images from drawings

Origina Paper [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)

Gaugan uses a special normalization technique for improving the quality of the data. The generator is capable of taking as input a semantic map (a drawing) and generating a photorealistic image as the output. Further it is also capable of multimodal image synthesis - which means, it can generate images in various different styles. So for the same drawing, it can generate multiple images.

In my implementation, I downloaded a dataset of landscape images from kaggle and used a pretrained semantic segmentation model (deeplab v2) to generate semantic maps of the image. This is how I compiled the dataset.

### Results:
![1.gif](https://github.com/kvsnoufal/GauGanPytorch/blob/main/doc/1.gif)
![2.png](https://github.com/kvsnoufal/GauGanPytorch/blob/main/doc/2.png)
![3.png](https://github.com/kvsnoufal/GauGanPytorch/blob/main/doc/3.png)
![4.png](https://github.com/kvsnoufal/GauGanPytorch/blob/main/doc/4.png)
![5.png](https://github.com/kvsnoufal/GauGanPytorch/blob/main/doc/5.png)

### Shoulders of Giants:
1. Semantic Image Synthesis with Spatially-Adaptive Normalization (https://arxiv.org/abs/1903.07291)
2. Official Github Implementation : https://github.com/NVlabs/SPADE
3. Implementation in Keras : https://keras.io/examples/generative/gaugan/
4. Flickr Landscape Dataset: https://www.kaggle.com/datasets/arnaud58/landscape-pictures
5. DeepLab model for semantic segmentation: https://github.com/kazuto1011/deeplab-pytorch
