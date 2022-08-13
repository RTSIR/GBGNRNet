# GBGNRNet
This repo contains the KERAS implementation of "Gradient and Multi Scale Feature Inspired Deep Blind Gaussian Denoiser"

# Run Experiments

To test for blind gray denoising using GBGNRNet write:

python Test_gray.py

The resultant images will be stored in 'Result/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.

# Train GBGNRNet denoising network

To train the GBGNRNet denoising network, first download the clean image training patch from [here](https://drive.google.com/file/d/1GjNTNadXaTgruckfq8tTY7Vu8fK8zDj9/view?usp=sharing) and copy the 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the GBGNRNet model file using:

python GBGNRNet_Gray.py

This will save the 'GBGNRNet_Gray.h5' file in the folder 'Pretrained_models/'.

# GBGNRNet image denoising visual comparison

![Screenshot 2022-08-13 at 5 02 02 PM](https://user-images.githubusercontent.com/89151608/184490258-efa08c8b-35eb-4cb6-8376-ab4f75333ed8.png)
![Screenshot 2022-08-13 at 5 02 36 PM](https://user-images.githubusercontent.com/89151608/184490307-ea9f5a0e-084e-4442-9537-91e03fd4667c.png)
![Screenshot 2022-08-13 at 5 03 07 PM](https://user-images.githubusercontent.com/89151608/184490382-5a68f0ed-b9aa-4028-8cb9-e9dda1ce77f8.png)

# Citation
@article{thakur2022gradient,
  title={Gradient and Multi Scale Feature Inspired Deep Blind Gaussian Denoiser},
  author={Thakur, Ramesh Kumar and Maji, Suman Kumar},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
