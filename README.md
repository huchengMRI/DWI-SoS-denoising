# DWI-SoS-denoising
A CNN network for reducing SoS noise in diffusion weighted images
This works on tensorflow version 1.14.

The script CNN_DWI_Denoising_ABCD_Optimize.py runs to find the optimal learning rate and batch size in terms of loss, MSE of FA, correlation between signals. The script  Apply_CNN_Denoising.py apply the learning model to actual data using optimal parameters derived from the last step.
