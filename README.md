# FEniCS-on-GPU

FEniCS on GPU takes advantage of CUDA cores to solve SPARSE matrix using cuPy and SciPy libraries.\
Here we first obtain assembled matrix from bilinear and linear form.\
Assembled Bilinear Matrix is convetred into SPARSE Matrix by using SciPy subroutines.\
SPARSE Matrix is transfered to cuPyx Multidimensional array.\
Here we have used lsqr (Least Square) to solve. \
(You can use any of the available solver from here https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html \
Results = We got arround 38X speed up using Least Square (For 63000 Grid Points). (Speed will vary depending upon solver types and configuration of Machine).\
You can also use multiple GPU's see documentation here https://docs.cupy.dev/en/stable/


