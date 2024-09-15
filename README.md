# MLib
linear algebra library with built in auto differentiation 

credits and resources I used for research:

https://en.wikipedia.org/wiki/LU_decomposition

http://blog.ezyang.com/2019/05/pytorch-internals/

https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc

https://medium.com/sfu-cspmp/diy-deep-learning-crafting-your-own-autograd-engine-from-scratch-for-effortless-backpropagation-ddab167faaf5

Matrix Differentiation ( and some other stuff ), Randal J. Barnes, Department of Civil Engineering, University of Minnesota Minneapolis, Minnesota, USA

https://numpy.org/devdocs/dev/internals.html

Set Up:

To build C library, use Makefile by running:
% make
and you can delete build files using:
% make clean

To get familiar with the library, go through tutorial.ipynb, which contains examples of basic operations and use cases (ex linear regression). You might need to install matplotlib if not already installed


To run tests, use:
% python3 -m unittest discover -s tests
