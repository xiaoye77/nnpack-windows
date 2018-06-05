[![BSD (2 clause) License](https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg)](https://github.com/Maratyszcza/NNPACK/blob/master/LICENSE)
# NNPACK for Windows (AVX2 backend)
#### port of Marat Dukhan NNPACK (https://github.com/Maratyszcza/NNPACK)

Before building this repo you have to install PeachPy.

Open a Python command prompt with Administrator rights and type:
  ```bash
pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
```
  
Now you're ready for building with Visual Studio 2017. I recommend to use the provided Visual Sudio 2017 solution when targeting Windows.


Results of the unit tests:

### convolution-inference:

  * implicit gemm:  passed
  
  * direct conv:    passed
  
  * FT8x8:          passed
  
  * FT16x16:        passed (kernel failed for AVX2, using psimd implementation instead)

  * WT8x8:          passed
  
### convolution-output:

  * FT8x8:    passed

  * FT16x16:  passed (kernel failed for AVX2, using psimd implementation instead)

  * WT8x8:    passed


### convolution-input-gradient:

  * FT8x8:    passed

  * FT16x16:  passed (kernel failed for AVX2, using psimd implementation instead)

  * WT8x8:    passed


### convolution-kernel-gradient:

  * FT8x8:    passed

  * FT16x16:  passed (kernel failed for AVX2, using psimd implementation instead)

  * WT8x8:    disabled

 
### fourier:
### fully-connected-inference:
### fully-connected:
### max-pooling-output:
### relu-input-gradient:
### relu-output
### sgemm:
### softmax-output:
### winograd:

  * all passed

The only c-interface difference with NNPACK is the omission of the pthreadpool parameter and using the Microsoft's concurrency primtives under Windows and a simple generic c++ threadpool implementation otherwise.
The AVX2 FT16x16 kernels are currently non-functional under Windows and are bypassed with the psimd FT16x16 kernels. This is because the PeachPy generated AVX2 FT16x16 kernels didn't pass the unit tests.
This only affects the Windows build. Under Linux/Mac OS/Android/iOS all kernels are passing the unit tests without having to bypass some kernels. The psimd and scalar backend are also fully Windows compatible. 
You always can change the default AVX2 backend and exclude the files from the x86_64-fma folder from building and include the files from for example the psimd folder if you want a psimd build instead.
Here are the steps if you want a linux build for example:
```bash
sudo apt-get install ninja-build
sudo pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
sudo pip install --upgrade git+https://github.com/Maratyszcza/confu
git clone https://github.com/zeno40/nnpack-windows.git
cd nnpack-windows
confu setup
python ./configure.py --toolchain=clang
ninja
ninja smoketest
```

### Dependencies & Licenses
* Agner Fog C++ Vector Class Library - GNU License (http://www.agner.org/optimize)
