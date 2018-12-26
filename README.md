[![BSD (2 clause) License](https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg)](https://github.com/Maratyszcza/NNPACK/blob/master/LICENSE)
# NNPACK for Windows (AVX2 backend)
#### port of Marat Dukhan NNPACK (https://github.com/Maratyszcza/NNPACK)

Before building this repo you have to install PeachPy.

Open a Python command prompt with Administrator rights and type:
  ```bash
pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
```
  
Now you're ready for building with Visual Studio 2017. I recommend to use the provided Visual Sudio 2017 solution when targeting Windows. NNPACK depends on cpuinfo wich is included in this repo but NOT in the provided solution. You must uses CMake to generate the
Visual Studio project files and add it to your solution and link it with NNPACK. Don't use CMake for NNPACK itseft under Windows because it will not without some changes provide you with a functional library. This is because the library mix AVX2 code ans psimd under windows and not on other operating systems.


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

The only c-interface difference beween this port and NNPACK is the omission of the pthreadpool parameter. I use a stardard c++ threadpool implementation in non-Windows environments and the Microsoft Concurrency Runtime under Windows instead.
The AVX2 FT16x16 kernels are sadly currently not usable under Windows and are bypassed with the psimd FT16x16 kernels. This is because the PeachPy generated AVX2 FT16x16 kernels didn't pass the unit tests. This only affects the Windows build. Under Linux/Mac OS/Android/iOS all kernels are passing the unit tests without having to bypass some kernels. The psimd and scalar backend are also fully Windows compatible. You always can change the default AVX2 backend and exclude the files from the x86_64-fma folder from building and include the files from for example the psimd folder if you want a psimd build instead.
Here are the steps if you want a non-Windows build like linux for example:
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
