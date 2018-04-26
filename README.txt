# CriticalPointExtraction
Project for extracting the critical point programming

# Run-time environment
Visual Studio and CUDA8.0 are installed on a computer with the NVDIA graphics card which the compute capability is 2.1 or above

#test data
DEM_2980_3515.img

#detail procedure of the parallel project
1)Download all of the files except the ReadMe.txt from the website 
  (https://github.com/parallelProject/CriticalPointExtraction)
  (Tip: the gdal.lib in the "lib" folder of the "include" folder is larger than 50M, it must be downloaded in seperate.)
2)Copy these files to the file directory of the project
4)Add the four files(ComBase.cpp, ComBase.h, kernel.cu and main.cpp) into the project
5)Configure the path of the library file and properties, Click the Project->Properties
  ->Configuration properties and do this:
   a)Click the “General”, set “Use of MFC” as “Use MFC in a Shared DLL”,set “Character Set” 
     as “Use Unicode Character Set”
   b)Click the “VC++Directories”，add the file directory of the “include” and “inc” in the installation 
     directory and CUDA samples of CUDA8.0 (like ...\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include,
     ...\NVIDIA Corporation\CUDASamples\v8.0\common\inc), the file directory of the “TinDLL” and 
     “include” in the “include” folder(like...\include\TinDLL,...\include\include),and the file directory
     of the “include” folder (like...\include) into the “Include Directories”
   c)Add the file directory of the include in the installation directory of CUDA8.0 (like...\NVIDIA GPU 
     Computing Toolkit\CUDA\v8.0\include) and the file directory of the “lib”  in the “include” folder
     (like ...\include\lib) into the “Library Directories”
   d)Click the “C/C++”->Code Generation, set Runtime Library as Multi-threaded DLL(/MD) 
   e)Click the “Linker”->Input,add gdal_i.lib, TinDLL.lib and gdal.lib into "Additional Dependencies"
6)Compile the program and prompt success
7)Download the test data from the above website, set the file directory of inName and outNamein the 
  main.cpp and the pre-defined thresholdVal
8)Run the program, maybe prompt the loss of some DLL(Copy the gdal14.dll, geos.dll, geos_c.dll and 
  TinDLL.dll in the “bin” folder to the “Debug” folder (like ...\Debug)
9)Run again and finally get the critical point and a txt file recorded the computing time. 

# Tip
If it reports an error (0xC000005: Read location access conflicts) and prompts the “the monitor driver 
has stopped responding and has been restored” during the running, right click the NVIDIA Nsight Monitor,
click the option and set the WDDM TDR Delay as more than 20,reboot the computer and it will ok
