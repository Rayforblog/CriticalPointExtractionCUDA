# CriticalPointExtractionCUDA
Project for accurately and rapidly extracting the critical point programming

# Run-time environment
Visual Studio and CUDA8.0 are installed on a computer with the NVDIA graphics card which the compute capability is 2.1 or above (The Visual Studio 2013 is used here)

#test data
DEM_2980_3515.img

#detail procedure of the parallel project

1)Downloading all of the files except the ReadMe.txt from the website (https://github.com/parallelProject/CriticalPointExtractionCUDA)(Tip:the gdal.lib file in the "lib" folder of the "include" folder is larger than 50M, it must be downloaded in seperate and replace the gdal.lib file in the *.ZIP file.)
 
2)Setting the file directory of the variables "inName" and "outName" in the main.cpp and the pre-defined threshold value
  
3)Modifing the variables "VSInstallDir" "CUDAInstallDir" "CUDASampleDir" "DOWNLOADFILE_PATH" in the "makefile" file to the your path

4)Running the VS developer command prompt

5)Calling the 'nmake' to compile the program and generating the "criticalpointextractioncuda.exe" in the "Debug" folder of your path

6)Running the "criticalpointextractioncuda.exe" and prompting the loss of the gdal14.dll, TinDLL.dll and geos_c.dll, then copy the three files in the "bin" folder of the "include" folder to the “Debug” folder

7)Running again and finally obtaining the image of the critical point and a txt file recording the computing time

# Tip
If it reports an error (0xC000005: Read location access conflicts) during the running, right clicking the NVIDIA Nsight Monitor, clicking the option, setting the WDDM TDR Delay to enough large (e.g it is 20 here), rebooting the computer and it will ok.
