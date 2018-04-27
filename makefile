#This is a makefile for criticalpointcuda project

#Specifying the compiler
CC=cl
CXX=nvcc

#The installation directory of the VS and CUDA and the directory of the download files from the github
VSInstallDir=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC
CUDAInstallDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
CUDASampleDir=C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
DOWNLOADFILE_PATH=D:\CriticalPointExtraction-master

#The environmental configuration
CC_PROJ=/GS /analyze- /W3 /Zc:wchar_t /I"$(CUDAInstallDir)\include" /ZI /Gm /Od /fp:precise /D "WIN32" \
/D "_DEBUG" /D "_CONSOLE" /D "_UNICODE" /D "UNICODE" /D "_AFXDLL" /errorReport:prompt /WX- /Zc:forScope /RTC1 /Gd /Oy- /MD
CXX_PROJ=-gencode=arch=compute_20,code=\"sm_20,compute_20\" -G --keep-dir Debug -maxrregcount=0 --machine 32 \
--compile -cudart static  -g -DWIN32 -D_DEBUG -D_CONSOLE -D_UNICODE -D_UNICODE -D_AFXDLL -Xcompiler "/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MD"

#The directory of the include and library files
INCLUDES_PATH=/I"$(DOWNLOADFILE_PATH)\include\include" /I"$(DOWNLOADFILE_PATH)\include\TinDLL" /I"$(VSInstallDir)\include" /I"$(VSInstallDir)\atlmfc\include"
KERNELINCLUDE_PATH=-I"$(DOWNLOADFILE_PATH)\include\include" -I"$(DOWNLOADFILE_PATH)\\include\TinDLL" -I"$(CUDAInstallDir)\include" -I"$(CUDASampleDir)\common\inc"
LIBRARYS_PATH=/LIBPATH:"$(DOWNLOADFILE_PATH)\include\lib" 
CUDALIBRARY_PATH=/LIBPATH:"$(CUDAInstallDir)\lib\win32"

#Put the consoleTest.exe into the debug folder
OUTDIR=.\Debug

#Tell the compiler that the project is trying to generate an exe
all:$(OUTDIR) $(OUTDIR)\criticalpointcuda.exe

#If the Debug folder is not exist,create it
$(OUTDIR):
	if not exist "$(OUTDIR)" mkdir $(OUTDIR)

#calling 'nmake' will ensure that every feature of the program is compiled
#compile
$(OUTDIR)\ComBase.obj:ComBase.cpp
	$(CC) $(INCLUDES_PATH) $(CC_PROJ) /c /Istdafx.h /Iresource.h /IComBase.h /Fo"$(OUTDIR)\\" ComBase.cpp

$(OUTDIR)\kernel.cu.obj:kernel.cu
	$(CXX) $(CXX_PROJ) -ccbin "$(VSInstallDir)\bin" $(KERNELINCLUDE_PATH) -o $(OUTDIR)\kernel.cu.obj kernel.cu

$(OUTDIR)\main.obj:main.cpp
	$(CC) $(INCLUDES_PATH) $(CC_PROJ) /c /Istdafx.h /IComBase.h /ITinClass.h /Fo"$(OUTDIR)\\" main.cpp

#link
$(OUTDIR)\criticalpointcuda.exe:$(OUTDIR)\ComBase.obj $(OUTDIR)\kernel.cu.obj $(OUTDIR)\main.obj
	link $(LIBRARYS_PATH) $(CUDALIBRARY_PATH) /out:$(OUTDIR)\criticalpointcuda.exe $(OUTDIR)\ComBase.obj $(OUTDIR)\kernel.cu.obj $(OUTDIR)\main.obj cudart.lib kernel32.lib\
	user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib gdal_i.lib TinDLL.lib gdal.lib

#clean
#Calling 'nmake clean' to delete the obj and pdb file
clean:
	if exist $(OUTDIR) del $(OUTDIR)\*.obj
	if exist $(OUTDIR) del $(OUTDIR)\*.exe
	if exist $(OUTDIR) del $(OUTDIR)\*.lib
	if exist $(OUTDIR) del $(OUTDIR)\*.exp
	if exist .\ del *.idb
	if exist .\ del *.pdb