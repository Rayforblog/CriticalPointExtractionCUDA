// main.cpp : define the entry point for the console application
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include "stdafx.h"
#include "stdio.h" 
#include "ComBase.h"
#include "TinClass.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CWinApp theApp;
using namespace std;

extern "C" BOOL ztolerancePointSelect(float *psrc, int imgWidth, int imgHeight, double bboxMinX, double bboxMaxY, double dx, double dy, double ztolerValue, float *pdes, float *pIdx);

void GDALPointSelect(BSTR inName, double thresholdVal, BSTR outName)
{
	CString strOutName(outName);
	CComBase comBase;
	int imgWidth = 0, imgHeight = 0;

	cout << "It has entered the PointSelect and will open the DEM image!" << endl;
	if (!comBase.OpenImg(inName, imgWidth, imgHeight))
		return;
	float *pBufferIn = new float[imgWidth*imgHeight];
	double dx = 0, dy = 0, Xmin = 0, Ymax = 0;
	CString prjRef;
	if (!comBase.OpenImg(inName, imgWidth, imgHeight, dx, dy, Xmin, Ymax, prjRef, pBufferIn))
		return;

	//Create a new DEM image for recording the output critical point
	float *DEMPtSel = new float[imgWidth*imgHeight];
	memset(DEMPtSel, 0, imgWidth*imgHeight*sizeof(float));

	//Set the index value for recording the calculation sequence
	float *DEMPtIdx = NULL;
	DEMPtIdx = new float[imgWidth*imgHeight];
	memset(DEMPtIdx, 0, imgWidth*imgHeight*sizeof(float));

	cout << "It will enter the function for extracting the critical point!" << endl;

	//It adds the point with  the maximum  difference in elevation over each triangle during one iteration
	//The premise is the maximum difference in elevation over each triangle is larger than the threshold value
	ztolerancePointSelect(pBufferIn, imgWidth, imgHeight, Xmin, Ymax, dx, dy, thresholdVal, DEMPtSel, DEMPtIdx);

	cout << "It will create the new image!" << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << DEMPtSel[i] << endl;
	}
	BOOL b = comBase.CreateNewImg(outName, imgWidth, imgHeight, Xmin, Ymax, dx, dy, -9999, prjRef, DEMPtSel);
	if (!b)
	{
		cout << "CreateNewImg failed!" << endl;
	}
	if (DEMPtIdx != NULL)
	{
		cout << "DEMPtIdx != NULL!" << endl;
		BSTR name1 = L"_idx.img";
		CString OutName = strOutName + name1;
		comBase.CreateNewImg(OutName, imgWidth, imgHeight, Xmin, Ymax, dx, dy, -9999, prjRef, DEMPtIdx);
		delete[]DEMPtIdx; DEMPtIdx = NULL;
	}

	delete[]pBufferIn; pBufferIn = NULL;
	delete[]DEMPtIdx; DEMPtIdx = NULL;
}

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	GDALAllRegister();
	clock_t begin, end;
	double time;
	cout << "It starts extracting the critical point!" << endl;
	BSTR inName = L"D:\\CriticalPointExtractionCUDA-master\\DEM_2980_3515.img";			//It is the directory of the input DEM image
	BSTR outName = L"D:\\CriticalPointExtractionCUDA-master\\DEM_2980_3515_CP_18.0.img";	//It is the directory of the output cirtical point
	ofstream timeTxt;
	timeTxt.open("D:\\CriticalPointExtractionCUDA-master\\DEM_2980_3515_CP_18.0.txt");		//The text is recording the computation time
	begin = clock();
	GDALPointSelect(inName, 18.0, outName);			//18.0 is the threshold value
	end = clock();
	time = (double)(end - begin) / CLOCKS_PER_SEC;
	timeTxt << "time:" << time << endl;
	timeTxt << flush;
	timeTxt.close();
	system("pause");
	return 0;
}
