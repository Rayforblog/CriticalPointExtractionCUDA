// main.cpp : 定义控制台应用程序的入口点
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

	cout << "已经进入PointSelect,即将打开DEM影像！" << endl;

	if (!comBase.OpenImg(inName, imgWidth, imgHeight))
		return;
	cout << "第一次打开DEM完成！" << endl;

	float *pBufferIn = new float[imgWidth*imgHeight];
	double dx = 0, dy = 0, Xmin = 0, Ymax = 0;
	CString prjRef;
	if (!comBase.OpenImg(inName, imgWidth, imgHeight, dx, dy, Xmin, Ymax, prjRef, pBufferIn))
		return;

	cout << "第二次打开DEM完成！" << endl;

	//创建一个新的输出特征点的DEM影像
	float *DEMPtSel = new float[imgWidth*imgHeight];
	memset(DEMPtSel, 0, imgWidth*imgHeight*sizeof(float));

	//设置计算排列顺序的index数值信息
	float *DEMPtIdx = NULL;
	DEMPtIdx = new float[imgWidth*imgHeight];
	memset(DEMPtIdx, 0, imgWidth*imgHeight*sizeof(float));

	cout << "即将进入DEM特征点的提取函数！" << endl;

	//一次增加每个三角形中具有最大高程差值的点（前提是每个三角形中的最大高程差值大于阈值，则添加到地形特征点集合中）
	ztolerancePointSelect(pBufferIn, imgWidth, imgHeight, Xmin, Ymax, dx, dy, thresholdVal, DEMPtSel, DEMPtIdx);

	cout << "即将createNewImg!" << endl;
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
	cout << "开始提取特征点：" << endl;
	BSTR inName = L"D:\\workstation\\DEM_2980_3515.img";
	BSTR outName = L"D:\\workstation\\DEM_2980_3515_cp.img";
	ofstream timeTxt;
	timeTxt.open("D:\\DEM_2980_3515_48.txt");
	begin = clock();
	GDALPointSelect(inName, 48, outName);
	end = clock();
	time = (double)(end - begin) / CLOCKS_PER_SEC;
	timeTxt << "DEM_5958_3514_55_time:" << time << endl;
	timeTxt << flush;
	timeTxt.close();
	system("pause");
	return 0;
}
