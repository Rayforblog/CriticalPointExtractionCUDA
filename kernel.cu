#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>
#include "stdafx.h"
#include "stdio.h" 
#include "ComBase.h"
#include "TinClass.h"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include <helper_cuda.h>
using namespace std;

#define TC 16

//x, y, z数组（成员个数：3）:记录三角形三个顶点行列好、高程值
//border:记录矩形区域左上角和右下角在原始图像中的行列号
//psrc数组（成员个数imgWidth * imgHeight）:记录原始图像像素值，行主序
//imgSize数组（成员个数2）:记录原始图像行、列数
//bbox数组（成员个数2）:记录图像左上角横纵坐标。
//pd数组（成员个数2）:记录象元大小，横、纵表示的长度
//elv:所有像素点计算的高程差值，不在范围内的点计算结果为0
__global__ void kernelPixel(int *x, int *y, double *z, int *border, float *psrc, int *imgSize, float *elv)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = gridDim.x * blockDim.x * tidy + tidx;

	int borderWidth = border[2] - border[0] + 1;
	int borderHeight = border[3] - border[1] + 1;
	
	//超出矩形区域范围，终止当前线程
	if (tid >= borderWidth * borderHeight)
		return;

	//计算矩形区域中的行列号位置
	int br = tid / borderWidth;
	int bc = tid % borderWidth;

	//根据矩形区域中的位置和矩形左上角的位置，推算在原图像中的位置
	int oc = bc + border[0];
	int or = br + border[1];

	//推算点在psrc一维数组中的位置
	int index = imgSize[0] * or + oc;

	//三角形三点坐标，待计算像素点坐标
	int x3 = oc;
	int y3 = or;
	double z3 = psrc[index];

	//面积法计算点是否在三角形范围内,不在范围内则退出当前线程
	//计算原始三角形面积
	double ds0 = sqrt((double)((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1])));
	double ds1 = sqrt((double)((x[0] - x[2]) * (x[0] - x[2]) + (y[0] - y[2]) * (y[0] - y[2])));
	double ds2 = sqrt((double)((x[1] - x[2]) * (x[1] - x[2]) + (y[1] - y[2]) * (y[1] - y[2])));
	double t0 = (double)((ds0 + ds1 + ds2) * (0.5));
	double s0 = sqrt((double)(t0 * (t0 - ds0) * (t0 - ds1) * (t0 - ds2)));
	
	//计算待计算点与原始三角形三点构成的三个三角形的面积
	double ds3 = sqrt((double)((x3 - x[0]) * (x3 - x[0]) + (y3 - y[0]) * (y3 - y[0])));
	double ds4 = sqrt((double)((x3 - x[1]) * (x3 - x[1]) + (y3 - y[1]) * (y3 - y[1])));
	double ds5 = sqrt((double)((x3 - x[2]) * (x3 - x[2]) + (y3 - y[2]) * (y3 - y[2])));
	if (ds3 < 0.0001 || ds4 < 0.0001 || ds5 < 0.0001) return;
	double t1 = (ds0 + ds3 + ds4) * (0.5);
	double s1 = sqrt((double)(t1 * (t1 - ds0) * (t1 - ds3) * (t1 - ds4)));
	double t2 = (ds1 + ds3 + ds5) * (0.5);
	double s2 = sqrt((double)(t2 * (t2 - ds1) * (t2 - ds3) * (t2 - ds5)));
	double t3 = (ds2 + ds4 + ds5) * (0.5);
	double s3 = sqrt((double)(t3 * (t3 - ds2) * (t3 - ds4) * (t3 - ds5)));
	double ts = fabs(s0 - s1 - s2 - s3);
	if (ts > 0.0001)
		return;

	//计算三角形所在面方程并计算高程差
	double t = (x[0] - x[1]) * (y[0] - y[2]) - (x[0] - x[2]) * (y[0] - y[1]);
	if (t == 0)
		return;
	double a = ((y[0] - y[2]) * (z[0] - z[1]) - (y[0] - y[1]) * (z[0] - z[2])) * ((1.0f) / t);
	double b = -((x[0] - x[2]) * (z[0] - z[1]) - (x[0] - x[1]) * (z[0] - z[2])) * ((1.0f) / t);
	double c = z[0] - a * x[0] - b * y[0];
	double offset = fabsf(z3 - a * x3 - b * y3 - c);

	elv[tid] = offset;
}

//psrc:一位数组形式的图像像素值，大小imgWidth * imgHeight
//imgSize:图像行列象元个数，大小2
//upLeftCord:图像左上角地理坐标（x, y)，大小2
//pixelSize:	象元分辨率（ dx, dy)，大小2
//toler:容差，大小1
//x0, y0, z0, x1, y1, z1, x2, y2, z2:三角形三个脚点坐标
//triCount:三角形个数
//row:记录查找到的三角形中大于容差的具有最大高程值的点行号
//col:记录查找到的三角形中大于容差的具有最大高程值的点列号
__global__ void kernelTri(float* psrc, int* imgSize, double* upLeftCord, double* pixelSize, double* toler, double* x0, double* x1, double* x2, double* y0, double* y1, double* y2, float* z0, float* z1, float* z2, int* triCount, int* row, int* col)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = gridDim.x * blockDim.x * tidy + tidx;
	if (tid >= triCount[0])	return;

	int colt[3] = {
		(int)((x0[tid] - upLeftCord[0]) / pixelSize[0] + 0.5),
		(int)((x1[tid] - upLeftCord[0]) / pixelSize[0] + 0.5),
		(int)((x2[tid] - upLeftCord[0]) / pixelSize[0] + 0.5)
	};

	int rowt[3] = {
		(int)((upLeftCord[1] - y0[tid]) / pixelSize[1] + 0.5),
		(int)((upLeftCord[1] - y1[tid]) / pixelSize[1] + 0.5),
		(int)((upLeftCord[1] - y2[tid]) / pixelSize[1] + 0.5)
	};

	float alt[3] = { z0[tid], z1[tid], z2[tid] };

	int xMin, xMax, yMin, yMax;
	xMin = colt[0] < colt[1] ? (colt[0] < colt[2] ? colt[0] : colt[2]) : (colt[1] < colt[2] ? colt[1] : colt[2]);
	xMax = colt[0] > colt[1] ? (colt[0] > colt[2] ? colt[0] : colt[2]) : (colt[1] > colt[2] ? colt[1] : colt[2]);
	yMin = rowt[0] < rowt[1] ? (rowt[1] < rowt[2] ? rowt[0] : rowt[2]) : (rowt[1] < rowt[2] ? rowt[1] : rowt[2]);
	yMax = rowt[0] > rowt[1] ? (rowt[0] > rowt[2] ? rowt[0] : rowt[2]) : (rowt[1] > rowt[2] ? rowt[1] : rowt[2]);

	if (xMin == xMax || yMin == yMax) return;

	//三角面公式
	double t = (colt[0] - colt[1])*(rowt[0] - rowt[2]) - (colt[0] - colt[2])*(rowt[0] - rowt[1]);
	if (t == 0) return;
	double a = ((rowt[0] - rowt[2]) * (alt[0] - alt[1]) - (rowt[0] - rowt[1]) * (alt[0] - alt[2]))*(1.0f / t);
	double b = -((colt[0] - colt[2]) * (alt[0] - alt[1]) - (colt[0] - colt[1]) * (alt[0] - alt[2]))*(1.0f / t);
	double c = alt[0] - a * colt[0] - b * rowt[0];

	//计算当前三角形面积
	double ds01 = sqrt((double)(colt[0] - colt[1])*(colt[0] - colt[1]) + (rowt[0] - rowt[1])*(rowt[0] - rowt[1]));
	double ds02 = sqrt((double)(colt[0] - colt[2])*(colt[0] - colt[2]) + (rowt[0] - rowt[2])*(rowt[0] - rowt[2]));
	double ds12 = sqrt((double)(colt[1] - colt[2])*(colt[1] - colt[2]) + (rowt[1] - rowt[2])*(rowt[1] - rowt[2]));
	double pm012 = (ds01 + ds02 + ds12) * 0.5;
	double s012 = sqrt(pm012 * (pm012 - ds01) * (pm012 - ds02) * (pm012 - ds12));

	//开始计算最大高程差及其行列号
	double maxVal = 0;
	int maxX = 0, maxY = 0;
	for (int i = yMin; i < yMax; i++)
	for (int j = xMin; j < xMax; j++)
	{
		float pVal = psrc[i * imgSize[0] + j];
		if (pVal <= 0) continue;

		//海伦公式法确定待计算点是否在三角形范围内
		double dsi0 = sqrt((double)(j - colt[0]) * (j - colt[0]) + (i - rowt[0])*(i - rowt[0]));
		double dsi1 = sqrt((double)(j - colt[1]) * (j - colt[1]) + (i - rowt[1])*(i - rowt[1]));
		double dsi2 = sqrt((double)(j - colt[2]) * (j - colt[2]) + (i - rowt[2])*(i - rowt[2]));
		if (dsi0 < 0.0001 || dsi1 < 0.0001 || dsi2 < 0.0001) continue;
		double pmi01 = (dsi0 + dsi1 + ds01) * 0.5;
		double pmi02 = (dsi0 + dsi2 + ds02) * 0.5;
		double pmi12 = (dsi1 + dsi2 + ds12) * 0.5;
		double si01 = sqrt(pmi01 * (pmi01 - dsi0) * (pmi01 - dsi1) * (pmi01 - ds01));
		double si02 = sqrt(pmi02 * (pmi02 - dsi0) * (pmi02 - dsi2) * (pmi02 - ds02));
		double si12 = sqrt(pmi12 * (pmi12 - dsi1) * (pmi12 - dsi2) * (pmi12 - ds12));
		double s = fabs(s012 - si01 - si02 - si12);
		if (s > 0.0001)
			continue;

		//计算高程差
		double dVal = fabs(pVal - (a * j + b * i + c));
		if (dVal > toler[0] && dVal > maxVal)
		{
			maxVal = dVal;
			maxX = j;
			maxY = i;
		}
	}
	row[tid] = maxY;
	col[tid] = maxX;
}

//向TIN中增加一个点(用于在TIN的构建时，增加TIN的4个角点)
extern "C" BOOL TinAddPoint1By1(float *psrc, int imgWidth, double bboxMinX, double bboxMaxY, double dx, double dy, float *pdes, long& pntCount, CTINClass *pTinDll, int I, int J)
{
	point3D pAddPnt;
	int k = I * imgWidth + J;
	if (psrc[k] < 0)
	{
		cout << "psrc[k]小于0，返回FALSE" << endl;
	}

	pAddPnt.x = bboxMinX + J * dx;
	pAddPnt.y = bboxMaxY - I * dy;
	pAddPnt.z = psrc[k];
	pdes[k] = psrc[k];

	pTinDll->AddPoint(pAddPnt.x, pAddPnt.y, pAddPnt.z, 0, NULL);
	pntCount++;

	return TRUE;
}

//一次增加每个三角形中具有最大高程差的像元点
extern "C" BOOL ztolerancePointSelect(float *psrc, int imgWidth, int imgHeight, double bboxMinX, double bboxMaxY, double dx, double dy, double ztolerValue, float *pdes, float *pIdx)
{
	cout << "进入ztolerancePointSelect1By1." << endl;
	point3D* pPnt = new point3D[imgWidth*imgHeight];     //记录用于构建TIN的点信息
	long pntCount = 0;									//记录pnt的个数

	CTINClass *pTinDll = new CTINClass("pczAenVQ");
	pTinDll->BeginAddPoints();

	//构建第一个TIN，取4个角点坐标，若所取得的角点个数小于4，则退出
	TinAddPoint1By1(psrc, imgWidth, bboxMinX, bboxMaxY, dx, dy, pdes, pntCount, pTinDll, 0, 0);
	TinAddPoint1By1(psrc, imgWidth, bboxMinX, bboxMaxY, dx, dy, pdes, pntCount, pTinDll, 0, imgWidth - 1);
	TinAddPoint1By1(psrc, imgWidth, bboxMinX, bboxMaxY, dx, dy, pdes, pntCount, pTinDll, imgHeight - 1, 0);
	TinAddPoint1By1(psrc, imgWidth, bboxMinX, bboxMaxY, dx, dy, pdes, pntCount, pTinDll, imgHeight - 1, imgWidth - 1);

	pPnt[0].x = bboxMinX;
	pPnt[0].y = bboxMaxY;
	pPnt[0].z = psrc[0];

	pPnt[1].x = bboxMinX + (imgWidth - 1) * dx;
	pPnt[1].y = bboxMaxY;
	pPnt[1].z = psrc[imgWidth - 1];

	pPnt[2].x = bboxMinX;
	pPnt[2].y = bboxMaxY - (imgHeight - 1) * dy;
	pPnt[2].z = psrc[imgWidth * (imgHeight - 1)];

	pPnt[3].x = bboxMinX + (imgWidth - 1) * dx;
	pPnt[3].y = bboxMaxY - (imgHeight - 1) * dy;
	pPnt[3].z = psrc[imgWidth * imgHeight - 1];

	if (pntCount < 4)
	{
		cout << "pntCount为" << pntCount << "小于4" << endl;
		return FALSE;
	}

	//此处可根据影像的四个角点选取高程>0的角点值。初次构建TIN时，增加的点数要大于3
	pTinDll->EndAddPoints();
	pTinDll->FastConstruct();

	//数据拷贝进device
	float* psrc_d;
	int *imgSize_d, imgSize_h[2] = { imgWidth, imgHeight };
	double *upLeftCordinate_d, upLeftCordinate_h[2] = { bboxMinX, bboxMaxY };
	double *pixelSize_d, pixelSize_h[2] = { dx, dy };
	double *toler_d, toler_h[1] = { ztolerValue };
	cudaMalloc((void**)&psrc_d, imgWidth * imgHeight * sizeof(float));
	cudaMalloc((void**)&imgSize_d, 2 * sizeof(int));
	cudaMalloc((void**)&upLeftCordinate_d, 2 * sizeof(double));
	cudaMalloc((void**)&pixelSize_d, 2 * sizeof(double));
	cudaMalloc((void**)&toler_d, 1 * sizeof(double));
	cudaMemcpy(psrc_d, psrc, imgWidth * imgHeight * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(imgSize_d, imgSize_h, 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(upLeftCordinate_d, upLeftCordinate_h, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pixelSize_d, pixelSize_h, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(toler_d, toler_h, 1 * sizeof(double), cudaMemcpyHostToDevice);

	int x_h[3], y_h[3], border_h[4];		//三角形三点的列、行号；外接矩形区域最小最大行列号
	double z_h[3];							//三角形三点的高程值；

	int *x_d, *y_d, *border_d;
	double *z_d;
	cudaMalloc((void**)&x_d, 3 * sizeof(int));
	cudaMalloc((void**)&y_d, 3 * sizeof(int));
	cudaMalloc((void**)&z_d, 3 * sizeof(double));
	cudaMalloc((void**)&border_d, 4 * sizeof(int));
	cudaError_t r0 = cudaGetLastError();

	BOOL flag = true;
	int indicator = 0;
	while (flag)
	{
		flag = false;
		if (indicator < 9 )
		{
#pragma region 遍历外接矩形中的像素点
			pTinDll->TriangleTraversalInit();
			TRIANGLE *tri = 0;

			double x[3], y[3], z[3];					//记录三角形的三个顶点，其中x,y坐标为行、列号，z坐标为高程值
			long pntIsAddedNum = pntCount;						//记录初始的pntCount数目，用于后面判断构建TIN的点数目是否增加
			while (tri = pTinDll->TriangleTraverse())
			{
				x[0] = tri->vertex[0]->x;
				y[0] = tri->vertex[0]->y;
				z[0] = tri->vertex[0]->attr;
				x_h[0] = (int)((x[0] - upLeftCordinate_h[0]) / pixelSize_h[0] + 0.5);
				y_h[0] = (int)((upLeftCordinate_h[1] - y[0]) / pixelSize_h[1] + 0.5);
				z_h[0] = z[0];

				x[1] = tri->vertex[1]->x;
				y[1] = tri->vertex[1]->y;
				z[1] = tri->vertex[1]->attr;
				x_h[1] = (int)((x[1] - upLeftCordinate_h[0]) / pixelSize_h[0] + 0.5);
				y_h[1] = (int)((upLeftCordinate_h[1] - y[1]) / pixelSize_h[1] + 0.5);
				z_h[1] = z[1];

				x[2] = tri->vertex[2]->x;
				y[2] = tri->vertex[2]->y;
				z[2] = tri->vertex[2]->attr;
				x_h[2] = (int)((x[2] - upLeftCordinate_h[0]) / pixelSize_h[0] + 0.5);
				y_h[2] = (int)((upLeftCordinate_h[1] - y[2]) / pixelSize_h[1] + 0.5);
				z_h[2] = z[2];

				int xMax = x_h[0] > x_h[1] ? (x_h[2] > x_h[0] ? x_h[2] : x_h[0]) : (x_h[2] > x_h[1] ? x_h[2] : x_h[1]);
				int xMin = x_h[0] < x_h[1] ? (x_h[2] < x_h[0] ? x_h[2] : x_h[0]) : (x_h[2] < x_h[1] ? x_h[2] : x_h[1]);
				int yMax = y_h[0] > y_h[1] ? (y_h[2] > y_h[0] ? y_h[2] : y_h[0]) : (y_h[2] > y_h[1] ? y_h[2] : y_h[1]);
				int yMin = y_h[0] < y_h[1] ? (y_h[2] < y_h[0] ? y_h[2] : y_h[0]) : (y_h[2] < y_h[1] ? y_h[2] : y_h[1]);

				if (xMin == xMax && yMin == yMax)
				{
					return FALSE;
				}

				border_h[0] = xMin;
				border_h[1] = yMin;
				border_h[2] = xMax;
				border_h[3] = yMax;

				//计算矩形范围大小
				int rectWidth = xMax - xMin + 1;
				int rectHeight = yMax - yMin + 1;
				int rectSize = rectWidth * rectHeight;
				float *elv_h, *elv_d;
				elv_h = (float*)malloc(rectSize * sizeof(float));
				memset(elv_h, 0, rectSize * sizeof(float));
				cudaMalloc((void**)&elv_d, rectSize * sizeof(float));
				cudaMemset(elv_d, 0, rectSize * sizeof(float));
				cudaMemcpy(x_d, x_h, 3 * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(y_d, y_h, 3 * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(z_d, z_h, 3 * sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(border_d, border_h, 4 * sizeof(int), cudaMemcpyHostToDevice);

				dim3 threads(TC, TC);
				dim3 blocks((rectWidth + TC - 1) / TC, (rectHeight + TC - 1) / TC);
				cudaError_t r0 = cudaGetLastError();

				kernelPixel<<<blocks, threads>>>(x_d, y_d, z_d, border_d, psrc_d, imgSize_d, elv_d);

				cudaError_t r1 = cudaGetLastError();
				cudaError_t r2 = cudaDeviceSynchronize();

				cudaMemcpy(elv_h, elv_d, rectSize * sizeof(float), cudaMemcpyDeviceToHost);

				if (r0 != 0 || r1 != 0 || r2 != 0)
					int abcd = 0;

				int maxIndex = 0;
				double maxElv = 0.0;
				for (int i = 0; i < rectSize; i++)
				{
					if (elv_h[i] > maxElv)
					{
						maxElv = elv_h[i];
						maxIndex = i;
					}
				}

				if (maxElv >= ztolerValue && pntCount < imgHeight * imgWidth)
				{
					int x = maxIndex % rectWidth + border_h[0];
					int y = maxIndex / rectWidth + border_h[1];
					float z = psrc[(int)(imgWidth * y + x)];
					if (z <= 0.0001) continue;
					pPnt[pntCount].x = bboxMinX + x * dx;
					pPnt[pntCount].y = bboxMaxY - y * dy;
					pPnt[pntCount].z = z;

					if (pdes[y * imgWidth + x] > 0.0001) continue;
					pdes[y * imgWidth + x] = z;
					pIdx[y * imgWidth + x] = (float)pntCount;
					pntCount++;
					flag = true;
				}

				free(elv_h);
				cudaFree(elv_d);
			}

			if (flag == false || pntCount >= imgHeight * imgWidth)
				break;
			if (pntCount == pntIsAddedNum)
				break;   //如果pntCount数目没有发生变化说明没有新的点加入，则退出

			//构建新的TIN，以便进行下一次循环计算
			delete pTinDll;
			pTinDll = new CTINClass("pczAenVQ");
			pTinDll->BeginAddPoints();

			for (int i = 0; i < pntCount; i++)
			{
				pTinDll->AddPoint(pPnt[i].x, pPnt[i].y, pPnt[i].z, 0, NULL);
			}
			pTinDll->EndAddPoints();
			pTinDll->FastConstruct();
#pragma endregion
		}
		else
		{
#pragma region 遍历所有三角形
			long pntIsAddedNum = pntCount;   //记录初始的pntCount数目，用于后面判断构建TIN的点数目是否增加
			pTinDll->TriangleTraversalInit();
			vector<TRIANGLE> tris;
			TRIANGLE* triangle = 0;
			while (triangle = pTinDll->TriangleTraverse())
			{
				tris.push_back(*triangle);
			}
			int triCount = tris.size();
			double *x0_d, *x0_h = (double*)malloc(triCount * sizeof(double));
			double *x1_d, *x1_h = (double*)malloc(triCount * sizeof(double));
			double *x2_d, *x2_h = (double*)malloc(triCount * sizeof(double));
			double *y0_d, *y0_h = (double*)malloc(triCount * sizeof(double));
			double *y1_d, *y1_h = (double*)malloc(triCount * sizeof(double));
			double *y2_d, *y2_h = (double*)malloc(triCount * sizeof(double));
			float *z0_d, *z0_h = (float*)malloc(triCount * sizeof(float));
			float *z1_d, *z1_h = (float*)malloc(triCount * sizeof(float));
			float *z2_d, *z2_h = (float*)malloc(triCount * sizeof(float));
			int *triCount_d, triCount_h[1] = { triCount };

			//返回三角形中具有最大高程差的点
			int *row_d, *row_h = (int*)malloc(triCount * sizeof(int));
			int *col_d, *col_h = (int*)malloc(triCount * sizeof(int));

			cudaError_t r1 = cudaGetLastError();
			for (int i = 0; i < triCount; i++)
			{
				triangle = &tris[i];
				x0_h[i] = triangle->vertex[0]->x;
				y0_h[i] = triangle->vertex[0]->y;
				z0_h[i] = triangle->vertex[0]->attr;
				x1_h[i] = triangle->vertex[1]->x;
				y1_h[i] = triangle->vertex[1]->y;
				z1_h[i] = triangle->vertex[1]->attr;
				x2_h[i] = triangle->vertex[2]->x;
				y2_h[i] = triangle->vertex[2]->y;
				z2_h[i] = triangle->vertex[2]->attr;
			}
			cudaMalloc((void**)&x0_d, triCount * sizeof(double));
			cudaMalloc((void**)&x1_d, triCount * sizeof(double));
			cudaMalloc((void**)&x2_d, triCount * sizeof(double));
			cudaMalloc((void**)&y0_d, triCount * sizeof(double));
			cudaMalloc((void**)&y1_d, triCount * sizeof(double));
			cudaMalloc((void**)&y2_d, triCount * sizeof(double));
			cudaMalloc((void**)&z0_d, triCount * sizeof(float));
			cudaMalloc((void**)&z1_d, triCount * sizeof(float));
			cudaMalloc((void**)&z2_d, triCount * sizeof(float));
			cudaMalloc((void**)&triCount_d, 1 * sizeof(int));
			cudaMalloc((void**)&row_d, triCount * sizeof(int));
			cudaMalloc((void**)&col_d, triCount * sizeof(int));
			cudaMemcpy(x0_d, x0_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(x1_d, x1_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(x2_d, x2_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(y0_d, y0_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(y1_d, y1_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(y2_d, y2_h, triCount * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(z0_d, z0_h, triCount * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(z1_d, z1_h, triCount * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(z2_d, z2_h, triCount * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(triCount_d, triCount_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(row_d, 0, triCount * sizeof(int));
			cudaMemset(col_d, 0, triCount * sizeof(int));
			cudaError_t r2 = cudaGetLastError();

			dim3 threads(TC, TC);
			dim3 blocks(TC, triCount / (TC * TC * TC) + 1);

			//call kernel
			kernelTri<<<blocks, threads>>>(psrc_d, imgSize_d, upLeftCordinate_d, pixelSize_d, toler_d, x0_d, x1_d, x2_d, y0_d, y1_d, y2_d, z0_d, z1_d, z2_d, triCount_d, row_d, col_d);
			cudaError_t r3 = cudaGetLastError();

			cudaMemcpy(row_h, row_d, triCount * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(col_h, col_d, triCount * sizeof(int), cudaMemcpyDeviceToHost);
			cudaError_t r4 = cudaGetLastError();

			for (int i = 0; i < triCount; i++)
			{
				int x = col_h[i];
				int y = row_h[i];
				if ((x == 0 && y == 0) || pntCount >= imgWidth * imgHeight)
					continue;
				float z = psrc[y * imgWidth + x];
				pPnt[pntCount].x = bboxMinX + x * dx;
				pPnt[pntCount].y = bboxMaxY - y * dy;
				pPnt[pntCount].z = z;
				pntCount++;
				if (pdes[y * imgWidth + x] > 0.0001) continue;
				pdes[y * imgWidth + x] = z;
				pIdx[y * imgWidth + x] = (float)pntCount;
				flag = true;
			}

#pragma region 释放动态参数
			cudaFree(x0_d);
			cudaFree(x1_d);
			cudaFree(x2_d);
			cudaFree(y0_d);
			cudaFree(y1_d);
			cudaFree(y2_d);
			cudaFree(z0_d);
			cudaFree(z1_d);
			cudaFree(z2_d);
			cudaFree(triCount_d);
			cudaFree(row_d);
			cudaFree(col_d);
			free(x0_h);
			free(x1_h);
			free(x2_h);
			free(y0_h);
			free(y1_h);
			free(y2_h);
			free(z0_h);
			free(z1_h);
			free(z2_h);
			free(row_h);
			free(col_h);
#pragma endregion			
			cudaError_t r5 = cudaGetLastError();

			if (flag == false || pntCount >= imgHeight * imgWidth)
				break;
			if (pntCount == pntIsAddedNum)
				break;   //如果pntCount数目没有发生变化说明没有新的点加入，则退出

			//构建新的TIN，以便进行下一次循环计算
			delete pTinDll;
			pTinDll = new CTINClass("pczAenVQ");
			pTinDll->BeginAddPoints();

			for (int i = 0; i < pntCount; i++)
			{
				pTinDll->AddPoint(pPnt[i].x, pPnt[i].y, pPnt[i].z, 0, NULL);
			}
			pTinDll->EndAddPoints();
			pTinDll->FastConstruct();
#pragma endregion
		}

		indicator++;
	}

	cudaFree(imgSize_d);
	cudaFree(upLeftCordinate_d);
	cudaFree(pixelSize_d);
	cudaFree(toler_d);
	cudaFree(psrc_d);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
	cudaFree(imgSize_d);
	cudaFree(border_d);
	cudaFree(psrc_d);

	delete pPnt;		pPnt = NULL;
	delete pTinDll;		pTinDll = NULL;

	cout << "特征点提取完成！" << endl;

	return TRUE;
}