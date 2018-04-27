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

//x, y, z array（number of member is 3）:record the row number, column number and elevation value of the three vertices of a triangle
//border:record the row and column number of the upper left and right corner of the rectangular area in the original image
//psrc array（number of member is imgWidth * imgHeight）:record the pixel value of the original image
//imgSize array（number of member is 2）:record the row and column number of the original image
//elv:record the difference in elevation of all of the point within the triangle (the value is 0 when the point is not within the triangle )
__global__ void kernelPixel(int *x, int *y, double *z, int *border, float *psrc, int *imgSize, float *elv)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = gridDim.x * blockDim.x * tidy + tidx;

	int borderWidth = border[2] - border[0] + 1;
	int borderHeight = border[3] - border[1] + 1;
	
	//The current thread is terminated when it is without the rectangle area
	if (tid >= borderWidth * borderHeight)
		return;

	//Calculate the row and column number  in the rectangular area
	int br = tid / borderWidth;
	int bc = tid % borderWidth;

	//The position in the original image is calculated 
	//according to the position in the rectangle area and the position in the upper left corner of the rectangle
	int oc = bc + border[0];
	int or = br + border[1];

	//Calculate the position of the point in the psrc array
	int index = imgSize[0] * or + oc;

	//Calculate the coordinate value in the x,y and z directions
	int x3 = oc;
	int y3 = or;
	double z3 = psrc[index];

	//The area method calculates whether the point is within the triangle range
	//If it is without the scope, the current thread is terminated
	//Calculate the area of the original triangle
	double ds0 = sqrt((double)((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1])));
	double ds1 = sqrt((double)((x[0] - x[2]) * (x[0] - x[2]) + (y[0] - y[2]) * (y[0] - y[2])));
	double ds2 = sqrt((double)((x[1] - x[2]) * (x[1] - x[2]) + (y[1] - y[2]) * (y[1] - y[2])));
	double t0 = (double)((ds0 + ds1 + ds2) * (0.5));
	double s0 = sqrt((double)(t0 * (t0 - ds0) * (t0 - ds1) * (t0 - ds2)));
	
	//Calculate the area of the three triangles that are composed of the calculated point and the three vertices of the original triangle
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

	//Calculate the surface equation of the triangle and the difference in elevation
	double t = (x[0] - x[1]) * (y[0] - y[2]) - (x[0] - x[2]) * (y[0] - y[1]);
	if (t == 0)
		return;
	double a = ((y[0] - y[2]) * (z[0] - z[1]) - (y[0] - y[1]) * (z[0] - z[2])) * ((1.0f) / t);
	double b = -((x[0] - x[2]) * (z[0] - z[1]) - (x[0] - x[1]) * (z[0] - z[2])) * ((1.0f) / t);
	double c = z[0] - a * x[0] - b * y[0];
	double offset = fabsf(z3 - a * x3 - b * y3 - c);

	elv[tid] = offset;
}

//psrc array(number of member is imgWidth * imgHeight）:record the pixel value of the original image
//imgSize array(number of member is 2）:record the row and column of the original image
//upLeftCord array(number of member is 2）:record the top left geographic coordinates of the upper left corner（x, y)
//pixelSizearray(number of member is 2）:record the pixel resolution（ dx, dy)
//toler:the threshold value
//x0, y0, z0, x1, y1, z1, x2, y2, z2:record the coordinates of three vertices of the triangle
//triCount:reord the number of the triangle
//row:record the row number of the point with the maximum difference in elevation which is larger that the toler
//col:record the column number of the point with the maximum difference in elevation which is larger that the toler
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

	//Calculate the  plane formula of the triangular
	double t = (colt[0] - colt[1])*(rowt[0] - rowt[2]) - (colt[0] - colt[2])*(rowt[0] - rowt[1]);
	if (t == 0) return;
	double a = ((rowt[0] - rowt[2]) * (alt[0] - alt[1]) - (rowt[0] - rowt[1]) * (alt[0] - alt[2]))*(1.0f / t);
	double b = -((colt[0] - colt[2]) * (alt[0] - alt[1]) - (colt[0] - colt[1]) * (alt[0] - alt[2]))*(1.0f / t);
	double c = alt[0] - a * colt[0] - b * rowt[0];

	//Calculate the area of the current triangular
	double ds01 = sqrt((double)(colt[0] - colt[1])*(colt[0] - colt[1]) + (rowt[0] - rowt[1])*(rowt[0] - rowt[1]));
	double ds02 = sqrt((double)(colt[0] - colt[2])*(colt[0] - colt[2]) + (rowt[0] - rowt[2])*(rowt[0] - rowt[2]));
	double ds12 = sqrt((double)(colt[1] - colt[2])*(colt[1] - colt[2]) + (rowt[1] - rowt[2])*(rowt[1] - rowt[2]));
	double pm012 = (ds01 + ds02 + ds12) * 0.5;
	double s012 = sqrt(pm012 * (pm012 - ds01) * (pm012 - ds02) * (pm012 - ds12));

	//Calculate the maximum difference in elevation  and its row and column numbers
	double maxVal = 0;
	int maxX = 0, maxY = 0;
	for (int i = yMin; i < yMax; i++)
	for (int j = xMin; j < xMax; j++)
	{
		float pVal = psrc[i * imgSize[0] + j];
		if (pVal <= 0) continue;

		//Helen formula method determines whether the calculated point is within the triangle area
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

		//Calculate the maximum difference in elevation
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

//Add a point to the TIN
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

//It adds the point with  the maximum  difference in elevation over each triangle during one iteration
extern "C" BOOL ztolerancePointSelect(float *psrc, int imgWidth, int imgHeight, double bboxMinX, double bboxMaxY, double dx, double dy, double ztolerValue, float *pdes, float *pIdx)
{
	cout << "Enter the ztolerancePointSelect1By1." << endl;
	point3D* pPnt = new point3D[imgWidth*imgHeight];     //Record the point information used to build the TIN
	long pntCount = 0;									//Record the number of the critical point

	CTINClass *pTinDll = new CTINClass("pczAenVQ");
	pTinDll->BeginAddPoints();

	//Take the four points of the image to construct the first TIN
	//If the number of obtained point is less than 4, it is terminated
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
	pTinDll->EndAddPoints();
	pTinDll->FastConstruct();

	//Copy the data into the device
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

	int x_h[3], y_h[3], border_h[4];		//Record the row and column numbers of the triangle;the minimum and maximum row and column numbers of the external rectangular region
	double z_h[3];							//Record the elevation of the three vertices of the triangle

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
#pragma region Traverse the pixel point in the external rectangle
			pTinDll->TriangleTraversalInit();
			TRIANGLE *tri = 0;

			double x[3], y[3], z[3];					//Record the three vertices of the triangle, where x,y and z coordinate is row number, column number and the elevation
			long pntIsAddedNum = pntCount;				//Record the number of initial pntCount used to determine whether the number of points to build TIN has increased
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

				//Calculate the range of the rectangle
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

				//Define the thread
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

			//If the number of pntCount does not change,it indicates that no new points are added and then exits
			if (flag == false || pntCount >= imgHeight * imgWidth)
				break;
			if (pntCount == pntIsAddedNum)
				break;   

			//Construct a new TIN for the next loop calculation
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
#pragma region Traverse all of the triangle
			long pntIsAddedNum = pntCount;   //Record the number of initial pntCount used to determine whether the number of points to construct the TIN has increased
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

			//Returns the point with the maximum difference in elevation in the triangle
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

#pragma region Release dynamic parameter
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
			
			//If the number of pntCount does not change,it indicates that no new points are added and then exits
			if (pntCount == pntIsAddedNum)
				break;

			//Construct a new TIN for the next loop calculation
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

	cout << "It had finished the extraction of the critical point！" << endl;

	return TRUE;
}