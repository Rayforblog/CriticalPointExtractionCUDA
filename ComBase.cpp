// ComBase.cpp: implementation of the CComBase class
//

#include "stdafx.h"
#include "resource.h"
#include "ComBase.h"

#include <string>
#include <cstringt.h>
using namespace std;

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CComBase::CComBase()
{

}

CComBase::~CComBase()
{

}

//获得影像的后缀名
CString CComBase::GetImgFormat(LPCTSTR lpstrFileName)
{
	string saveFileName = string((char*)lpstrFileName);

	string suffix = "";

	if (saveFileName.length() != 0)
	{
		const char* charName;
		charName = saveFileName.data();
		charName = strrchr(charName, '.');
		if (charName)
			suffix = charName + 1;
	}

	CString result(suffix.c_str());

	return result;
}

//通过影像的后缀名得到对应GDAL能处理的类型名
CString CComBase::GetGDALFormat(CString imgFormat)
{
	CString strFormat;

	if (imgFormat.GetLength() == 0)
	{
		return strFormat;
	}

	if (imgFormat == "bmp")
		strFormat = "BMP";
	else if (imgFormat == "bt")
		strFormat = "BT";
	else if (imgFormat == "gif")
		strFormat = "GIF";
	else if (imgFormat == "img")
		strFormat = "HFA";
	else if (imgFormat == "jpg")
		strFormat = "JPEG";
	else if (imgFormat == "png")
		strFormat = "PNG";
	else if (imgFormat == "tif")
		strFormat = "GTiff";
	else if (imgFormat == "vrt")
		strFormat = "VRT";

	return strFormat;
}

//释放GDALDataset数据集
void  CComBase::RELEASE(GDALDataset* pData)
{
	if (pData != NULL)
	{
		GDALClose((GDALDatasetH)pData);
		pData = NULL;
	}
}

//判断输入的strImgName是否有效，此处的imgName支持*.tif & *.img格式
BOOL  CComBase::bImgNameVerf(CString strImgName, CString& imgFormat, CString& strFormat)
{
	if (strImgName.GetLength()<1)	return FALSE;

	imgFormat = GetImgFormat(strImgName);
	strFormat = GetGDALFormat(imgFormat);

	return TRUE;
}


//创建一个新的影像(带投影信息)。返回True说明创建成功，否则创建失败
//ImgName:输入影像的路径
//imgWidth、imgHeight:设置影像的宽和高
//Xmin、Ymax:设置影像中X、Y的范围
//dx、dy:设置影像的格网分辨率
//invalidVal:设置影像中的无效值
//demZ:设置影像中每个像素的值
BOOL  CComBase::CreateNewImg(CString strImgName, int imgWidth, int imgHeight, double Xmin, double Ymax, double dx, double dy, double invalidVal, CString projRef, float *demZ)
{
	GDALAllRegister();

	//检查输出影像格式是否正确
	CString imgFormat, strFormat;
	if (bImgNameVerf(strImgName, imgFormat, strFormat) == FALSE)
	{
		printf("影像格式错误！\n");
		return FALSE;
	}

	GDALDriverH hDriver = NULL;
	USES_CONVERSION;
	LPSTR charFormat = T2A(strFormat);
	LPCSTR charImgName = T2A(strImgName);

	hDriver = GetGDALDriverManager()->GetDriverByName("HFA");
	if (hDriver == NULL) printf("hDriver == NULL \n");
	if (hDriver == NULL || GDALGetMetadataItem(hDriver, GDAL_DCAP_CREATE, NULL) == NULL)
	{
		printf("hDriver == NULL || GDALGetMetadataItem( hDriver, GDAL_DCAP_CREATE, NULL ) == NULL\n");
		return FALSE;
	}

	char **papszOptions = NULL;
	GDALDataset *pDataset = (GDALDataset *)GDALCreate(hDriver, charImgName, imgWidth, imgHeight, 1, GDT_Float32, papszOptions);

	double adfGeoTransform[6] = { Xmin, dx, 0, Ymax, 0, -dy };
	pDataset->SetGeoTransform(adfGeoTransform);
	LPCSTR charprojRef = T2A(projRef);

	pDataset->SetProjection(charprojRef);
	GDALRasterBand *pBand = pDataset->GetRasterBand(1);

	pBand->SetNoDataValue((double)invalidVal);  //设置波段的无效值，这里设置影像=invalidVal为无效值
	pBand->RasterIO(GF_Write, 0, 0, imgWidth, imgHeight, demZ, imgWidth, imgHeight, GDT_Float32, 0, 0);
	double min = 0, max = 0, mean = 0, dev = 0; //设置的影像的最大值和最小值
	pBand->ComputeStatistics(FALSE, &min, &max, &mean, &dev, NULL, NULL);

	GDALDeleteDataset(hDriver, charImgName);
	RELEASE(pDataset);

	return TRUE;
}


//打开一影像数据，获得该影像的信息。返回True说明创建成功，否则创建失败
//ImgName:输入影像的路径
//imgWidth,imgHeight:返回影像的宽、高
BOOL  CComBase::OpenImg(BSTR ImgName, int& imgWidth, int& imgHeight)
{
	CString strImgName(ImgName);
	if (strImgName.GetLength()<1)
	{
		cout << "length < 1,return FALSE!" << endl;
		return FALSE;
	}

	//打开DEM影像    
	GDALDataset *pImgDataset = NULL;
	USES_CONVERSION;
	LPCSTR charImgName = T2A(strImgName);

	int len = strlen(charImgName);

	pImgDataset = (GDALDataset *)GDALOpen(charImgName, GA_ReadOnly);
	if (pImgDataset == NULL)
	{
		cout << "Img cannot open here!" << endl;
		return FALSE;
	}

	imgHeight = pImgDataset->GetRasterYSize();
	imgWidth = pImgDataset->GetRasterXSize();

	cout << "DEM已经打开," << "高为" << imgHeight << ",宽为" << imgWidth << "!" << endl;

	RELEASE(pImgDataset);

	if (imgHeight <= 0 || imgWidth <= 0)
		return FALSE;

	return TRUE;
}

//打开一影像数据，获得该影像的信息(包括投影信息)。返回True说明创建成功，否则创建失败
//ImgName:输入影像的路径；
//imgWidth,imgHeight:影像的宽、高；
//dx,dy:返回影像的分辨率；
//Xmin,Ymax:返回影像的坐标范围；
//pBuffer:返回影像的各个像素值。
BOOL  CComBase::OpenImg(BSTR ImgName, int imgWidth, int imgHeight, double& dx, double& dy, double& Xmin, double& Ymax, CString& projRef, float *pBuffer)
{
	CString strImgName(ImgName);
	if (strImgName.GetLength()<1 || pBuffer == NULL)
	{
		cout << "strImgName.GetLength()<1||pBuffer==NULL,return FALSE!" << endl;
		return FALSE;
	}

	//打开DEM影像    
	GDALDataset *pImgDataset = NULL;

	USES_CONVERSION;
	LPCSTR charImgName = T2A(strImgName);

	pImgDataset = (GDALDataset *)GDALOpen(charImgName, GA_ReadOnly);
	if (pImgDataset == NULL)
	{
		cout << "img cannot open!" << endl;
		return FALSE;
	}

	double geoTransform[6];
	pImgDataset->GetGeoTransform(geoTransform);  //获得DEM的坐标范围，以及像素分辨率
	dx = geoTransform[1], dy = fabs(geoTransform[5]);
	Xmin = geoTransform[0], Ymax = geoTransform[3];

	projRef = pImgDataset->GetProjectionRef();

	int BandNum = pImgDataset->GetRasterCount();
	GDALRasterBand *pBand = pImgDataset->GetRasterBand(BandNum);
	if (pBand == NULL)
	{
		RELEASE(pImgDataset);
		return FALSE;
	}
	pBand->RasterIO(GF_Read, 0, 0, imgWidth, imgHeight, pBuffer, imgWidth, imgHeight, GDT_Float32, 0, 0);

	RELEASE(pImgDataset);

	if (pBuffer == NULL || imgWidth <= 0 || imgHeight <= 0 || dx <= 0 || dy <= 0)
		return FALSE;

	return TRUE;
}

//已知三顶点坐标求三角形的面积
double CComBase::AreaTrig(double x1, double y1, double x2, double y2, double x3, double y3)
{
	double a = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
	double b = sqrt((x1 - x3)*(x1 - x3) + (y1 - y3)*(y1 - y3));
	double c = sqrt((x3 - x2)*(x3 - x2) + (y3 - y2)*(y3 - y2));

	double p = (a + b + c) / 2;

	double area = sqrt(p*(p - a)*(p - b)*(p - c));
	return area;
}

//判断点(cenX,cenY,cenZ)是否在三角形(xt,yt,zt)的内(包括边界)。判断一个点到三角形各个顶点的面积和是否等于三角形面积
BOOL CComBase::IsInTriangle(double cenX, double cenY, double* xt, double* yt)
{
	double c01 = AreaTrig(cenX, cenY, xt[0], yt[0], xt[1], yt[1]);
	double c02 = AreaTrig(cenX, cenY, xt[0], yt[0], xt[2], yt[2]);
	double c21 = AreaTrig(cenX, cenY, xt[2], yt[2], xt[1], yt[1]);

	double trig = AreaTrig(xt[0], yt[0], xt[1], yt[1], xt[2], yt[2]);

	if (fabs(trig - (c01 + c02 + c21))<0.00001)
		return true;

	return false;
}

//判断点(cenX,cenY,cenZ)是否在三角形(xt,yt,zt)的边界上。判断一个点到线两个端点距离和是否等于线的距离
BOOL CComBase::IsInEdge(double cenX, double cenY, double* xt, double* yt)
{
	double dis01 = sqrt((xt[0] - xt[1])*(xt[0] - xt[1]) + (yt[0] - yt[1])*(yt[0] - yt[1]));
	double dis02 = sqrt((xt[0] - xt[2])*(xt[0] - xt[2]) + (yt[0] - yt[2])*(yt[0] - yt[2]));
	double dis12 = sqrt((xt[2] - xt[1])*(xt[2] - xt[1]) + (yt[2] - yt[1])*(yt[2] - yt[1]));

	double disc0 = sqrt((cenX - xt[0])*(cenX - xt[0]) + (cenY - yt[0])*(cenY - yt[0]));
	double disc1 = sqrt((cenX - xt[1])*(cenX - xt[1]) + (cenY - yt[1])*(cenY - yt[1]));
	double disc2 = sqrt((cenX - xt[2])*(cenX - xt[2]) + (cenY - yt[2])*(cenY - yt[2]));

	if (fabs((disc0 + disc1) - dis01)<0.00001 || fabs((disc0 + disc2) - dis02)<0.00001 || fabs((disc2 + disc1) - dis12)<0.00001)
		return true;

	return false;
}