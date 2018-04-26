// ComBase.h: interface for the CComBase class.
//

#if !defined(AFX_COMBASE_H__C606D7F6_9F66_4F4C_B994_A70CAE97BD92__INCLUDED_)
#define AFX_COMBASE_H__C606D7F6_9F66_4F4C_B994_A70CAE97BD92__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "gdal.h"
#include "gdal_priv.h"
#include "shapefil.h"
#include <vector>
#include <algorithm>

#define DBL_MAX         1.7976931348623158e+308 /* max value */

struct point
{
	double x, y;

	point(){
	};

	point(double X, double Y) :x(X), y(Y){
	}

	bool operator == (const point &a)const{
		return x == a.x && y == a.y;
	}

	bool operator != (const point &a)const{
		return x != a.x || y != a.y;
	}

	bool operator < (const point &a)const{
		return x == a.x ? y<a.y : x<a.x;
	}

	bool operator >(const point &a)const{
		return x == a.x ? y>a.y : x>a.x;
	}
};

struct point3D
{
	double x, y, z;
	double val;  //Sort by the val

	point3D(){
	};

	point3D(double X, double Y, double Z) :x(X), y(Y), z(Z){
	}

	point3D(double X, double Y, double Z, double value) :x(X), y(Y), z(Z), val(value){
	}

	bool operator == (const point3D &a)const{
		return x == a.x && y == a.y && z == a.z;
	}

	bool operator != (const point3D &a)const{
		return x != a.x || y != a.y || z != a.z;
	}

	bool operator < (const point3D &a)const{
		return val<a.val;
	}

	bool operator >(const point3D &a)const{
		return val>a.val;
	}
};

typedef std::vector<int>      intArray;
typedef std::vector<point>    polyline;   //Define a fold line
typedef std::vector<point3D>  polyline3D; //Define a fold line

class CComBase
{
public:
	CComBase();
	virtual ~CComBase();

public:
	//Get the suffix name of the image
	CString GetImgFormat(LPCTSTR lpstrFileName);

	//Get the type name that the GDAL can handle by the suffix name of the image
	CString GetGDALFormat(CString imgFormat);

	//Release the data set of the GDALDataset 
	void  RELEASE(GDALDataset* pData);

	//Determine whether the input strImgName is valid, and the imgName supports *. Tif & *.img format here
	BOOL  bImgNameVerf(CString strImgName, CString& imgFormat, CString& strFormat);

	//Open an image and obtain the width and height imformation of the image
	BOOL  OpenImg(BSTR ImgName, int& imgWidth, int& imgHeight);

	//Open an image and obtain the imformation of the image(including the projection information)
	BOOL  OpenImg(BSTR ImgName, int imgWidth, int imgHeight, double& dx, double& dy,
		double& Xmin, double& Ymax, CString& projRef, float *pBuffer);

	//Create a new image (with projection information)
	BOOL  CreateNewImg(CString strImgName, int imgWidth, int imgHeight, double Xmin, double Ymax,
		double dx, double dy, double invalidVal, CString projRef, float *demZ);

	//Whether the point (cenX,cenY,cenZ) is within the triangle (xt,yt,zt) 
	//Whether the distance of the line is equal to the sum of the distances of the point to the two endpoints of the line
	BOOL IsInEdge(double cenX, double cenY, double* xt, double* yt);

	//Whether the point (cenX,cenY,cenZ) is within the triangle (xt,yt,zt) (including the boundary)
	//Whether the area of the tirangle is equal to the sum of the areas of the three triangle composed by the point with three vertices of the triangle
	BOOL IsInTriangle(double cenX, double cenY, double* xt, double* yt);

	//Calculate the area of a triangle according to the coordinates of the three vertex 
	double AreaTrig(double x1, double y1, double x2, double y2, double x3, double y3);
};

#endif // !defined(AFX_COMBASE_H__C606D7F6_9F66_4F4C_B994_A70CAE97BD92__INCLUDED_)
