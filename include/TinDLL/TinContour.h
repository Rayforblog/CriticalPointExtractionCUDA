// TinContour.h: interface for the CTinContour class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TINCONTOUR_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_)
#define AFX_TINCONTOUR_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000

#ifdef TINDLL_EXPORTS
#define TINDLL_API __declspec(dllexport)
#else
#define TINDLL_API __declspec(dllimport)
#endif

#include "TINClass.h"

class TINDLL_API CTinContour  
{
private:
	CTINClass *m_tin;

	long	 m_triangleSum;
	TRIANGLE **m_triangles;

	int m_tri;
	int m_cntpSum;
	int m_IndexFlag;
	realPOINT *m_cnt;

	float m_z;
	float m_minZ, m_maxZ, m_idz, m_idz5;

	void ReverseCnt(void);
	void TraceACnt(float z, TRIANGLE *tri);
	void SetCntMarks(float z);
	void SetCntMarks(float z, void *extraAttr );

	void LinearInterpolate(triPOINT *p1, triPOINT *p2, float z );
	void NonLinearInterpolate(triPOINT *p1, triPOINT *p2, float z, triEDGE &tri  );

public:
	CTinContour(CTINClass *theTin, float xmin, float ymin, float xmax, float ymax, float minZ,float maxZ,float idz);
	virtual ~CTinContour();

	realPOINT *GetNextContour(int *cntpSum, float *zVal, void *extraAttr = NULL);

	void SmoothContour( realPOINT *cnt, int n);
};


// for deformation of fan triangles


struct	fanTRI	{
	triPOINT *tinPt[2];		// org and dest
	double a, b;			// axis length of ellipse
	double cosA, sinA;	// angle for axis a
};



struct	fanPlusTRI	{
	triPOINT *tinPt[2];	// begin and end point of fan triangles

	triPOINT *lPt;		// left side point, on the same contour	
	triPOINT *rPt;		// right side point, on the same contour

	int	lSum;			// fan segments at left
	int	rSum;			// an segments at right, it maybe the index of the mid point 

	int sign;			// 

	/////////////
//	double	lRatio;		// the ratio it get
//	double	rRatio;		// the ratio it get
	
//	double lenL, lenR;	//
//	double cosL, sinL;	// left point
//	double cosR, sinR;	// right point

}	;


// edge.sh->fanIndex
//
//  contour point index on the fan	
// 

//
// extraAttr, value meaning, 
//	0 - contour point 
//  1 - contour point of  flat triangle
//
//  400 - inserted point
//  500 - root point or inserted point, need deformation process
//
//  > 10000:  pointer of fan structur of fanTRI
//
//

#define markINS	400
#define markFAN 500

class TINDLL_API CTINContourPreProceess
{
private: 
	// will be removed automatically
	CMemoryPool <fanTRI, fanTRI &> m_fanTriPool;

	CMemoryPool <fanPlusTRI, fanPlusTRI &> m_fanPlusPool;

	CTINClass *m_tin;

private:
	void AddFanTriangle ( triPOINT *pt);

	void AddFanPlusTriangle ( triPOINT *pt);

	int	 F100I( float z )	{	return (int)(100*z);	};
public:
	CTINContourPreProceess( CTINClass *theTin );
};


double TINDLL_API Dist2dZ( double r, double dSum, double d, double dzSum );
double TINDLL_API Dz2Dist( double r, double dzSum, double dz, double dSum );



#endif // !defined(AFX_TINCONTOUR_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_)
