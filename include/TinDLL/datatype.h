#if !defined( _dataType_h_INC_ )

#define _dataType_h_INC_


#include "TINClass.h"



struct DEMInfo {
	double x0,y0,dirAng;
	short nx,ny;
	float dx,dy;
}	;	



//点
struct POINT3D {
	double x,y,z;
	triPOINT *tinPt;
};

//点集
//线
// 
enum objTYPE {
	typePOINTSET,
	typePOLYLINE
};



struct geoOBJ {
	BYTE type;
	BYTE delFlag;
	BYTE color;
	short pSum;
	POINT3D *pts;
};



struct rectREGISTER	{
	int maxSum;
	int objSum;
	geoOBJ **objs;
} ;



//文件格式

struct fileHEAD {
	char tag[32];
	long objSum;
	long fileSize;
	double xOffset, yOffset;
}	;

//三角形的三个顶点
struct triVERTEXES {
	triPOINT *vertex[3];
};


//插入点指针串
struct InsertPOINT {
	triPOINT *insertpt;
	InsertPOINT *father;
	double  distance;//累加距离
} ;


//角点数组
struct cornerPOINT {
	triPOINT *triPoint;
	InsertPOINT *insPoint;
	double  distance;//累加距离
} ;


/////////////////////////////



#endif