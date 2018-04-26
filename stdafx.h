// stdafx.h : the include file including the standard system contains files,
// or often used but not often changed
// project-specific include file
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // Some CString constructors will be explicit

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            //  Exclude very rarely used information from Windows header files
#endif

#include <afx.h>
#include <afxwin.h>         // MFC Core components and standard components
#include <afxext.h>         // MFC extension
#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           // MFC Support for the public controls of the Internet Explorer 4
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>                     // MFC Support for Windows public controls
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <iostream>



// TODO: The other header files required by the program are referenced here
