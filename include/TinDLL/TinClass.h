#if !defined(AFX_TINCLASS_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_)
#define AFX_TINCLASS_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_

#include "MemoryPool.h"

#if !defined( tREAL )

#define tREAL double

struct realPOINT {
	tREAL x,y;
}	;

#endif

#define _DUMMY_MARKER		-9999999L
#define _DUMMY_POINT_ATTR	-9999999.0
#define _MAX_POINT_ATTR	9999990.0
#define _MIN_POINT_ATTR	-9999990.0


///////////////////////////////////////////////////

/* Labels that signify the result of triPOINT location.  The result of a        */
/*   search indicates that the triPOINT falls in the interior of a TRIANGLE, on */
/*   an shEDGE, on a vertex, or outside the mesh.                              */

enum locateresult {INTRIANGLE, ONEDGE, ONVERTEX, OUTSIDE};

/* Labels that signify the result of site insertion.  The result indicates   */
/*   that the triPOINT was inserted with complete success, was inserted but     */
/*   encroaches on a segment, was not inserted because it lies on a segment, */
/*   or was not inserted because another triPOINT occupies the same location.   */

enum insertsiteresult {SUCCESSFULPOINT, ENCROACHINGPOINT, VIOLATINGPOINT,DUPLICATEPOINT};

/* Labels that signify the result of direction finding.  The result          */
/*   indicates that a segment connecting the two query points falls within   */
/*   the direction TRIANGLE, along the left shEDGE of the direction TRIANGLE,  */
/*   or along the right shEDGE of the direction TRIANGLE.                      */

enum finddirectionresult {WITHIN, LEFTCOLLINEAR, RIGHTCOLLINEAR};

/* Labels that signify the result of the circumcenter computation routine.   */
/*   The return value indicates which shEDGE of the TRIANGLE is shortest.      */

enum circumcenterresult {OPPOSITEORG, OPPOSITEDEST, OPPOSITEAPEX};

/*****************************************************************************/
/*                                                                           */
/*  The basic mesh data structures                                           */
/*                                                                           */
/*  There are three:  points, triangles, and shell edges (abbreviated        */
/*  `SHELLE').  These three data structures, linked by pointers, comprise    */
/*  the mesh.  A triPOINT simply represents a triPOINT in space and its properties.*/
/*  A TRIANGLE is a TRIANGLE.  A shell shEDGE is a special data structure used */
/*  to represent impenetrable segments in the mesh (including the outer      */
/*  boundary, boundaries of holes, and internal boundaries separating two    */
/*  triangulated regions).  Shell edges represent boundaries defined by the  */
/*  user that triangles may not lie across.                                  */
/*                                                                           */
/*  A TRIANGLE consists of a list of three vertices, a list of three         */
/*  adjoining triangles, a list of three adjoining shell edges (when shell   */
/*  edges are used), an arbitrary number of optional user-defined floating-  */
/*  triPOINT attributes, and an optional area constraint.  The latter is an     */
/*  upper bound on the permissible area of each TRIANGLE in a region, used   */
/*  for mesh refinement.                                                     */
/*                                                                           */
/*  For a TRIANGLE on a boundary of the mesh, some or all of the neighboring */
/*  triangles may not be present.  For a TRIANGLE in the interior of the     */
/*  mesh, often no neighboring shell edges are present.  Such absent         */
/*  triangles and shell edges are never represented by NULL pointers; they   */
/*  are represented by two special records:  `dummytri', the TRIANGLE that   */
/*  fills "outer space", and `dummysh', the omnipresent shell shEDGE.          */
/*  `dummytri' and `dummysh' are used for several reasons; for instance,     */
/*  they can be dereferenced and their contents examined without causing the */
/*  memory protection exception that would occur if NULL were dereferenced.  */
/*                                                                           */
/*  However, it is important to understand that a TRIANGLE includes other    */
/*  information as well.  The pointers to adjoining vertices, triangles, and */
/*  shell edges are ordered in a way that indicates their geometric relation */
/*  to each other.  Furthermore, each of these pointers contains orientation */
/*  information.  Each pointer to an adjoining TRIANGLE indicates which face */
/*  of that TRIANGLE is contacted.  Similarly, each pointer to an adjoining  */
/*  shell shEDGE indicates which side of that shell shEDGE is contacted, and how */
/*  the shell shEDGE is oriented relative to the TRIANGLE.                     */
/*                                                                           */
/*  Shell edges are found abutting edges of triangles; either sandwiched     */
/*  between two triangles, or resting against one TRIANGLE on an exterior    */
/*  boundary or hole boundary.                                               */
/*                                                                           */
/*  A shell shEDGE consists of a list of two vertices, a list of two           */
/*  adjoining shell edges, and a list of two adjoining triangles.  One of    */
/*  the two adjoining triangles may not be present (though there should      */
/*  always be one), and neighboring shell edges might not be present.        */
/*  Shell edges also store a user-defined integer "boundary marker".         */
/*  Typically, this integer is used to indicate what sort of boundary        */
/*  conditions are to be applied at that location in a finite element        */
/*  simulation.                                                              */
/*                                                                           */
/*  Like triangles, shell edges maintain information about the relative      */
/*  orientation of neighboring objects.                                      */
/*                                                                           */
/*  Points are relatively simple.  A triPOINT is a list of floating triPOINT       */
/*  numbers, starting with the x, and y coordinates, followed by an          */
/*  arbitrary number of optional user-defined floating-triPOINT attributes,     */
/*  followed by an integer boundary marker.  During the segment insertion    */
/*  phase, there is also a pointer from each triPOINT to a TRIANGLE that may    */
/*  contain it.  Each pointer is not always correct, but when one is, it     */
/*  speeds up segment insertion.  These pointers are assigned values once    */
/*  at the beginning of the segment insertion phase, and are not used or     */
/*  updated at any other time.  Edge swapping during segment insertion will  */
/*  render some of them incorrect.  Hence, don't rely upon them for          */
/*  anything.  For the most part, points do not have any information about   */
/*  what triangles or shell edges they are linked to.                        */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  Handles                                                                  */
/*                                                                           */
/*  The oriented TRIANGLE (`triEDGE') and oriented shell shEDGE (`shEDGE') data  */
/*  structures defined below do not themselves store any part of the mesh.   */
/*  The mesh itself is made of `TRIANGLE's, `SHELLE's, and `triPOINT's.         */
/*                                                                           */
/*  Oriented triangles and oriented shell edges will usually be referred to  */
/*  as "handles".  A handle is essentially a pointer into the mesh; it       */
/*  allows you to "hold" one particular part of the mesh.  Handles are used  */
/*  to specify the regions in which one is traversing and modifying the mesh.*/
/*  A single `TRIANGLE' may be held by many handles, or none at all.  (The   */
/*  latter case is not a memory leak, because the TRIANGLE is still          */
/*  connected to other triangles in the mesh.)                               */
/*                                                                           */
/*  A `triEDGE' is a handle that holds a TRIANGLE.  It holds a specific side */
/*  of the TRIANGLE.  An `shEDGE' is a handle that holds a shell shEDGE.  It     */
/*  holds either the left or right side of the shEDGE.                         */
/*                                                                           */
/*  Navigation about the mesh is accomplished through a set of mesh          */
/*  manipulation primitives, further below.  Many of these primitives take   */
/*  a handle and produce a new handle that holds the mesh near the first     */
/*  handle.  Other primitives take two handles and glue the corresponding    */
/*  parts of the mesh together.  The exact position of the handles is        */
/*  important.  For instance, when two triangles are glued together by the   */
/*  bond() primitive, they are glued by the sides on which the handles lie.  */
/*                                                                           */
/*  Because points have no information about which triangles they are        */
/*  attached to, I commonly represent a triPOINT by use of a handle whose       */
/*  origin is the triPOINT.  A single handle can simultaneously represent a     */
/*  TRIANGLE, an shEDGE, and a triPOINT.                                          */
/*                                                                           */
/*****************************************************************************/

/* The TRIANGLE data structure.  Each TRIANGLE contains three pointers to    */
/*   adjoining triangles, plus three pointers to vertex points, plus three   */
/*   pointers to shell edges (defined below; these pointers are usually      */
/*   `dummysh').  It may or may not also contain user-defined attributes     */
/*   and/or a floating-triPOINT "area constraint".  It may also contain extra   */
/*   pointers for nodes, when the user asks for high-order elements.         */
/*   Because the size and structure of a `TRIANGLE' is not decided until     */
/*   runtime, I haven't simply defined the type `TRIANGLE' to be a struct.   */

typedef struct tagSHELLE	SHELLE;
typedef struct tagTriEDGE	triEDGE;
typedef struct tagShEDGE	shEDGE;

typedef struct tagTRIANGLE2 TRIANGLE;

struct triPOINT;

//!!!
// because there are three version of struct TRIANGLE
// the pointer TRIANGLE * can not be used for browsing directly  
//

/******* when order == 1  *******/
struct  tagTRIANGLE1 {
	TRIANGLE	*adjoin[3];
	triPOINT	*vertex[3];
	SHELLE		*sh[3];
	float		attr;
	float		area;
	long		node;		/* use area to replace */
	long		cntMark;
	void		*extraAttr;	// a structure pointer
}	;	


/******* when order == 2  *******/
struct  tagTRIANGLE2 {
	TRIANGLE	*adjoin[3];
	triPOINT	*vertex[3];
	SHELLE		*sh[3];
	float		attr;
	float		area;
	long		node;		/* use area to replace */
	long		cntMark;
	void		*extraAttr;	// a structure pointer
	triPOINT	*highorder[3];
}	;


/******* when order == 3  *******/
struct  tagTRIANGLE3 {
	TRIANGLE	*adjoin[3];
	triPOINT	*vertex[3];
	SHELLE		*sh[3];
	float		attr;
	float		area;
	long		node;		/* use area to replace */
	long		cntMark;
	void		*extraAttr;	// a structure pointer
	triPOINT	*highorder[7];
}	;


/* An oriented TRIANGLE:  includes a pointer to a TRIANGLE and orientation.  */
/*   The orientation denotes an shEDGE of the TRIANGLE.  Hence, there are      */
/*   three possible orientations.  By convention, each shEDGE is always        */
/*   directed to triPOINT counterclockwise about the corresponding TRIANGLE.    */

struct tagTriEDGE	{
	TRIANGLE *tri;
	int orient;                                         /* Ranges from 0 to 2. */
}	;


//typedef triEDGE triEDGE;

/* The shell data structure.  Each shell shEDGE contains two pointers to       */
/*   adjoining shell edges, plus two pointers to vertex points, plus two     */
/*   pointers to adjoining triangles, plus one shell marker.                 */

struct tagSHELLE {
	SHELLE		*adjoin[2];
	triPOINT	*vertex[2];
	TRIANGLE	*tri[2];
	long		marker;
	long		fanIndex;		// index of point on a FAN triangles
	void		*extraAttr;		// a structure pointer
};



//
// 与三角网内线段的交点
//
struct	profilePOINT	{
	double x, y, attr;
	double cosA;	// 线段之间的夹角余弦
	SHELLE *sh;
};


//typedef SHELLE SHELLE;                // old:  typedef SHELLE *SHELLE 

/* An oriented shell edge:  includes a pointer to a shell edge and an        */
/*   orientation.  The orientation denotes a side of the edge.  Hence, there */
/*   are two possible orientations.  By convention, the shEDGE is always       */
/*   directed so that the "side" denoted is the right side of the shEDGE.      */
//	
//	为什么是右侧?
//
//
struct tagShEDGE {
	SHELLE *sh;
	int shorient;                                       /* Ranges from 0 to 1. */
};

// typedef shEDGE shEDGE;

/* The triPOINT data structure.  Each triPOINT is actually an array of tREALs.      */
/*   The number of tREALs is unknown until runtime.  An integer boundary      */
/*   marker, and sometimes a pointer to a TRIANGLE, is appended after the    */
/*   tREALs.                                                                  */

struct triPOINT	{
	tREAL x,y;
	float attr;
	long marker;
	union 	{
		triPOINT *pt;
		int	freeCount;	// 
	}	dup;
	TRIANGLE *tri;
	void *extraAttr;	// a structure pointer
}	;


//typedef triPOINT triPOINT;


/* A queue used to store encroached segments.  Each segment's vertices are   */
/*   stored so that one can check whether a segment is still the same.       */

struct badSEGMENt {
	shEDGE encsegment;                          /* An encroached segment. */
	triPOINT segorg, segdest;                                /* The two vertices. */
	badSEGMENt  *nextsegment;     /* Pointer to next encroached segment. */
};

/* A queue used to store bad triangles.  The key is the square of the cosine */
/*   of the smallest angle of the TRIANGLE.  Each TRIANGLE's vertices are    */
/*   stored so that one can check whether a TRIANGLE is still the same.      */

struct badFACE {
	triEDGE badfacetri;                              /* A bad TRIANGLE. */
	tREAL key;                             /* cos^2 of smallest (apical) angle. */
	triPOINT *faceorg, *facedest, *faceapex;                  /* The three vertices. */
	badFACE *nextface;                 /* Pointer to next bad TRIANGLE. */
};



/* A node in a heap used to store events for the sweepline Delaunay          */
/*   algorithm.  Nodes do not point directly to their parents or children in */
/*   the heap.  Instead, each node knows its position in the heap, and can   */
/*   look up its parent and children in a separate array.  The `eventptr'    */
/*   points either to a `triPOINT' or to a TRIANGLE (in encoded format, so that */
/*   an orientation is included).  In the latter case, the origin of the     */
/*   oriented TRIANGLE is the apex of a "circle event" of the sweepline      */
/*   algorithm.  To distinguish site events from circle events, all circle   */
/*   events are given an invalid (smaller than `xmin') x-coordinate `xkey'.  */

struct sweepEVENT {
	tREAL xkey, ykey;                              /* Coordinates of the event. */
	union	{
		TRIANGLE *tri;
		triPOINT *pt;
		sweepEVENT *event;
	}	*eventPtr;       /* Can be a triPOINT or the location of a circle event. */
	int heapposition;              /* Marks this event's position in the heap. */
};

/* A node in the splay tree.  Each node holds an oriented ghost TRIANGLE     */
/*   that represents a boundary shEDGE of the growing triangulation.  When a   */
/*   circle event covers two boundary edges with a TRIANGLE, so that they    */
/*   are no longer boundary edges, those edges are not immediately deleted   */
/*   from the tree; rather, they are lazily deleted when they are next       */
/*   encountered.  (Since only a random sample of boundary edges are kept    */
/*   in the tree, lazy deletion is faster.)  `keydest' is used to verify     */
/*   that a TRIANGLE is still the same as when it entered the splay tree; if */
/*   it has been rotated (due to a circle event), it no longer represents a  */
/*   boundary shEDGE and should be deleted.                                    */

struct splayNODE {
	triEDGE keyedge;                  /* Lprev of an shEDGE on the front. */
	triPOINT *keydest;            /* Used to verify that splay node is still live. */
	splayNODE *lchild, *rchild;              /* Children in splay tree. */
};

///////////////////////////////////////////////////////////////////



/*****************************************************************************/
/*                                                                           */
/*  (triangle.h)                                                             */
/*                                                                           */
/*  Include file for programs that call Triangle.                            */
/*                                                                           */
/*  Accompanies Triangle Version 1.3                                         */
/*  July 19, 1996                                                            */
/*                                                                           */
/*  Copyright 1996                                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  How to call Triangle from another program                                */
/*                                                                           */
/*                                                                           */
/*  If you haven't read Triangle's instructions (run "triangle -h" to read   */
/*  them), you won't understand what follows.                                */
/*                                                                           */
/*  Triangle must be compiled into an object file (triangle.o) with the      */
/*  TRILIBRARY symbol defined (preferably by using the -DTRILIBRARY compiler */
/*  switch).  The makefile included with Triangle will do this for you if    */
/*  you run "make trilibrary".  The resulting object file can be called via  */
/*  the procedure triangulate().                                             */
/*                                                                           */
/*  If the size of the object file is important to you, you may wish to      */
/*  generate a reduced version of triangle.o.  The REDUCED symbol gets rid   */
/*  of all features that are primarily of research interest.  Specifically,  */
/*  the -DREDUCED switch eliminates Triangle's -i, -F, -s, and -C switches.  */
/*  The CDT_ONLY symbol gets rid of all meshing algorithms above and beyond  */
/*  constrained Delaunay triangulation.  Specifically, the -DCDT_ONLY switch */
/*  eliminates Triangle's -r, -q, -a, -S, and -s switches.                   */
/*                                                                           */
/*  IMPORTANT:  These definitions (TRILIBRARY, REDUCED, CDT_ONLY) must be    */
/*  made in the makefile or in triangle.c itself.  Putting these definitions */
/*  in this file will not create the desired effect.                         */
/*                                                                           */
/*                                                                           */
/*  The calling convention for triangulate() follows.                        */
/*                                                                           */
/*      void triangulate(triswitches, in, out, vorout)                       */
/*      char *triswitches;                                                   */
/*      TinIO *in;                                            */
/*      TinIO *out;                                           */
/*      TinIO *vorout;                                        */
/*                                                                           */
/*  `triswitches' is a string containing the command line switches you wish  */
/*  to invoke.  No initial dash is required.  Some suggestions:              */
/*                                                                           */
/*  - You'll probably find it convenient to use the `z' switch so that       */
/*    points (and other items) are numbered from zero.  This simplifies      */
/*    indexing, because the first item of any type always starts at index    */
/*    [0] of the corresponding array, whether that item's number is zero or  */
/*    one.                                                                   */
/*  - You'll probably want to use the `Q' (quiet) switch in your final code, */
/*    but you can take advantage of Triangle's printed output (including the */
/*    `V' switch) while debugging.                                           */
/*  - If you are not using the `q' or `a' switches, then the output points   */
/*    will be identical to the input points, except possibly for the         */
/*    boundary markers.  If you don't need the boundary markers, you should  */
/*    use the `N' (no nodes output) switch to save memory.  (If you do need  */
/*    boundary markers, but need to save memory, a good nasty trick is to    */
/*    set out->pointList equal to in->pointList before calling triangulate(),*/
/*    so that Triangle overwrites the input points with identical copies.)   */
/*  - The `I' (no iteration numbers) and `g' (.off file output) switches     */
/*    have no effect when Triangle is compiled with TRILIBRARY defined.      */
/*                                                                           */
/*  `in', `out', and `vorout' are descriptions of the input, the output,     */
/*  and the Voronoi output.  If the `v' (Voronoi output) switch is not used, */
/*  `vorout' may be NULL.  `in' and `out' may never be NULL.                 */
/*                                                                           */
/*  Certain fields of the input and output structures must be initialized,   */
/*  as described below.                                                      */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  The `triangulateIO' structure.                                           */
/*                                                                           */
/*  Used to pass data into and out of the triangulate() procedure.           */
/*                                                                           */
/*                                                                           */
/*  Arrays are used to store points, triangles, markers, and so forth.  In   */
/*  all cases, the first item in any array is stored starting at index [0].  */
/*  However, that item is item number `1' unless the `z' switch is used, in  */
/*  which case it is item number `0'.  Hence, you may find it easier to      */
/*  index points (and triangles in the neighbor list) if you use the `z'     */
/*  switch.  Unless, of course, you're calling Triangle from a Fortran       */
/*  program.                                                                 */
/*                                                                           */
/*  Description of fields (except the `numberof' fields, which are obvious): */
/*                                                                           */
/*  `pointList':  An array of point coordinates.  The first point's x        */
/*    coordinate is at index [0] and its y coordinate at index [1], followed */
/*    by the coordinates of the remaining points.  Each point occupies two   */
/*    tREALs.                                                                 */
/*  `pointAttrList':  An array of point attributes.  Each point's       */
/*    attributes occupy `numOfPointAttrs' tREALs.                     */
/*  `pointMarkList':  An array of point markers; one int per point.        */
/*                                                                           */
/*  `triList':  An array of triangle corners.  The first triangle's     */
/*    first corner is at index [0], followed by its other two corners in     */
/*    counterclockwise order, followed by any other nodes if the triangle    */
/*    represents a nonlinear element.  Each triangle occupies                */
/*    `numOfCorners' ints.                                                */
/*  `triAttrList':  An array of triangle attributes.  Each         */
/*    triangle's attributes occupy `numOfTriAttributes' tREALs.       */
/*  `triAtREAList':  An array of triangle area constraints; one tREAL per */
/*    triangle.  Input only.                                                 */
/*  `neighborList':  An array of triangle neighbors; three ints per          */
/*    triangle.  Output only.                                                */
/*                                                                           */
/*  `segmentList':  An array of segment endpoints.  The first segment's      */
/*    endpoints are at indices [0] and [1], followed by the remaining        */
/*    segments.  Two ints per segment.                                       */
/*  `segMarkList':  An array of segment markers; one int per segment.  */
/*                                                                           */
/*  `holeList':  An array of holes.  The first hole's x and y coordinates    */
/*    are at indices [0] and [1], followed by the remaining holes.  Two      */
/*    tREALs per hole.  Input only, although the pointer is copied to the     */
/*    output structure for your convenience.                                 */
/*                                                                           */
/*  `regionList':  An array of regional attributes and area constraints.     */
/*    The first constraint's x and y coordinates are at indices [0] and [1], */
/*    followed by the regional attribute and index [2], followed by the      */
/*    maximum area at index [3], followed by the remaining area constraints. */
/*    Four tREALs per area constraint.  Note that each regional attribute is  */
/*    used only if you select the `A' switch, and each area constraint is    */
/*    used only if you select the `a' switch (with no number following), but */
/*    omitting one of these switches does not change the memory layout.      */
/*    Input only, although the pointer is copied to the output structure for */
/*    your convenience.                                                      */
/*                                                                           */
/*  `edgeList':  An array of edge endpoints.  The first edge's endpoints are */
/*    at indices [0] and [1], followed by the remaining edges.  Two ints per */
/*    edge.  Output only.                                                    */
/*  `edgeMarkList':  An array of edge markers; one int per edge.  Output   */
/*    only.                                                                  */
/*  `normList':  An array of normal vectors, used for infinite rays in       */
/*    Voronoi diagrams.  The first normal vector's x and y magnitudes are    */
/*    at indices [0] and [1], followed by the remaining vectors.  For each   */
/*    finite edge in a Voronoi diagram, the normal vector written is the     */
/*    zero vector.  Two tREALs per edge.  Output only.                        */
/*                                                                           */
/*                                                                           */
/*  Any input fields that Triangle will examine must be initialized.         */
/*  Furthermore, for each output array that Triangle will write to, you      */
/*  must either provide space by setting the appropriate pointer to point    */
/*  to the space you want the data written to, or you must initialize the    */
/*  pointer to NULL, which tells Triangle to allocate space for the results. */
/*  The latter option is preferable, because Triangle always knows exactly   */
/*  how much space to allocate.  The former option is provided mainly for    */
/*  people who need to call Triangle from Fortran code, though it also makes */
/*  possible some nasty space-saving tricks, like writing the output to the  */
/*  same arrays as the input.                                                */
/*                                                                           */
/*  Triangle will not free() any input or output arrays, including those it  */
/*  allocates itself; that's up to you.                                      */
/*                                                                           */
/*  Here's a guide to help you decide which fields you must initialize       */
/*  before you call triangulate().                                           */
/*                                                                           */
/*  `in':                                                                    */
/*                                                                           */
/*    - `pointList' must always point to a list of points; `numOfPoints'  */
/*      and `numOfPointAttrs' must be properly set.                  */
/*      `pointMarkList' must either be set to NULL (in which case all      */
/*      markers default to zero), or must point to a list of markers.  If    */
/*      `numOfPointAttrs' is not zero, `pointAttrList' must     */
/*      point to a list of point attributes.                                 */
/*    - If the `r' switch is used, `triList' must point to a list of    */
/*      triangles, and `numOfTriangles', `numOfCorners', and           */
/*      `numOfTriAttributes' must be properly set.  If               */
/*      `numOfTriAttributes' is not zero, `triAttrList'    */
/*      must point to a list of triangle attributes.  If the `a' switch is   */
/*      used (with no number following), `triAtREAList' must point to a  */
/*      list of triangle area constraints.  `neighborList' may be ignored.   */
/*    - If the `p' switch is used, `segmentList' must point to a list of     */
/*      segments, `numOfSegments' must be properly set, and               */
/*      `segMarkList' must either be set to NULL (in which case all    */
/*      markers default to zero), or must point to a list of markers.        */
/*    - If the `p' switch is used without the `r' switch, then               */
/*      `numOfHoles' and `numOfRegions' must be properly set.  If      */
/*      `numOfHoles' is not zero, `holeList' must point to a list of      */
/*      holes.  If `numOfRegions' is not zero, `regionList' must point to */
/*      a list of region constraints.                                        */
/*    - If the `p' switch is used, `holeList', `numOfHoles',              */
/*      `regionList', and `numOfRegions' is copied to `out'.  (You can    */
/*      nonetheless get away with not initializing them if the `r' switch is */
/*      used.)                                                               */
/*    - `edgeList', `edgeMarkList', `normList', and `numberofedges' may be */
/*      ignored.                                                             */
/*                                                                           */
/*  `out':                                                                   */
/*                                                                           */
/*    - `pointList' must be initialized (NULL or pointing to memory) unless  */
/*      the `N' switch is used.  `pointMarkList' must be initialized       */
/*      unless the `N' or `B' switch is used.  If `N' is not used and        */
/*      `in->numOfPointAttrs' is not zero, `pointAttrList' must */
/*      be initialized.                                                      */
/*    - `triList' must be initialized unless the `E' switch is used.    */
/*      `neighborList' must be initialized if the `n' switch is used.  If    */
/*      the `E' switch is not used and (`in->numberofelementattributes' is   */
/*      not zero or the `A' switch is used), `elementattributelist' must be  */
/*      initialized.  `triAtREAList' may be ignored.                     */
/*    - `segmentList' must be initialized if the `p' or `c' switch is used,  */
/*      and the `P' switch is not used.  `segMarkList' must also be    */
/*      initialized under these circumstances unless the `B' switch is used. */
/*    - `edgeList' must be initialized if the `e' switch is used.            */
/*      `edgeMarkList' must be initialized if the `e' switch is used and   */
/*      the `B' switch is not.                                               */
/*    - `holeList', `regionList', `normList', and all scalars may be ignored.*/
/*                                                                           */
/*  `vorout' (only needed if `v' switch is used):                            */
/*                                                                           */
/*    - `pointList' must be initialized.  If `in->numOfPointAttrs'   */
/*      is not zero, `pointAttrList' must be initialized.               */
/*      `pointMarkList' may be ignored.                                    */
/*    - `edgeList' and `normList' must both be initialized.                  */
/*      `edgeMarkList' may be ignored.                                     */
/*    - Everything else may be ignored.                                      */
/*                                                                           */
/*  After a call to triangulate(), the valid fields of `out' and `vorout'    */
/*  will depend, in an obvious way, on the choice of switches used.  Note    */
/*  that when the `p' switch is used, the pointers `holeList' and            */
/*  `regionList' are copied from `in' to `out', but no new space is          */
/*  allocated; be careful that you don't free() the same array twice.  On    */
/*  the other hand, Triangle will never copy the `pointList' pointer (or any */
/*  others); new space is allocated for `out->pointList', or if the `N'      */
/*  switch is used, `out->pointList' remains uninitialized.                  */
/*                                                                           */
/*  All of the meaningful `numberof' fields will be properly set; for        */
/*  instance, `numberofedges' will represent the number of edges in the      */
/*  triangulation whether or not the edges were written.  If segments are    */
/*  not used, `numOfSegments' will indicate the number of boundary edges. */
/*                                                                           */
/*****************************************************************************/


struct triangulateIO {
	tREAL *pointList;        	/* In / out */
	float *pointAttrList; 		/* In / out */
	long *pointMarkList;		/* In / out */
	long   numOfPoints;			/* In / out */
	long   numOfPointAttrs;		/* In / out */

	long *triList; 				/* In / out */
	float *triAttrList;			/* In / out */
	float *triAtREAList; 		/* In only */
	long *neighborList;			/* Out only */
	long numOfTriangles;			/* In / out */
	long numOfCorners; 			/* In / out */
	long numOfTriAttrs; 			/* In / out */

	long *segmentList;		/* In / out */
	long *segMarkList;		/* In / out */
	long numOfSegments;		/* In / out */

	tREAL *holeList;			/* In / pointer to array copied out */
	long numOfHoles;		/* In / copied out */

	tREAL *regionList;		/* In / pointer to array copied out */
	long numOfRegions;		/* In / copied out */

	long *edgeList;    		/* Out only */
	long *edgeMarkList;	/* Not used with Voronoi diagram; out only */
	tREAL *normList;			/* Used only with Voronoi diagram; out only */
	long numOfEdges;		/* Out only */
};


typedef triangulateIO TinIO;


#define	NUM_BadFaceQueues 64

////////////////////////////////////////////////////////////////////

#ifdef TINDLL_EXPORTS
#define TINDLL_API __declspec(dllexport)
#else
#define TINDLL_API __declspec(dllimport)
#endif


TINDLL_API extern int plus1mod3[3];
TINDLL_API extern int minus1mod3[3];

class TINDLL_API CTINClass	{
public:
	/********* Primitives for m_triangles                                  *********/
	/* decode() converts a pointer to an oriented TRIANGLE.  The orientation is  */
	/*   extracted from the two least significant bits of the pointer.           */
	
	void decode(TRIANGLE *ptr, triEDGE &triEdge) 
	{
		triEdge.orient = (int) ((unsigned long) (ptr) & (unsigned long) 3l);  
		triEdge.tri = (TRIANGLE *) ((unsigned long) (ptr) ^ (unsigned long) triEdge.orient);
	};
	
	/* encode() compresses an oriented TRIANGLE into a single pointer.  It       */
	/*   relies on the assumption that all m_triangles are aligned to four-byte    */
	/*   boundaries, so the two least significant bits of triEdge.tri are zero.*/
	
	TRIANGLE *encode(triEDGE &triEdge) 
	{
		return (TRIANGLE *) ((unsigned long) triEdge.tri | (unsigned long) triEdge.orient);
	};
	
	/* The following shEDGE manipulation primitives are all described by Guibas    */
	/*   and Stolfi.  However, they use an edge-based data structure, whereas I  */
	/*   am using a TRIANGLE-based data structure.                               */
	
	/* SymmTriEdge() finds the abutting TRIANGLE, on the same edge.  Note that the       */
	/*   edge direction is necessarily reversed, because TRIANGLE/edge handles   */
	/*   are always directed counterclockwise around the TRIANGLE.               */
	
	void SymmTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		decode( triEdge1.tri->adjoin[triEdge1.orient], triEdge2 );
	};
	
	void SymmTriEdgeSelf(triEDGE &triEdge)  
	{
		decode(triEdge.tri->adjoin[triEdge.orient], triEdge);
	};
	
	/* NextTriEdge() finds the next shEDGE (counterclockwise) of a TRIANGLE.             */
	
	void NextTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		triEdge2.tri = triEdge1.tri;
		triEdge2.orient = plus1mod3[triEdge1.orient];
	};
	
	void NextTriEdgeSelf(triEDGE &triEdge)  
	{
		triEdge.orient = plus1mod3[triEdge.orient];
	};
	
	/* PrevTriEdge() finds the previous shEDGE (clockwise) of a TRIANGLE.                */
	void PrevTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		triEdge2.tri = triEdge1.tri;
		triEdge2.orient = minus1mod3[triEdge1.orient];
	};
	
	void PrevTriEdgeSelf(triEDGE &triEdge)
	{
		triEdge.orient = minus1mod3[triEdge.orient];
	};
	
	/* oNextSpinTriEdge() spins counterclockwise around a triPOINT; that is, it finds the next */
	/*   edge with the same origin in the counterclockwise direction.  This edge */
	/*   will be part of a different TRIANGLE.                                   */
	
	void oNextSpinTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		PrevTriEdge(triEdge1, triEdge2);   
		SymmTriEdgeSelf(triEdge2);
	};
	
	void oNextSpinTriEdgeSelf(triEDGE &triEdge) 
	{
		PrevTriEdgeSelf(triEdge);
		SymmTriEdgeSelf(triEdge);
	};

	/* oPrevSpinTriEdge() spins clockwise around a triPOINT; that is, it finds the next edge   */
	/*   with the same origin in the clockwise direction.  This edge will be     */
	/*   part of a different TRIANGLE.                                           */
	
	void oPrevSpinTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		SymmTriEdge(triEdge1, triEdge2);
		NextTriEdgeSelf(triEdge2);
	};
	
	void oPrevSpinTriEdgeSelf(triEDGE &triEdge)
	{
		SymmTriEdgeSelf(triEdge); 
		NextTriEdgeSelf(triEdge);
	};

	/* dNextSpinTriEdge() spins counterclockwise around a triPOINT; that is, it finds the next */
	/*   edge with the same destination in the counterclockwise direction.  This */
	/*   edge will be part of a different TRIANGLE.                              */
	
	void dNextSpinTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		SymmTriEdge(triEdge1, triEdge2);
		PrevTriEdgeSelf(triEdge2);
	};
	
	void dNextSpinTriEdgSelf(triEDGE &triEdge)
	{
		SymmTriEdgeSelf(triEdge);  
		PrevTriEdgeSelf(triEdge);
	};

	/* dPrevSpinTriEdge() spins clockwise around a triPOINT; that is, it finds the next shEDGE   */
	/*   with the same destination in the clockwise direction.  This edge will   */
	/*   be part of a different TRIANGLE.                                        */
	
	void dPrevSpinTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		NextTriEdge(triEdge1, triEdge2);
		SymmTriEdgeSelf(triEdge2);
	};

	void dPrevSpinTriEdgeSelf(triEDGE &triEdge)
	{
		NextTriEdgeSelf(triEdge); 
		SymmTriEdgeSelf(triEdge);
	};
	
	/* rnext() moves one edge counterclockwise about the adjacent TRIANGLE.      */
	/*   (It's best understood by reading Guibas and Stolfi.  It involves        */
	/*   changing triangles twice.)                                              */
	
	void rNextTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		SymmTriEdge(triEdge1, triEdge2); 
		NextTriEdgeSelf(triEdge2); 
		SymmTriEdgeSelf(triEdge2);
	};

	void rNextTriEdgeSelf(triEDGE &triEdge)
	{
		SymmTriEdgeSelf(triEdge); 
		NextTriEdgeSelf(triEdge);
		SymmTriEdgeSelf(triEdge);
	};

	/* rnext() moves one edge clockwise about the adjacent TRIANGLE.             */
	/*   (It's best understood by reading Guibas and Stolfi.  It involves        */
	/*   changing triangles twice.)                                              */
	
	void rPrevTriEdge(triEDGE &triEdge1, triEDGE &triEdge2) 
	{
		SymmTriEdge(triEdge1, triEdge2); 
		PrevTriEdgeSelf(triEdge2);  
		SymmTriEdgeSelf(triEdge2);
	};
	
	void rPrevTriEdgeSelf(triEDGE &triEdge)  
	{
		SymmTriEdgeSelf(triEdge);     
		PrevTriEdgeSelf(triEdge);
		SymmTriEdgeSelf(triEdge);
	};
	
	/* These primitives determine or set the origin, destination, or apex of a   */
	/* TRIANGLE.                                                                 */
	
	triPOINT *org(triEDGE &triEdge) 
	{                                               
		return triEdge.tri->vertex[ plus1mod3[triEdge.orient] ];
	};
	
	triPOINT *dest(triEDGE &triEdge)
	{                                            
		return triEdge.tri->vertex[minus1mod3[triEdge.orient]];
	};
	
	triPOINT *apex(triEDGE &triEdge)
	{                                      
		return triEdge.tri->vertex[triEdge.orient];
	};
	
	void SetOrg(triEDGE &triEdge, triPOINT *pt)
	{                                           
		triEdge.tri->vertex[plus1mod3[triEdge.orient]] = pt;

		// 定位加速
		if( pt && triEdge.tri != m_dummytri )	{
			SetPoint2Tri( pt, encode(triEdge) );
			OnTriangleUpdated( triEdge.tri );
		}
	};

	void SetDest(triEDGE &triEdge, triPOINT *pt)  
	{                                         
		triEdge.tri->vertex[minus1mod3[triEdge.orient]] = pt;

		// 定位加速
		if( pt && triEdge.tri != m_dummytri )	{
			SetPoint2Tri( pt, (TRIANGLE *) ((unsigned long) triEdge.tri | plus1mod3[triEdge.orient] ) );
			OnTriangleUpdated( triEdge.tri );
		};
	}
	
	void SetApex(triEDGE &triEdge, triPOINT *pt)  
	{
		triEdge.tri->vertex[triEdge.orient] =  pt;

		// 定位加速
		if( pt && triEdge.tri != m_dummytri )	{
			SetPoint2Tri( pt, (TRIANGLE *) ((unsigned long) triEdge.tri | minus1mod3[triEdge.orient] ) );
			OnTriangleUpdated( triEdge.tri );
		}
	};

	void SetVertices2Null(triEDGE &triEdge) 
	{                                 
		triEdge.tri->vertex[0] = (triPOINT *) NULL;  
		triEdge.tri->vertex[1] = (triPOINT *) NULL; 
		triEdge.tri->vertex[2] = (triPOINT *) NULL;
	};

	/* Bond two m_triangles together.                                              */
	void bond(triEDGE &triEdge1, triEDGE &triEdge2)  
	{
		(triEdge1).tri->adjoin[(triEdge1).orient] = encode(triEdge2);
		(triEdge2).tri->adjoin[(triEdge2).orient] = encode(triEdge1);
	};
	
	/* Dissolve a bond (from one side).  Note that the other TRIANGLE will still */
	/*   think it's connected to this TRIANGLE.  Usually, however, the other     */
	/*   TRIANGLE is being deleted entirely, or bonded to another TRIANGLE, so   */
	/*   it doesn't matter.                                                      */
	
	void dissolve(triEDGE &triEdge )
	{                                                   
		triEdge.tri->adjoin[triEdge.orient] = m_dummytri;
	};
	
	/* Copy a TRIANGLE/shEDGE handle.                                              */
	void TriEdgeCopy(triEDGE &triEdge1, triEDGE &triEdge2)
	{
		(triEdge2).tri = (triEdge1).tri;
		(triEdge2).orient = (triEdge1).orient;
	};
	
	/* Test for equality of TRIANGLE/shEDGE handles.                               */
	
	bool TriEdgeEqual(triEDGE &triEdge1, triEDGE &triEdge2)
	{
		return (((triEdge1).tri == (triEdge2).tri) && ((triEdge1).orient == (triEdge2).orient));
	};
	
	/* Primitives to infect or cure a TRIANGLE with the virus.  These rely on    */
	/*   the assumption that all shell edges are aligned to four-byte boundaries.*/
	
	void infect(triEDGE &triEdge )
	{
		triEdge.tri->sh[0] = (SHELLE *)((unsigned long) triEdge.tri->sh[0] | (unsigned long) 2l);
	};
	
	void uninfect(triEDGE &triEdge ) 
	{
		triEdge.tri->sh[0] = (SHELLE *) ((unsigned long) triEdge.tri->sh[0] & ~ (unsigned long) 2l);
	};
	
	/* Test a TRIANGLE for viral infection.                                      */
	
	bool infected(triEDGE &triEdge) 
	{
		return (((unsigned long) triEdge.tri->sh[0] & (unsigned long) 2l) != 0);
	};
	
	float ElemAttribute(triEDGE &triEdge , int attnum)
	{
		return triEdge.tri->attr;
	};
	
	void SetElemAttribute(triEDGE &triEdge, int attnum, float value)
	{
		triEdge.tri->attr = value;
	};
	
	/* Check or set a TRIANGLE's maximum area bound.                             */
	
	double AreaBound(triEDGE  &triEdge)	{ return triEdge.tri->area; };
	
	void SetAreaBound(triEDGE &triEdge, float value) { triEdge.tri->area = value; };
	
	/********* Primitives for shell edges                                *********/
	/*                                                                           */
	/*                                                                           */
	
	/* shDecode() converts a pointer to an oriented shell edge.  The orientation  */
	/*   is extracted from the least significant bit of the pointer.  The two    */
	/*   least significant bits (one for orientation, one for viral infection)   */
	/*   are masked out to produce the tREAL pointer.                             */
	
	void shDecode(SHELLE *sptr, shEDGE &edge) 
	{
		edge.shorient = (int) ((unsigned long) (sptr) & (unsigned long) 1l); 
		edge.sh = (SHELLE *)((unsigned long) (sptr) & ~ (unsigned long) 3l);
	};
	
	/* shEncode() compresses an oriented shell edge into a single pointer.  It    */
	/*   relies on the assumption that all shell edges are aligned to two-byte   */
	/*   boundaries, so the least significant bit of edge.sh is zero.          */
	
	SHELLE * shEncode(shEDGE &edge)  
	{
		return (SHELLE *) ((unsigned long) edge.sh | (unsigned long) edge.shorient);
	};
	
	/* symmShEdge() toggles the orientation of a shell shEDGE.                           */
	void SymmShEdge(shEDGE &edge1, shEDGE &edge2) 
	{
		(edge2).sh = (edge1).sh;
		(edge2).shorient = 1 - (edge1).shorient;
	};
	
	void SymmShEdgeSelf(shEDGE &edge) {edge.shorient = 1 - edge.shorient; };
	
	/* spivot() finds the other shell shEDGE (from the same segment) that shares   */
	/*   the same origin.                                                        */
	
	void AdjoinShEdge(shEDGE &edge1, shEDGE &edge2) 
	{                                            
		shDecode( (edge1).sh->adjoin[(edge1).shorient], edge2);
	};
	
	void AdjoinShEdgeSelf( shEDGE &edge )  { shDecode(edge.sh->adjoin[edge.shorient], edge); };
	
	// NextShEdge() finds the next shell edge (from the same segment) in sequence;  
	//  one whose origin is the input shell shEDGE's destination. 
	void NextShEdge(shEDGE &edge1, shEDGE &edge2)
	{
		shDecode((edge1).sh->adjoin[1 - (edge1).shorient], edge2);
	};
	
	void NextShEdgeSelf(shEDGE &edge) { shDecode(edge.sh->adjoin[1 - edge.shorient], edge); };
	
	// These primitives determine or set the origin or destination of a shell edge
	triPOINT *shOrg(shEDGE &edge)  { return edge.sh->vertex[edge.shorient]; };
	
	triPOINT *shDest(shEDGE &edge) { return edge.sh->vertex[1 - edge.shorient]; };
	
	void SetShOrg(shEDGE &edge, triPOINT *pointptr) { 
		edge.sh->vertex[edge.shorient] = pointptr; 
		OnShelleUpdated( edge.sh );
	};

	void SetShDest(shEDGE &edge, triPOINT *pointptr) { 
		edge.sh->vertex[1-edge.shorient] = pointptr; 
		OnShelleUpdated( edge.sh );
	};
	
	/* These primitives read or set a shell marker.  Shell markers are used to   */
	/*   hold user boundary information.                                         */
	
	long shMark(shEDGE &edge) { return edge.sh->marker; };
	
	void SetShellMark(shEDGE &edge, long value) { edge.sh->marker = value; };
	
	/* Bond two shell edges together.                                            */
	void shBond(shEDGE &edge1, shEDGE &edge2) 
	{
		(edge1).sh->adjoin[(edge1).shorient] = shEncode(edge2);
		(edge2).sh->adjoin[(edge2).shorient] = shEncode(edge1);
	};
	
	/* Dissolve a shell shEDGE bond (from one side).  Note that the other shell    */
	/*   shEDGE will still think it's connected to this shell shEDGE.                */
	
	void shDissolve(shEDGE &edge)
	{
		edge.sh->adjoin[edge.shorient] = (SHELLE *) m_dummysh;
	};
	
	/* Copy a shell shEDGE.                                                        */
	
	void shelleCopy(shEDGE &edge1, shEDGE &edge2)
	{
		(edge2).sh = (edge1).sh;
		(edge2).shorient = (edge1).shorient;
	};
	/* Test for equality of shell edges.                                         */
	
	bool shelleEqual(shEDGE &edge1, shEDGE &edge2) 
	{ 
		return (((edge1).sh == (edge2).sh) && ((edge1).shorient == (edge2).shorient));
	};
	
	/********* Primitives for interacting m_triangles and shell edges      *********/
	/*                                                                           */
	/*                                                                           */
	
	/* tspivot() finds a shell shEDGE abutting a TRIANGLE.                         */
	void ShEdgeOnTriEdge( triEDGE &triEdge, shEDGE &edge ) 
	{
		shDecode( triEdge.tri->sh[triEdge.orient], edge);
	};
	
	/* stpivot() finds a TRIANGLE abutting a shell shEDGE.  It requires that the   */
	/*   variable `ptr' of type `TRIANGLE' be defined.                           */
	
	void TriEdgeOnShEdge( shEDGE &edge, triEDGE &triEdge )
	{
		decode(edge.sh->tri[edge.shorient], triEdge);
	};

	/* Bond a TRIANGLE to a shell shEDGE.                                          */
	void tshBond(triEDGE &triEdge, shEDGE &edge)
	{
		triEdge.tri->sh[triEdge.orient] = shEncode(edge);
		edge.sh->tri[edge.shorient] = encode(triEdge);
	};

	/* Dissolve a bond (from the TRIANGLE side).                                 */
	
	void tshDissolve(triEDGE &triEdge)
	{
		triEdge.tri->sh[triEdge.orient] =  m_dummysh;
	};

	/* Dissolve a bond (from the shell shEDGE side).                               */
	void shtDissolve(shEDGE &edge)
	{
		edge.sh->tri[edge.shorient] =  m_dummytri;
	};
	
	/********* Primitives for points                                     *********/
	/*                                                                           */
	/*                                                                           */
	long PointMark(triPOINT *pt)	{  return pt->marker; };
	
	void SetPointMark(triPOINT * pt,long value) { pt->marker = value;};
	
	TRIANGLE *Point2Tri(triPOINT *pt)  { return pt->tri; };
	
	void SetPoint2Tri(triPOINT * pt,TRIANGLE * value) { pt->tri = value; };
	
	int PointDupCount(triPOINT * pt) { 
		if( pt->dup.freeCount < 0 ) return -pt->dup.freeCount;
		else return 0;
	}

	triPOINT *PointDup(triPOINT * pt) { 
		if( pt->dup.freeCount < 0 ) return NULL;
		else return pt->dup.pt;
	}

	void SetPointDup(triPOINT *pt, triPOINT *dupPt) { 
		if( dupPt == NULL )
			pt->dup.freeCount = -1L;
		else	{
			pt->dup.pt = dupPt; 
			dupPt->dup.freeCount--;
		}
	};

	int DecreasePointDupCount(triPOINT * pt) { 
		if( pt->dup.freeCount < 0  ) 
			pt->dup.freeCount++; 
		return -pt->dup.freeCount; 
	};
	
	/********* Mesh manipulation primitives end here                     *********/
	
protected:
	/* Variables used to allocate memory for triangles, shell edges, points,     */
	/*   viri (triangles being eaten), bad (encroached) segments, bad (skinny    */
	/*   or too large) triangles, and splay tree nodes.                          */
	
	CMemoryPool <TRIANGLE, TRIANGLE &>		m_triangles;
	CMemoryPool <SHELLE, SHELLE &>			m_shelles;
	CMemoryPool <triPOINT, triPOINT & >		m_points;
	CMemoryPool <TRIANGLE *, TRIANGLE *>	m_viri;
	CMemoryPool <shEDGE, shEDGE &>			m_badSegments;
	CMemoryPool <badFACE, badFACE &>		m_badTriangles;
	CMemoryPool <splayNODE,splayNODE &>		m_splayNodes;
	
	/* Variables that maintain the bad TRIANGLE queues.  The tails are pointers  */
	/*   to the pointers that have to be filled in to enqueue an item.           */

	badFACE *queuefront[NUM_BadFaceQueues];
	badFACE **queuetail[NUM_BadFaceQueues];
	
	tREAL m_xmin, m_xmax, m_ymin, m_ymax;                    /* x and y bounds. */
	tREAL m_xminextreme;        /* Nonexistent x value used as a flag in sweepline. */
	
	int inelements;                                /* Number of input triangles. */
	int insegments;                                 /* Number of input segments. */
	int holes;                                         /* Number of input holes. */
	int regions;                                     /* Number of input regions. */
	long edges;                                       /* Number of output edges. */
	int mesh_dim;                                  /* Dimension (ought to be 2). */
	int nextras;                              /* Number of attributes per triPOINT. */
	int eextras;                           /* Number of attributes per TRIANGLE. */
	long hullsize;                            /* Number of edges of convex hull. */
	int triwords;                                   /* Total words per TRIANGLE. */
	int shwords;                                  /* Total words per shell shEDGE. */
	int readnodefile;                             /* Has a .node file been read? */
	long samples;                /* Number of random samples for triPOINT location. */

	bool m_checksegments;           /* Are there segments in the triangulation yet? */
	
	/* Switches for the triangulator.                                            */
	/*   poly: -p switch.  refine: -r switch.                                    */
	/*   quality: -q switch.                                                     */
	/*     minangle: minimum angle bound, specified after -q switch.             */
	/*     goodangle: cosine squared of minangle.                                */
	/*   vararea: -a switch without number.                                      */
	/*   fixedarea: -a switch with number.                                       */
	/*     maxarea: maximum area bound, specified after -a switch.               */
	/*   regionattrib: -A switch.  convex: -c switch.                            */
	/*   firstnumber: inverse of -z switch.  All items are numbered starting     */
	/*     from firstnumber.                                                     */
	/*   edgesout: -e switch.  voronoi: -v switch.                               */
	/*   neighbors: -n switch.  geomview: -g switch.                             */
	/*   nobound: -B switch.  nopolywritten: -P switch.                          */
	/*   nonodewritten: -N switch.  noelewritten: -E switch.                     */
	/*   noiterationnum: -I switch.  noholes: -O switch.                         */
	/*   noexact: -X switch.                                                     */
	/*   order: element order, specified after -o switch.                        */
	/*   nobisect: count of how often -Y switch is selected.                     */
	/*   steiner: maximum number of Steiner points, specified after -S switch.   */
	/*     steinerleft: number of Steiner points not yet used.                   */
	/*   incremental: -i switch.  sweepline: -F switch.                          */
	/*   dwyer: inverse of -l switch.                                            */
	/*   splitseg: -s switch.                                                    */
	/*   docheck: -C switch.                                                     */
	/*   quiet: -Q switch.  verbose: count of how often -V switch is selected.   */
	/*   useshelles: -p, -r, -q, or -c switch; determines whether shell edges    */
	/*     are used at all.                                                      */
	/*                                                                           */
	/* Read the instructions to find out the meaning of these switches.          */
	
	bool m_poly, refine, quality, vararea, fixedarea, regionattrib, m_convex;
	bool edgesout, voronoi, neighbors, geomview;
	bool nobound, nopolywritten, nonodewritten, noelewritten, noiterationnum;
	bool noholes, noexact;
	bool incremental, sweepline, dwyer;
	bool m_splitSeg;
	bool docheck;
	bool quiet ;
	bool m_useShelles;
	bool nobisect;

	int  m_verbose;
	int  steiner, steinerleft;
	
	int  m_firstnumber;
	int  m_order;
	tREAL m_minangle, m_goodangle;
	tREAL m_maxarea;
	
	
	/* Triangular bounding box points.                                           */
	
	triPOINT *m_infpoint1, *m_infpoint2, *m_infpoint3;
	
	/* Pointer to the `TRIANGLE' that occupies all of "outer space".             */
	
	TRIANGLE *m_dummytri;
	TRIANGLE *m_dummytribase;      /* Keep base address so we can free() it later. */
	
	/* Pointer to the omnipresent shell shEDGE.  Referenced by any TRIANGLE or     */
	/*   shell shEDGE that isn't tREALly connected to a shell shEDGE at that          */
	/*   location.                                                               */
	
	SHELLE *m_dummysh;
	SHELLE *m_dummyshbase;         /* Keep base address so we can free() it later. */
	
	/* Pointer to a recently visited TRIANGLE.  Improves triPOINT location if       */
	/*   proximate points are inserted sequentially.                             */
	triEDGE m_recentTri;
	
protected:
	tREAL splitter;       /* Used to split tREAL factors for exact multiplication. */
	tREAL epsilon;                             /* Floating-triPOINT machine epsilon. */
	tREAL resulterrbound;
	tREAL ccwerrboundA, ccwerrboundB, ccwerrboundC;
	tREAL iccerrboundA, iccerrboundB, iccerrboundC;
	
	long incirclecount;                   /* Number of incircle tests performed. */
	long counterclockcount;       /* Number of counterclockwise tests performed. */
	long hyperbolacount;        /* Number of right-of-hyperbola tests performed. */
	long circumcentercount;    /* Number of circumcenter calculations performed. */
	long circletopcount;         /* Number of circle top calculations performed. */
	
protected:	
	/*****************************************************************************/
	/*  Mesh manipulation primitives.  Each TRIANGLE contains three pointers to  */
	/*  other triangles, with orientations.  Each pointer points not to the      */
	/*  first byte of a TRIANGLE, but to one of the first three bytes of a       */
	/*  TRIANGLE.  It is necessary to extract both the TRIANGLE itself and the   */
	/*  orientation.  To save memory, I keep both pieces of information in one   */
	/*  pointer.  To make this possible, I assume that all triangles are aligned */
	/*  to four-byte boundaries.  The `decode' routine below decodes a pointer,  */
	/*  extracting an orientation (in the range 0 to 2) and a pointer to the     */
	/*  beginning of a TRIANGLE.  The `encode' routine compresses a pointer to a */
	/*  TRIANGLE and an orientation into a single pointer.  My assumptions that  */
	/*  triangles are four-byte-aligned and that the `unsigned long' type is     */
	/*  long enough to hold a pointer are two of the few kludges in this program.*/
	/*                                                                           */
	/*  Shell edges are manipulated similarly.  A pointer to a shell shEDGE        */
	/*  carries both an address and an orientation in the range 0 to 1.          */
	/*                                                                           */
	/*  The other primitives take an oriented TRIANGLE or oriented shell shEDGE,   */
	/*  and return an oriented TRIANGLE or oriented shell shEDGE or triPOINT; or they */
	/*  change the connections in the data structure.                            */
	/*                                                                           */
	/*****************************************************************************/
	/* prototypes */
	void alternateaxes(triPOINT **sortarray,int arraysize,int axis);
	
	void badsegmentdealloc(shEDGE *dyingseg);
	shEDGE *badsegmenttraverse();
	
	void conformingedge(triPOINT * endpoint1,triPOINT * endpoint2,int newmark);
	
	void boundingbox();
	
	
	void checkmesh();
	void checkdelaunay();
	void Check4DeadEvent(triEDGE*checktri,sweepEVENT **freeevents,sweepEVENT **eventheap,long *heapsize);
	
	int checkedge4encroach(shEDGE *testedge);
	
	tREAL circletop(triPOINT *pa,triPOINT *pb,triPOINT *pc,tREAL ccwabc);
	
	void CreateEventHeap(sweepEVENT ***eventheap,sweepEVENT **events,sweepEVENT **freeevents);
	
	long delaunay();
	
	bool delaunayfixup(triEDGE*fixuptri,int  leftside);
	
	badFACE *dequeuebadtri();
	
	long divconqdelaunay();
	
	void divconqrecurse(triPOINT **sortarray,int vertices,int axis,triEDGE*farleft,triEDGE*farright);
	void dummyinit(int trianglewords,int shellewords);
	
	void enforcequality();
	void enqueuebadtri(triEDGE *instri,tREAL angle,triPOINT *insapex,triPOINT *insorg,triPOINT *insdest);
	
	tREAL estimate(int elen,tREAL *e);
	void eventheapdelete(sweepEVENT **heap,int heapsize,int eventnum);
	void eventHeapInsert(sweepEVENT **heap,int heapsize,sweepEVENT *newevent);
	void eventHeapify(sweepEVENT **heap,int heapsize,int eventnum);
	
	void exactinit();
	
	int fast_expansion_sum_zeroelim(int elen,tREAL *e,int flen,tREAL *f,tREAL *h);
	
	enum circumcenterresult findcircumcenter(triPOINT * torg,triPOINT * tdest,triPOINT * tapex,tREAL * circumcenter,tREAL *xi,tREAL *eta);
	
	void constrainededge(triEDGE*starttri,triPOINT *endpoint2,int newmark);
	enum finddirectionresult finddirection(triEDGE*searchtri,triPOINT *endpoint);
	
	void flip(triEDGE*flipedge);
	int formskeleton(long *segmentList,long *segMarkList,int numOfSegments);
	
	triPOINT *GetPoint(int number);
	
	void highorder();
	
	tREAL incircle(triPOINT * pa,triPOINT * pb,triPOINT * pc,triPOINT * pd);
	tREAL incircleadapt(triPOINT * pa,triPOINT * pb,triPOINT * pc,triPOINT * pd,tREAL permanent);
	long incrementaldelaunay();
	void initializepointpool();
	void InitializeTriSegPools();

	////////////////////////////////////////////
	// 如果没有给定三角形有可能引起缓慢的三角形定位，和三角形缓冲池遍历
	enum insertsiteresult InsertSite(triPOINT * insertpoint,triEDGE*searchtri,
		shEDGE *splitedge,int segmentflaws,int triflaws);

	void DeletesSite(triEDGE*deltri);
	
	bool insertsegment(triPOINT *endpoint1,triPOINT * endpoint2,int newmark);
	
	void internalerror();
	
	enum locateresult locate( double x, double y, triEDGE*searchtri);
	
	void makeshelle(shEDGE *newedge);
	void markhull();
	void mergehulls(triEDGE*farleft,triEDGE*innerleft,triEDGE*innerright,triEDGE*farright,int axis);
	
	void numbernodes();
	void parsecommandline(int argc, char **argv);
	
	void pointmedian(tREAL **sortarray,int arraysize,int median,int axis);
	void pointsort(triPOINT **sortarray,int arraysize);
	
	void precisionerror();
	enum locateresult PreciseLocate( double x, double y, triEDGE*searchtri);
	void printshelle(shEDGE *s);
	
	void regionplague(tREAL attribute,tREAL area);
	long removeghosts(triEDGE*startghost);
	int rightofhyperbola(triEDGE*fronttri,triPOINT *newsite);
	void quality_statistics();
	
	int reconstruct(long *triList,float *triangleattriblist,float *triAtREAList,int elements,
		int corners,int attribs,long *segmentList,long *segMarkList,int numOfSegments);
	
	long removebox();
	void repairencs(int flaws);
	
	int scale_expansion_zeroelim(int elen,tREAL *e,tREAL b,tREAL *h);
	int scoutsegment(triEDGE*searchtri,triPOINT *endpoint2,int newmark);
	void segmentintersection(triEDGE*splittri,shEDGE * splitshelle,triPOINT *endpoint2);
	
	splayNODE *circletopinsert(splayNODE *splayroot,triEDGE*newkey,
		triPOINT *pa,triPOINT *pb,triPOINT *pc,tREAL topy);
	
	void shelleDealloc(SHELLE *dyingshelle);
	
	void insertshelle(triEDGE*tri,int shellemark);
	
	splayNODE *splay(splayNODE *splaytreem,triPOINT *searchpoint,triEDGE*searchtri);
	splayNODE *splayinsert(splayNODE *splayroot,triEDGE*newkey,triPOINT *searchpoint);
	void splittriangle(badFACE *badtri);
	
	void statistics();
	long SweepLineDelaunay();
	
	void triangulatepolygon(triEDGE*firstedge,triEDGE*lastedge,int edgecount,int doflip,int triflaws);
	
	void tallyfaces();
	
	void tallyencs();
	void testtriangle(triEDGE*testtri);
	
	void triangledeinit();
	void triangleinit();
	void transfernodes(tREAL *pointList,float *pointattriblist,long *pointMarkList,int numOfPoints,int numberofpointattribs);
	void triangleDealloc(TRIANGLE *dyingtriangle);
	void printtriangle(triEDGE *t);
	
	void maketriangle(triEDGE *newtriedge);
	
	/////////////////////////////////////////////////////
	void writenodes(tREAL **pointList,float **pointattriblist,long **pointMarkList);
	void writepoly(long **segmentList,long **segMarkList);
	void writeelements(long **triList,float **triangleattriblist);
	void writeneighbors(long **neighborList);
	void writeedges(long **edgeList,long **edgeMarkList);
	void writevoronoi(tREAL **vpointList,float **vpointattriblist,long **vpointMarkList,
		long ** vedgeList,long **vedgeMarkList, tREAL **vnormList);
	
	////////////////////////////////////////////////////////
	tREAL counterclockwiseadapt(triPOINT *pa,triPOINT *pb,triPOINT *pc,tREAL detsum);
	tREAL counterclockwise(triPOINT * pa,triPOINT * pb,triPOINT * pc);
	
	splayNODE *frontlocate(splayNODE *splayroot,triEDGE*bottommost,
		triPOINT *searchpoint,triEDGE*searchtri,long *farright);
	
	void InfectHull();
	void Plague();

protected:

	// find direction 
	
	// 确定起点三角形朝向
	enum finddirectionresult FindDirection( triPOINT *p0, triPOINT *p1, triEDGE &searchTri, enum locateresult intersect );

	// 确定起点三角形及其朝向
	enum finddirectionresult FindDirection(	triPOINT p[2],	triEDGE	searchTri[2], enum locateresult intersect[2], bool *bReversed  );

	// 根据进入边确定下一三角形
	enum finddirectionresult FindDirection( enum finddirectionresult enterDir, triEDGE*searchtri, triPOINT *startpoint, triPOINT *endpoint);

	bool IsEndReached( triPOINT p[2], triEDGE searchTri[2], enum finddirectionresult collinear );

	bool DeScoutSegment(triEDGE*searchtri, triPOINT *endpoint2);

	bool IsInverted(triEDGE &tri);
	bool NeedFlip(triEDGE &flipedge );
	void UnConstrainedEdge(triEDGE &starttri);

	int  DelaunayFixupForRemoveSegment(triEDGE &fixuptri, triPOINT *checkpoint, bool leftSide );

protected:

	double AttribueInterpolate( triEDGE &searchtri, double x, double y );
	profilePOINT EdgeIntersection(triPOINT *p0, triPOINT *p1, triPOINT *p2, triPOINT *p3); 

	bool IsIntersect( triPOINT *p0, triPOINT *p1,  triPOINT *p2, triPOINT *p3 );

public:	// for debug
	triEDGE m_fixuptri;

public:	// old function for input/construct/output, key for black box of everything

	void triangulate(char *triswitches,TinIO *in,TinIO *out,TinIO *vorout);

	///////////////////////// new functions & members //////////////////////////////////////
private:
	bool	m_segmentIntersectionEnabled;

	bool	m_selectedChanged;
	tREAL	m_xMinSel, m_yMinSel, m_xMaxSel, m_yMaxSel;

	long	m_maxNumberOfSelectedTriangles;
	long	m_numberOfSelectedTriangles;
	TRIANGLE  **m_selectedTriangles;

public:
	void MakePointMap();

	bool PreInsertSegment(triEDGE &searchtri, triPOINT *endpoint2 );

	void EnableIntersection( bool enable=true )	{ m_segmentIntersectionEnabled = enable; };
	TRIANGLE  **SelectTriangles(long *numberOfSelectedTriangles, tREAL m_xMinSel=0, tREAL m_yMinSel=0,tREAL m_xMaxSel=0,tREAL m_yMaxSel=0);

	void GetRange( tREAL *xmin, tREAL *ymin, tREAL *xmax, tREAL *ymax)
	{ 
		*xmin = m_xmin;	*ymin = m_ymin;	*xmax = m_xmax;	*ymax = m_ymax;
	}


public:
	locateresult LocatePoint( double x, double y, triEDGE &searchtri );

	// searchPoint must be a vertex of tin
	bool LocatePoint( triPOINT *searchPoint, triEDGE &searchtri );

	// find triangle, whose point is endpoint
	bool FindTriEdge( triEDGE*searchtri, triPOINT *endpoint);

	double InterpolateAttribute( double x, double y );


public:
	
	long GetNumberOfPoints()	{ return m_points.GetNumberOfItems(); };
	long GetNumberOfTriangles()	{ return m_triangles.GetNumberOfItems(); };
	
	TRIANGLE *GetDummyTri()	{ return m_dummytri; };
	SHELLE *GetDummySh()	{ return m_dummysh; };

	void PointTraversalInit() { m_points.TraversalInit(); };
	triPOINT *PointTraverse();

	void ShelleTraversalInit() { m_shelles.TraversalInit(); };
	SHELLE *ShelleTraverse();

	void TriangleTraversalInit() { m_triangles.TraversalInit(); };
	TRIANGLE *TriangleTraverse();

protected:	// point handle

	triPOINT *PointAlloc();
	void PointDealloc(triPOINT *dyingpoint);

	//
	// virtual : callback function
	//
	// automatic free extAttr
	virtual void OnFreePointExtraAttr( void *dyingAttr ){};
	virtual void OnFreeTriExtraAttr( void *dyingAttr ){};
	virtual void OnFreeEdgeExtraAttr( void *dyingAttr ){};

	// automatic alloc extAttr
	virtual void OnAllocPointExtraAttr( triPOINT *pt ){};
	virtual void OnAllocTriExtraAttr( TRIANGLE *tri ){};
	virtual void OnAllocEdgeExtraAttr( SHELLE *sh ){};

	//
	// update triangle
	//
	virtual void OnTriangleUpdated( TRIANGLE *tri ){};

	//
	// update shelle
	//
	virtual void OnShelleUpdated( SHELLE *sh ){};	
public:

	CTINClass(char *triswitches =  "pczAenQY" );
	virtual ~CTINClass();
	

	/////////////////// for fast construct ////////////
	void BeginAddPoints();
	triPOINT *AddPoint( double x, double y, float attr, long marker=0, void *extAttr = NULL);
	void EndAddPoints();

	// 是否加上边框? 边框点的属性值为_DUMMY_POINT_ATTR
	bool FastConstruct( bool bAddBoundBox = false );

	//////////////////// for purely increamently construct //////////////////////////
	// if no point added and fastconstruct called before
	void BeginInsertPoints( double xmin, double ymin, double xmax, double ymax );

	// insert point can not be outside of the bounding box
	triPOINT *InsertPoint(double x, double y, float attr, long marker=0, void *extAttr = NULL);
	bool RemovePoint(triPOINT *point, bool bEnforced = false );

	// remove the bounding box
	void EndInsertPoints();

    //Add by cym.查找以thePoint为节点的所有三角形。
    bool GetSurroundTrangle(triPOINT *thePoint, TRIANGLE **SurroundTri, long *pSum);
    //End by cym.

	// 在给定的三角形内插入点，不会引起遍历 
	triPOINT *InsertInTriangle(triEDGE *searchTri, double x, double y, float attr, long marker=0, void *extAttr = NULL);


	//////////////////////////////////////
	bool InsertSegment(triPOINT *end1, triPOINT *end2, int boundmarker = 0);
	void InsertSegments(triPOINT **ends, int numberOfSegments, int boundmarker = 0);
	void RemoveSegment(triPOINT *end1, triPOINT *end2);

	bool RemoveShelle(triEDGE*tri);

	void MarkHull();

	///////////////////////////////////////

	/////////////////////////////////////
	void EnableConvex( bool state = true )	{ m_convex = state;	};
	void CarveHoles(tREAL *holeList = NULL, int holes = 0, tREAL *regionList = NULL, int regions = 0 );

	void RemoveVirtualBoundary( float attrTh );

	// 
	// do not free return pointer
	// 如果，bShelleOnly为真，只求与相交约束边的交点，不求线段端点在三角形中的内插值和与其它边的交点
	//	相当于PreSegment()的断面版本
	//
	profilePOINT *Profile( double x0, double y0, double x1, double y1, int *n, bool bShelleOnly = false );
	profilePOINT *SlowProfile( double x0, double y0, double x1, double y1, int *n, bool bShelleOnly = false  );

	bool GetAttrRange( double x, double y, float *minAttr, float *maxAttr, float *minDist );
	bool GetAttrRange( double x0, double y0, double x1, double y1, float *minAttr, float *maxAttr);

	/////////////////////
	void SaveTIN(const char *fileName, double xOff, double yOff);
	void ImportTIN( const char *fileName );  //add by cym.在创建CTINClass类时，通过输入SaveTIN()函数保存的"*.tin"数据，可得到CTINClass类赋值后的数据。

	void SaveTIN_Data( const char *fileName );  //add by JWS
	void ImportTIN_Data( const char *fileName );//add by JWS

	/////////////////////////////////////////////////////////////////////
	// don't free the memory returned, it is static, max : 256
	// don't call these functions again when the return data are still be used
	triPOINT **GetNeighborPoints(triPOINT *thePoint, int *pSum);

	// with symmetric apex
	triPOINT **GetNeighborPoints_ext(triPOINT *thePoint, int *pSum);

	// don't call the function when the return data are still be used
	//////////////// 取约束边
	SHELLE **GetAdjacentShelles( triPOINT *thePoint, int *shelleSum );

	void GetNextShEdge( triEDGE triEdge, shEDGE &shNext );
	void GetPrevShEdge( triEDGE triEdge, shEDGE &shPrev );

	void GetNextShEdge( shEDGE &shEdge, shEDGE &shNext )	
	{
		triEDGE triEdge;
		TriEdgeOnShEdge( shEdge, triEdge );
		GetNextShEdge( triEdge, shNext );
	};

	void GetPrevShEdge( shEDGE &shEdge, shEDGE &shPrev )
	{
		triEDGE triEdge;
		TriEdgeOnShEdge( shEdge, triEdge );
		GetPrevShEdge( triEdge, shPrev );
	}

	triEDGE SelectEdge( double x, double y, float r, bool bShelle );
};


#endif