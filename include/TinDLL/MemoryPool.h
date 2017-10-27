
#if !defined(AFX_MEMORYPOOL_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_)
#define AFX_MEMORYPOOL_H__5ABA2884_CE44_11D3_B875_0050BAAF35F4__INCLUDED_

/* The following obscenity seems to be necessary to ensure that this program */
/* will port to Dec Alphas running OSF/1, because their stdio.h file commits */
/* the unpardonable sin of including stdlib.h.  Hence, malloc(), free(), and */
/* exit() may or may not already be defined at this point.  I declare these  */
/* functions explicitly because some non-ANSI C compilers lack stdlib.h.     */


/* Labels that signify whether a record consists primarily of pointers or of */
/*   floating-point words.  Used to make decisions about data alignment.     */

enum wordtype {POINTER, FLOATINGPOINT};



/* A type used to allocate memory.  m_firstBlock is the first block of m_numberOfItems.  */
/*   m_nowBlock is the block from which m_numberOfItems are currently being allocated.   */
/*   m_nextItem points to the next slab of free memory for an item.            */
/*   m_deadItemStack is the head of a linked list (stack) of deallocated m_numberOfItems */
/*   that can be recycled.  m_unallocatedItems is the number of m_numberOfItems that     */
/*   remain to be allocated from m_nowBlock.                                   */
/*                                                                           */
/* Traversal is the process of walking through the entire list of m_numberOfItems, and */
/*   is separate from allocation.  Note that a traversal will visit m_numberOfItems on */
/*   the "m_deadItemStack" stack as well as live m_numberOfItems.  m_pathBlock points to   */
/*   the block currently being Traversed.  m_pathItem points to the next item  */
/*   to be Traversed.  m_pathItemsLeft is the number of m_numberOfItems that remain to   */
/*   be Traversed in m_pathBlock.                                              */
/*                                                                           */
/* itemwordtype is set to POINTER or FLOATINGPOINT, and is used to suggest   */
/*   what sort of word the record is primarily made up of.  alignbytes       */
/*   determines how new records should be aligned in memory.  itembytes and  */
/*   itemwords are the length of a record in bytes (after rounding up) and   */
/*   words.  m_numberOfItemsPerBlock is the number of m_numberOfItems allocated at once in a     */
/*   single block.  m_numberOfItems is the number of currently allocated m_numberOfItems.        */
/*   m_maxNumberOfItems is the maximum number of m_numberOfItems that have been allocated at     */
/*   once; it is the current number of m_numberOfItems plus the number of records kept */
/*   on m_deadItemStack.*/
                    
/*       
#ifdef TINDLL_EXPORTS
#define TINDLL_API __declspec(dllexport)
#else
#define TINDLL_API __declspec(dllimport)
#endif	//TINDLL_API
*/

template<class TYPE, class ARG_TYPE>
class  CMemoryPool {

	struct dataBLOCK	{
		TYPE *items;
	};

private:

	int m_blockSum, m_maxBlockSum;

	dataBLOCK *m_blocks;
	dataBLOCK *m_pathBlock;	// for Traverse()
	dataBLOCK *m_nowBlock;	// for PoolAlloc()

	//	void **m_firstBlock, **m_nowBlock;

	TYPE *m_nextItem;		// next alloc item
	TYPE *m_deadItemStack;

	TYPE *m_pathItem;

	int m_numberOfItemsPerBlock;
	long m_numberOfItems, m_maxNumberOfItems;
	int m_unallocatedItems;
	int m_pathItemsLeft;

private:
	void IncreaseABlock();

public:
	CMemoryPool();
	~CMemoryPool()	{	PoolDeinit();	};

	void PoolInit(int itemcount);
	void PoolDeinit();

	TYPE *PoolAlloc();
	void PoolDealloc(TYPE *dyingitem);
	void PoolRestart();
	void TraversalInit();
	TYPE *Traverse();

	void GotoFirstBlock(){	m_pathBlock = m_blocks;};
	void GotoNextBlock() { m_pathBlock++; };

	TYPE *GetItem( int number )	;
	TYPE *GotoItem( int number );
	TYPE *GetPathBlockItem( int number ) { return m_pathBlock->items + number; };

	int GetItemsPerBlock()	{return m_numberOfItemsPerBlock;};

	long GetNumberOfItems()	{return m_numberOfItems;};
	long GetMaxNumberOfItems()	{return m_maxNumberOfItems;};

	void NullDeadItemStack()	{ m_deadItemStack = NULL;};

}	;


template<class TYPE, class ARG_TYPE>
CMemoryPool<TYPE, ARG_TYPE>::CMemoryPool()
{
	m_maxNumberOfItems = m_numberOfItems = 0l;
	m_blockSum = 0;
	m_maxBlockSum = 0;
	m_blocks = NULL;
	m_unallocatedItems = 0;
	m_deadItemStack = NULL;

	m_numberOfItemsPerBlock = 4096;

//	PoolInit();
}


/********* Memory management routines begin here                     *********/
/**                                                                         **/
/**                                                                         **/

/*****************************************************************************/
/*                                                                           */
/*  PoolInit()   Initialize a pool of memory for allocation of m_numberOfItems.        */
/*                                                                           */
/*  This routine initializes the machinery for allocating m_numberOfItems.  A `pool'   */
/*  is created whose records have size at least `bytecount'.  Items will be  */
/*  allocated in `itemcount'-item blocks.  Each item is assumed to be a      */
/*  collection of words, and either pointers or floating-point values are    */
/*  assumed to be the "primary" word type.  (The "primary" word type is used */
/*  to determine alignment of m_numberOfItems.)  If `alignment' isn't zero, all m_numberOfItems  */
/*  will be `alignment'-byte aligned in memory.  `alignment' must be either  */
/*  a multiple or a factor of the primary word size; powers of two are safe. */
/*  `alignment' is normally used to create a few unused bits at the bottom   */
/*  of each item's pointer, in which information may be stored.              */
/*                                                                           */
/*  Don't change this routine unless you understand it.                      */
/*                                                                           */
/*****************************************************************************/
template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::PoolInit(int itemcount)
{
	if( NULL != m_blocks )
		return ;

	m_numberOfItemsPerBlock = itemcount;
	
	/* Allocate a block of m_numberOfItems.  Space for `m_numberOfItemsPerBlock' m_numberOfItems and one    */
	/*   pointer (to point to the next block) are allocated, as well as space */
	/*   to ensure alignment of the m_numberOfItems.                                    */
	
	IncreaseABlock();

	PoolRestart();
}


/*****************************************************************************/
/*                                                                           */
/*  PoolDeinit()   Free to the operating system all memory taken by a pool.  */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::PoolDeinit()
{
	for(int i=0; i<m_blockSum; i++) 
		delete m_blocks[i].items;
	
	if( m_blocks ) 
		delete m_blocks;

	m_maxNumberOfItems = m_numberOfItems = 0l;
	m_blockSum = 0;
	m_maxBlockSum = 0;
	m_blocks = NULL;
	m_unallocatedItems = 0;
	m_deadItemStack = NULL;
}

template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::IncreaseABlock()
{
	dataBLOCK *blocks;

	if( m_blockSum == m_maxBlockSum ) {
		m_maxBlockSum += 1024;
		blocks = new dataBLOCK[ m_maxBlockSum ];
		memcpy( blocks, m_blocks, m_blockSum*sizeof( dataBLOCK ) );
		memset( blocks+m_blockSum, 0, 1024*sizeof( dataBLOCK ) );
		delete m_blocks;
		m_blocks = blocks;

		m_nowBlock = m_blocks + m_blockSum;
	}
	else	{
		m_nowBlock++;
	}
		
	if( m_nowBlock->items == NULL ) {
		m_nowBlock->items = new TYPE[ m_numberOfItemsPerBlock ];
		m_blockSum++;
	}

	if( m_nowBlock->items == NULL) {
		printf("Error:  Out of memory.\n");
		exit(1);
	}

	m_nextItem = m_nowBlock->items;
	m_unallocatedItems = m_numberOfItemsPerBlock;
}


/*****************************************************************************/
/*                                                                           */
/*  PoolRestart()   Deallocate all m_numberOfItems in a pool.                          */
/*                                                                           */
/*  The pool is returned to its starting state, except that no memory is     */
/*  freed to the operating system.  Rather, the previously allocated blocks  */
/*  are ready to be reused.                                                  */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::PoolRestart()
{
	if( m_blockSum ) {
		m_numberOfItems = 0;
		m_maxNumberOfItems = 0;
		
		/* Set the currently active block. */
		m_nowBlock = m_blocks;
		m_nextItem = (TYPE *)m_nowBlock->items;
		
		m_unallocatedItems = m_numberOfItemsPerBlock;
		/* The stack of deallocated m_numberOfItems is empty. */
		m_deadItemStack = NULL;
	}
}


/*****************************************************************************/
/*                                                                           */
/*  PoolAlloc()   Allocate space for an item.                                */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
TYPE *CMemoryPool<TYPE, ARG_TYPE>::PoolAlloc()
{
	TYPE *newItem;
	
	/* First check the linked list of dead m_numberOfItems.  If the list is not   */
	/*   empty, allocate an item from the list rather than a fresh one. */
	if( m_deadItemStack != NULL) {
		newItem = m_deadItemStack;               /* Take first item in list. */
		m_deadItemStack = *(TYPE **)m_deadItemStack;
	} else {
		/* Check if there are any free m_numberOfItems left in the current block. */
		if( m_unallocatedItems == 0 ) {
			/* Check if another block must be allocated. */
			IncreaseABlock();
		}
		/* Allocate a new item. */
		newItem = m_nextItem;
		/* Advance `m_nextItem' pointer to next free item in block. */
		m_nextItem++;
		m_unallocatedItems--;
		m_maxNumberOfItems++;
	}
	m_numberOfItems++;

	return newItem;
}

/*****************************************************************************/
/*                                                                           */
/*  PoolDealloc()   Deallocate space for an item.                            */
/*                                                                           */
/*  The deallocated space is stored in a queue for later reuse.              */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::PoolDealloc(TYPE *dyingitem)
{
	/* Push freshly killed item onto stack. */
	*((TYPE **) dyingitem) = m_deadItemStack;
	m_deadItemStack = dyingitem;
	m_numberOfItems--;
}

/*****************************************************************************/
/*                                                                           */
/*  TraversalInit()   Prepare to Traverse the entire list of m_numberOfItems.          */
/*                                                                           */
/*  This routine is used in conjunction with Traverse().                     */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
void CMemoryPool<TYPE, ARG_TYPE>::TraversalInit()
{
	/* Begin the traversal in the first block. */

	if( m_blockSum ) {
		m_pathBlock = m_blocks;
		m_pathItem = m_pathBlock->items;
		m_pathItemsLeft = m_numberOfItemsPerBlock;
	}
}

/*****************************************************************************/
/*                                                                           */
/*  Traverse()   Find the next item in the list.                             */
/*                                                                           */
/*  This routine is used in conjunction with TraversalInit().  Be forewarned */
/*  that this routine successively returns all m_numberOfItems in the list, including  */
/*  deallocated ones on the deaditemqueue.  It's up to you to figure out     */
/*  which ones are actually dead.  Why?  I don't want to allocate extra      */
/*  space just to demarcate dead m_numberOfItems.  It can usually be done more         */
/*  space-efficiently by a routine that knows something about the structure  */
/*  of the item.                                                             */
/*                                                                           */
/*****************************************************************************/

template<class TYPE, class ARG_TYPE>
TYPE *CMemoryPool<TYPE, ARG_TYPE>::Traverse()
{
	TYPE *newitem;

	if( m_blockSum == 0 ) 
		return NULL;

	/* Stop upon exhausting the list of m_numberOfItems. */
	if( m_pathItem == m_nextItem)
		return NULL;

	/* Check whether any unTraversed m_numberOfItems remain in the current block. */
	if ( m_pathItemsLeft == 0 ) {
		/* Find the next block. */
		m_pathBlock++;
		m_pathItem = m_pathBlock->items;
		m_pathItemsLeft = m_numberOfItemsPerBlock;
	}

	newitem = m_pathItem;
	/* Find the next item in the block. */
	m_pathItem++;
	m_pathItemsLeft--;

	return newitem;
}


template<class TYPE, class ARG_TYPE>
TYPE *CMemoryPool<TYPE, ARG_TYPE>::GetItem( int number )
{
	int blockNum, itemNum;

	blockNum = number / m_numberOfItemsPerBlock;
	itemNum = number - blockNum*m_numberOfItemsPerBlock;

	return  m_blocks[blockNum].items + itemNum ;
}


template<class TYPE, class ARG_TYPE>
TYPE *CMemoryPool<TYPE, ARG_TYPE>::GotoItem( int number )
{
	int blockNum, itemNum;

	blockNum = number / m_numberOfItemsPerBlock;
	itemNum = number - blockNum*m_numberOfItemsPerBlock;

	m_pathBlock = m_blocks+blockNum;
	m_pathItem = m_pathBlock->items + itemNum;
	TYPE *newItem = m_pathItem++;

	m_pathItemsLeft = m_numberOfItemsPerBlock - itemNum -1;

	return newItem;
}

#endif