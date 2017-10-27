
#if !defined(AFX_RANDOM_H_INCLUDED_)
#define AFX_RANDOM_H_INCLUDED_


#ifdef TINDLL_EXPORTS
#define TINDLL_API __declspec(dllexport)
#else
#define TINDLL_API __declspec(dllimport)
#endif


TINDLL_API unsigned long Randomnation();
TINDLL_API unsigned long randomnation(unsigned int choices);
TINDLL_API void SetRandom(unsigned long randomSeed, unsigned int choices=0);


#endif