/*! @file
  @brief
  GURU - static data declarations

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "class.h"
#include "static.h"

__GURU__ guru_rom guru_device_rom { 0 };
__GURU__ guru_rom *_rom = &guru_device_rom;

#define _LOCK		{ MUTEX_LOCK(guru_device_rom); }
#define _UNLOCK 	{ MUTEX_FREE(guru_device_rom); }

__GURU__ guru_class*
_define_class(const char *name, GT cid, GT super_cid)
{
	guru_class *cx = _CLS(_rom->cls) + cid;		// offset from _rom->cls

    cx->rc     = cx->n = cx->kt = 0;			// BUILT-IN class
    cx->cid    = guru_rom_add_sym(name);		// symbol id
    cx->var    = 0;								// class variables, lazily allocated when needed
    cx->meta   = 0;								// meta-class, lazily allocated when needed
    cx->super  = guru_rom_get_class(super_cid);
    cx->vtbl   = 0;
    cx->flist  = 0;								// head of list

    return cx;
}

//================================================================
/* methods to add builtin (ROM) class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ int
guru_rom_init()
{
	guru_rom *rom = &guru_device_rom;

	rom->cls = MEMOFF(guru_alloc(sizeof(guru_class)*MAX_ROM_CLASS));
	rom->prc = MEMOFF(guru_alloc(sizeof(guru_proc) *MAX_ROM_PROC));
	rom->sym = MEMOFF(guru_alloc(sizeof(guru_sym)  *MAX_ROM_SYMBOL));
	rom->str = MEMOFF(guru_alloc(sizeof(U8)        *MAX_ROM_STRBUF));

	return !(rom->cls && rom->prc && rom->sym && rom->str);
}

__GURU__ void
guru_rom_burn()
{
	guru_device_rom.ncls = GT_MAX;
}

__GURU__ S32
guru_rom_get_sym(const char *s1)
{
	guru_sym *sym = _SYM(0);
	U32      hsh1 = HASH(s1);
	for (int i=0; i<_rom->nsym; i++, sym++) {			// sequential search
		if (sym->hash==hsh1) {
#if CC_DEBUG
			PRINTF("  sym[%02d]->str%04x:H%08x~%s\n", i, i, sym->raw, MEMPTR(_rom->str)+sym->raw);
#endif // CC_DEBUG
			return i;
		}
	}
	return -1;
}

__GURU__ S32
guru_rom_add_sym(const char *s1)						// create new symbol
{
	S32 sid = guru_rom_get_sym(s1);
	if (sid>=0) {
		return sid;
	}
	// create new symbol
	U16      ns   = _rom->nsym;
	guru_sym *sym = _SYM(ns);
	sym->hash = HASH(s1);
	sym->raw  = _rom->nstr;								// offset from _rom->str
	U32  asz  = STRLENB(s1)+1;							// add '\0'

	U8   *s0  = MEMPTR(_rom->str)+sym->raw;
	MEMCPY(s0, s1, asz);								// deep copy

#if GURU_DEBUG
	ASSERT((_rom->nstr + ALIGN4(asz)) < MAX_ROM_STRBUF);
	ASSERT((_rom->nsym + 1) < MAX_ROM_SYMBOL);
#endif // GURU_DEBUG
#if CC_DEBUG
	PRINTF("  sym[%02d]->str%04x:x%08x %s\n", ns, _rom->nstr, sym->hash, s1);
#endif // CC_DEBUG
	_rom->nstr += ALIGN4(asz);
	_rom->nsym++;

	return ns;
}

__GURU__ GP
guru_rom_get_class(GT cid)
{
	return cid ? MEMOFF(_CLS(_rom->cls)+cid) : 0;	// memory offset to the class object
}

__GURU__ GP
guru_rom_add_class(GT cid, const char *name, GT super_cid, const Vfunc vtbl[], int n)
{
#if CC_DEBUG
    PRINTF("%s:: class[%d] defined with %d method(s) at vtbl[%2d]\n", name, cid, n, _rom->nprc);
#endif // CC_DEBUG

#if GURU_DEBUG
    ASSERT((_rom->nprc + n) < MAX_ROM_PROC);	// size checking
#endif // GURU_DEBUG

    guru_class *cx  = _define_class(name, ((cid==GT_EMPTY) ? (GT)_rom->ncls : cid), super_cid);
    guru_proc  *px  = _PRC(_rom->prc) + _rom->nprc;

    cx->rc   = n;								// number of built-in functions
    cx->vtbl = n ? MEMOFF(px) : 0;				// built-in proc starting index

    Vfunc *fp = (Vfunc*)vtbl;					// TODO: nvcc allocates very sparsely for String literals
    for (int i=0; i<n; i++, px++, fp++) {
    	px->rc   = px->kt = px->n = 0;			// built-in class type (not USER_DEF_CLASS)
    	px->pid  = guru_rom_add_sym(fp->name);	// raw string function table defined in code
    	px->cid  = cx->cid;
    	px->func = MEMOFF(fp->func);
    	px->next = 0;
    }
    _rom->nprc += n;							// advance proc counter
    _rom->ncls++;								// count them

    return MEMOFF(cx);
}

//================================================================
/*!@brief
  define class method or instance method.

  @param  vm		pointer to vm.
  @param  cls		pointer to class.
  @param  name		method name.
  @param  cfunc		pointer to function.
*/
__GURU__ GP
guru_define_method(GP cls, const U8 *name, GP cfunc)
{
    if (!cls) cls = guru_rom_get_class(GT_OBJ);	// set default to Object.

#if GURU_DEBUG
    ASSERT((_rom->nprc + 1) < MAX_ROM_PROC);
#endif // GURU_DEBUG

    guru_proc  *px  = _PRC(_rom->prc) + _rom->nprc++;
    guru_class *cx = _CLS(cls);

    px->rc = px->kt = px->n = 0;				// No LAMBDA register file, C-function (from BUILT-IN class)
    px->pid   = name ? guru_rom_add_sym((char*)name) : 0xffff;
    px->cid   = cx->cid;						// keep class id
    px->func  = cfunc;							// set function pointer

    GP prc    = MEMOFF(px);
    _LOCK;										// cached class
    px->next  = cx->flist;						// add as the new list head
    cx->flist = prc;							// TODO: change to array implementation
    _UNLOCK;

    return prc;
}


