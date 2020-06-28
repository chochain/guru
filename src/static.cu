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

	_CLS(guru_rom_get_class(GT_EMPTY))->cid = 0xffff;		// make sure it will not be used

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
			U8 *s0 = MEMPTR(_rom->str)+sym->raw;
			PRINTF("  sym[%02x]->str%04x:x%08x~%s\n", i, sym->raw, MEMOFF(s0), s0);
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
	PRINTF("  sym[%02x]->str%04x:H%08x %s\n", ns, _rom->nstr, sym->hash, s1);
#endif // CC_DEBUG
	_rom->nstr += ALIGN4(asz);
	_rom->nsym++;

	return ns;
}

__GURU__ GP
guru_rom_get_class(GT cidx)
{
	return MEMOFF(_CLS(_rom->cls) + cidx);				// memory offset to the class object
}

__GURU__ GP
guru_rom_add_class(GT cidx, const char *name, GT super_cidx, const Vfunc mtbl[], int n)
{
#if CC_DEBUG
    PRINTF("%s:: class[%d] defined with %d method(s) at mtbl[%2d]\n", name, cidx, n, _rom->nprc);
#endif // CC_DEBUG

#if GURU_DEBUG
    ASSERT((_rom->nprc + n) < MAX_ROM_PROC);			// size checking
#endif // GURU_DEBUG

    guru_proc  *px = _PRC(_rom->prc) + _rom->nprc;
	guru_class *cx = _CLS(_rom->cls) + cidx;			// offset from _rom->cls
	GP cid   = guru_rom_add_sym(name);
	GP scls  = super_cidx ? guru_rom_get_class(super_cidx) : 0;	// 0: Object (root) class
    GP cls   = guru_define_class(cx, cid, scls);

    cx->kt  |= CLASS_BUILTIN;
    cx->rc   = n;										// number of built-in functions
    cx->meta = cls;										// TODO: for now, BUILTIN classes uses itself as metaclass to save one block
    cx->mtbl = n ? MEMOFF(px) : 0;						// built-in proc starting index

    Vfunc *fp = (Vfunc*)mtbl;							// TODO: nvcc allocates very sparsely for String literals
    for (int i=0; i<n; i++, px++, fp++) {
    	px->rc   = px->kt = px->n = 0;					// kt:0 built-in C-function
    	px->pid  = guru_rom_add_sym(fp->name);			// raw string function table defined in code
    	px->cid  = cid;
    	px->func = MEMOFF(fp->func);
    	px->next = 0;
    }
    _rom->nprc += n;									// advance proc counter
    _rom->ncls++;										// count them

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
    ASSERT(cls);								// set default to Object.

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
    cx->flist = prc;							// TODO: change to array implementation, to fix cls_08 unit test
    _UNLOCK;

    return prc;
}


