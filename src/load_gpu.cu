/*! @file
  @brief
  GURU bytecode loader (IREP code parse by CUDA device directly).

  alternatively, load.cu can be used for host built image (then passed into device for execution)
  <pre>
  Copyright (C) 2019- Greeni

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "mmu.h"
#include "util.h"
#include "symbol.h"
#include "c_string.h"

#include "base.h"
#include "state.h"
#include "vm.h"
#include "load.h"
#include "errorcode.h"

#define BU16(b)		(bin_to_u16((const void*)(b)))
#define BU32(b)		(bin_to_u32((const void*)(b)))

#if !GURU_HOST_IMAGE
//================================================================
/*!@brief
  Parse header section.

  @param  bp	A pointer of pointer of RITE header.
  @return int	zero if no error.

  <pre>
  Structure
  "RITE"	identifier
  "0004"	version
  0000		CRC
  0000_0000	total size
  "MATZ"	compiler name
  "0000"	compiler version
  </pre>
*/
__GURU__ int
_check_header(U8 **bp)
{
    const U8 *p = *bp;

    if (MEMCMP(p, "RITE000", 7)==0) {
    	// Rite binary version
    	// 0002: mruby 1.0
    	// 0003: mruby 1.1, 1.2
    	// 0004: mruby 1.3, 1.4
    	// 0005: mruby 2.0
    	U8 c = *(p+7);
        if (c < '3' || c > '4') {
        	return LOAD_FILE_HEADER_ERROR_VERSION;
        }
    }
    /* Ignore CRC */
    /* Ignore size */
    if (MEMCMP(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    // Rite VM version
    // 0000: mruby 1.x
    // 0002: mruby 2.x
    if (MEMCMP(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }

    *bp += 22;		// advance beyond header to IREP block

    return NO_ERROR;
}

__GURU__ void
_get_rite_size(rite_size *sz, U8 **bp)
{
    U8 *p = *bp;

    // sz,nlocals,nregs,rlen
    p += sizeof(U32)*2;									// IREP block size, nlocals, nregs
    U32 rsz = BU16(p);		p += sizeof(U16);			// number of child IREP blocks
    U32 isz = BU32(p);		p += sizeof(U32);			// ISEQ (bytecodes) length
    p += (-(U32A)p & 3);								// 32-bit align code pointer
    p += sizeof(U32)*isz;								// skip ISEQ (code) block

    // POOL block
    U32 psz = BU32(p);		p += sizeof(U32);			// pool element count
    for (U32 i=0; i<psz; i++) {							// 1st pass (skim through pool)
    	U8  tt  = *p++;
    	U32 len = BU16(p);	p += sizeof(U16)+len;
    	sz->ssz += (tt==0)
    		? ALIGN4(len+1) : ((tt==3) ? ALIGN4(len) : 0);
    }
    // SYM block
    U32 ysz = BU32(p);		p += sizeof(U32);			// symbol element count
    for (U32 i=0; i<ysz; i++) {							// 1st pass (skim through sym)
    	U32 len = BU16(p)+1;	p += sizeof(U16)+len;
    	sz->ssz += ALIGN4(len);
    }
    *bp = p;											// update scanner pointer

    sz->rsz += rsz;
    sz->isz += isz;
    sz->psz += (psz + ysz);

    for (int i=0; i<rsz; i++) {							// tail recursion
    	_get_rite_size(sz, bp);
    }
}

__GURU__ void
_reset_grit(GRIT *gr)
{
	gr->reps = (guru_irep*)U8PADD(gr, sizeof(GRIT));
	gr->pool = (GV*) U8PADD(gr->reps, gr->rsz * sizeof(guru_irep));
	gr->iseq = (U32*)U8PADD(gr->pool, gr->psz * sizeof(GV));
	gr->stbl = (U8*) U8PADD(gr->iseq, gr->isz * sizeof(U32));
}

__GURU__ GRIT*
_alloc_grit(U8 **bp)
{
    rite_size sz { .rsz=1, .psz=0, .isz=0, .ssz=0 };

    _get_rite_size(&sz, bp);

    U32 bsz =
    		sizeof(GRIT) +
    		sz.rsz * sizeof(guru_irep) +
    		sz.psz * sizeof(GV) +
    		sz.isz * sizeof(U32) +
    		sz.ssz;

    GRIT *gr = (GRIT*)guru_alloc(ALIGN8(bsz));

    MEMCPY(gr, &sz, sizeof(rite_size));

    return gr;
}

__GURU__ void
_to_gv(GV *v, U8 **stbl, U8 *p, U32 tt, U32 len)
{
    // build POOL or SYM block
	U8   *tgt = *stbl;
    char buf[64+1];
    U8   c;

    v->acl = 0;							// clear access bit (~HAS_REF, ~SCLASS, ~SELF)
    switch (tt) {
    case 0:	// String
    	v->gt  = GT_STR;
    	v->buf = tgt;

    	c = *(p+len);					// RITE does not 0 terminate (Ruby bug)
    	*(p+len) = '\0';				// so, we have to hack it here (BAD! write to input stream)
    	MEMCPY(tgt, p, len);
        tgt += ALIGN4(len+1);
    	*(p+len) = c;					// put it back
        break;
    case 1: // Integer (31-bit)
    	v->gt  = GT_INT;
    	MEMCPY(buf, p, len);
    	buf[len] = '\0';
    	v->i   = ATOI(buf, 10);
    	break;
    case 2: // Float (32-bit)
    	v->gt  = GT_FLOAT;
    	MEMCPY(buf, p, len);
    	buf[len] = '\0';
    	v->f   = (float)ATOF(buf);		// atof() returns double
    	break;
    case 3: // Symbol
    	v->gt  = GT_SYM;
    	v->buf = tgt;

    	MEMCPY(tgt, p, len);
    	tgt += ALIGN4(len);
    	break;
    default: // Others (not yet supported)
    	v->gt   = GT_NIL;
    	v->self = NULL;
    	break;
    }
    *stbl = tgt;
}

//================================================================
/*!@brief
  read one irep section.

  @param  gr    GRIT pointer
  @param  ix    offset index to IREP section
  @param  bp	A pointer of pointer of input byte stream
  @return       count of child IREPs

  <pre>
  (loop n of child irep bellow)
  0000_0000	record size
  0000		n of local variable
  0000		n of register
  0000		n of child irep

  0000_0000	n of byte code  (ISEQ BLOCK)
  ...		byte codes		(padded to 4-byte align)

  0000_0000	n of pool		(POOL BLOCK)
  (loop n of pool)
  00		type
  0000		length
  ...		pool data

  0000_0000	n of symbol		(SYMS BLOCK)
  (loop n of symbol)
  0000		length
  ...		symbol data
  </pre>
*/
__GURU__ int
_load_irep(GRIT *gr, int *r, int ix, U8 **bp)
{
    U8 *p = *bp;
    guru_irep *r0 = gr->reps + ix;

    // sz,nlocals,nregs,rlen
    p += sizeof(U32);									// IREP block size
    p += sizeof(U16);									// number of local variables
    r0->nr   = BU16(p);	p += sizeof(U16);				// number of registers used
    r0->r    = BU16(p);	p += sizeof(U16);				// number of child IREP blocks

    r0->reps = r0->r ? &gr->reps[*r] : NULL;

    // ISEQ block
    U32 isz = BU32(p);		p += sizeof(U32);			// number of ISEQ instructions

    U8 *iseq    = (p += -(U32A)p & 3);					// 32-bit align code pointer
    U32 iseq_sz = sizeof(U32) * isz;	p += iseq_sz;	// skip ISEQ (code) block

    r0->iseq = gr->iseq;
    gr->iseq += isz;									// advance ISEQ pointer
    MEMCPY(r0->iseq, iseq, iseq_sz);					// copy ISEQ

    // POOL block
    GV *pool = r0->pool = gr->pool;						// capture pool pointer
    U8 *stbl = gr->stbl;
    U32 psz  = r0->p = BU32(p);	p += sizeof(U32);		// pool element count
    for (U32 i=0; i<psz; i++) {							// 1st pass (skim through pool)
        U32  tt = *p++;
        U32  len = BU16(p);    	p += sizeof(U16);
    	_to_gv(pool++, &stbl, p, tt, len);
    	p += len;
    }
    // SYM block
    U32 ssz = r0->s = BU32(p);	p += sizeof(U32);		// symbol element count
    for (U32 i=0; i<ssz; i++) {							// 1st pass (skim through sym)
    	U32 len = BU16(p)+1;	p += sizeof(U16);
    	_to_gv(pool++, &stbl, p, 3, len);
    	p += len;
    }
    gr->pool += (psz + ssz);							// advance pool pointer
    gr->stbl  = stbl;
    *bp = p;

	return r0->r;
}

//================================================================
/*!@brief
  transform irep-tree from depth-first to breadth-first (using little brother algo)

  @param  gr    GRIT pointer
  @param  r     A pointer to forward index
  @param  ix    IREP index in GRIT
  @param  bp	A pointer of pointer of input byte-stream
  @return       void

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__GURU__ void
_build_image(GRIT *gr, int *r, int ix, U8 **bp)
{
    U32 rsz = _load_irep(gr, r, ix, bp);
    ix  = *r;								// remember where we were (little brother)
    *r += rsz;								// total allocated (big brother)

    // traverse irep-tree recursively
	for (U32 i=0; i<rsz; i++) {
		_build_image(gr, r, ix+i, bp);
	}
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  bp	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__GURU__ int
_load_lvar(GRIT *gr, U8 **bp)
{
    U8 *p = *bp;

    /* size */
    *bp += BU32(p+sizeof(U32));

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  src	Pointer to bytecode.
*/
__GURU__ GRIT*
parse_bytecode(U8 *src)
{
	U8 **bp = &src;
    int ret = _check_header(bp);

    GRIT *gr;
    while (ret==NO_ERROR) {
        if (MEMCMP(*bp, "IREP", 4)==0) {						// switch alternative
        	*bp += 4 + sizeof(U32);								// skip "IREP", irep_sz
            if (MEMCMP(*bp, "0000", 4) != 0) break;				// IREP version
            *bp += 4;											// skip "0000"

            U8 *p  = *bp;
            gr  = _alloc_grit(&p);
            _reset_grit(gr);

            int r = 1;
            _build_image(gr, &r, 0, bp);
        	_reset_grit(gr);

        	ret = NO_ERROR;
        }
        else if (MEMCMP(*bp, "LVAR", 4)==0) {
            ret = _load_lvar(gr, bp);
        }
        else if (MEMCMP(*bp, "END\0", 4)==0) {
            break;
        }
    }
    return (ret==NO_ERROR) ? gr : NULL;
}
#endif // !GURU_HOST_IMAGE
