/*! @file
  @brief
  GURU bytecode loader (host load IREP code, build image and copy into CUDA memory).

  alternatively, load_gpu.cu can be used for device image building
  <pre>
  Copyright (C) 2019- Greeni

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>
#include "vm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "guru.h"
#include "value.h"
#include "alloc.h"
#include "errorcode.h"
#include "vm.h"
#include "load.h"

#if GURU_HOST_IMAGE
//================================================================
/*!@brief
  Parse header section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of RITE header.
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
__HOST__ U32
bin_to_u32(const void *s)
{
    U32 x = *((U32P)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__HOST__ U16
bin_to_u16(const void *s)
{
    U16 x = *((U16 *)s);
    return (x << 8) | (x >> 8);
}

__HOST__ int
_check_header(U8P *pos)
{
    const U8 * p = *pos;

    if (memcmp(p, "RITE0004", 8) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    /* Ignore CRC */
    /* Ignore size */

    if (memcmp(p + 14, "MATZ", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_MATZ;
    }
    if (memcmp(p + 18, "0000", 4) != 0) {
        return LOAD_FILE_HEADER_ERROR_VERSION;
    }
    *pos += 22;

    return NO_ERROR;
}

//
// building memory image, offset-based with alignment
//
__HOST__ guru_irep *
_build_image(guru_irep *src, U8 * img)
{
	// compute CUDA alignment memory block sizes
    U32 irep_sz = sizeof(guru_irep) + (-sizeof(guru_irep) & 7);					// 8-byte alignment
    U32 reps_sz = sizeof(U32) * (src->c + (src->c & 1));
    U32 pool_sz = sizeof(U32) * (src->p + (src->p & 1));
    U32 sym_sz  = sizeof(U32) * (src->s + (src->s & 1));
    U32 iseq_sz = sizeof(U32) * (src->i + (src->i & 1));
    U32 stbl_sz = sizeof(U8P) * sym_sz * 2;										// string table with padded space
    U32 img_sz  = irep_sz + reps_sz + pool_sz + sym_sz + iseq_sz + stbl_sz;		// should be 8-byte aligned

    guru_irep *tgt = (guru_irep *)cuda_malloc(img_sz, 1);						// target CUDA IREP image (managed mem)
    U8 * base = (U8P)tgt;								// keep target image pointer
    if (!tgt) return NULL;

    // for debugging, blank out allocated memory
    memset(tgt, 0xaa, img_sz);

    // set CUDA memory pointers, and irep offsets
	U8 * reps = U8PADD(tgt,  irep_sz);
	U8 * pool = U8PADD(reps, reps_sz);
	U8 * sym  = U8PADD(pool, pool_sz);
	U8 * iseq = U8PADD(sym,  sym_sz);
	U8 * stbl = U8PADD(iseq, iseq_sz);					// raw string table pointer

    // start building the CUDA image (with alignment)
    memcpy(tgt, src, irep_sz);							// copy IREP header block
    memcpy(iseq, U8PADD(img, src->iseq), iseq_sz);		// copy ISEQ block

	tgt->reps = U8POFF(reps, tgt);						// overwrite with new reps offset
    tgt->iseq = U8POFF(iseq, tgt);						// new iseq offset
    tgt->pool = U8POFF(pool, tgt);						// new pool offset
    tgt->sym  = U8POFF(sym,  tgt);						// new sym offset

    // build POOL block
    U8 *  p = U8PADD(img, src->pool);					// point to source object pool
    U32 * v = (U32 *)pool;
    for (U32 i=0; i < src->p; i++, v++) {
        U32  tt = *p++;
        U32  len = bin_to_u16(p);	p += sizeof(U16);
        U32  asz;								// adjusted size
        char buf[64+2];							// 64-bit number

        switch (tt) {
        case 0:									// IREP_TT_STRING
            asz = len + 1;						// '\0'
            asz += (-asz & 3);					// 4-byte aligned
            memcpy(stbl, p, asz);				// duplicate the raw string
        	*v = U8POFF(stbl, base);
        	stbl += asz;						// advance string table pointer
        	break;
        case 1: 								// IREP_TT_FIXNUM
            memcpy(buf, p, len);
            buf[len] = '\0';
            *v = atoi(buf)<<1;					// mark as number i.e. LSB=0
            break;
#if GURU_USE_FLOAT
        case 2: 								// IREP_TT_FLOAT
            memcpy(buf, p, len);
            buf[len] = '\0';
            *(GF *)v = atof(buf);
            *v |= 1;							// mark as float  i.e. LSB=1
            break;
#endif
        default:
        	*v = 0;								// other object are not supported yet
        	break;
        }
        p += len;
    }

    p = U8PADD(img, src->sym);					// build symbol table
    v = (U32 *)sym;
    for (U32 i=0; i < src->s; i++, v++) {
        U32 len = bin_to_u16(p)+1;	p += sizeof(U16);	// symbol_len+'\0'
        U32 asz = len + (-len & 3);				// 4-byte alignment

    	memcpy(stbl, p, len ? len : 1);			// copy the symbol string
    	*v = U8POFF(stbl, base);
    	stbl += asz;							// advance string table pointer
        p    += len;
    }
    tgt->size = U8POFF(stbl, tgt);

    return tgt;
}

//================================================================
/*!@brief
  read one irep section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated IREP or NULL

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
__HOST__ U8P
_fetch_irep_size(guru_irep *irep, U8P img)						// pos will be advance to next IREP block
{
    U8 * p = img;

    // Header: sz,nlocals,nregs,rlen
    irep->size 	= bin_to_u32(p); 	p += sizeof(U32);			// IREP size
    irep->nv 	= bin_to_u16(p);	p += sizeof(U16);			// number of local variables
    irep->nr 	= bin_to_u16(p);	p += sizeof(U16);			// number of registers used
    irep->c  	= bin_to_u16(p);	p += sizeof(U16);			// number of child IREP blocks
    // ISEQ block
    irep->i = bin_to_u32(p);		p += sizeof(U32);			// ISEQ (bytecodes) length
    p += -(U32A)p & 03;											// supposedly 32-bit aligned already
    irep->iseq = U8POFF(p, img);	p += irep->i * sizeof(U32);	// ISEQ (code) block
    // POOL block
    irep->p    	= bin_to_u32(p);	p += sizeof(U32);			// pool element count
    irep->pool 	= U8POFF(p, img);								// pool offset
    for (U32 i=0; i < irep->p; i++) {	// scan through pool (so we know the size to allocate)
        int  tt  = *p++;										// object type
        int  len = bin_to_u16(p);   p += sizeof(U16) + len;
        tt = 0;
    }
    // SYM block
    irep->s 	= bin_to_u32(p);	p += sizeof(U32);			// compiled symbol counts
    irep->sym  	= U8POFF(p, img);								// symbol offset
    for (U32 i=0; i < irep->s; i++) {
        int len = bin_to_u16(p)+1;	p += sizeof(U16) + len;		// symbol_len+'\0'
        int tt = 0;
    }
    return p;													// position pointer ends here
}
//================================================================
/*!@brief
  Parse IREP section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of IREP section.
  @return       Pointer of allocated IREP or NULL

  <pre>
  Structure
  "IREP"	section identifier
  0000_0000	section size
  "0000"	rite version
  </pre>
*/
__HOST__ guru_irep *
_load_irep(U8P *pos)
{
    U8 * img = *pos;								// host image
    U8 * sp  = *pos;								// local pointer
    guru_irep src;									// temp store for source IREP

    sp = _fetch_irep_size(&src, img);				// populate metadata of current IREP, sp will be advanced to next IREP block
	guru_irep *irep = _build_image(&src, img);		// build CUDA image from host image

    // recursively create the child irep tree
    U32 * v = (U32P)U8PADD(irep, irep->reps);
    for (U32 i=0; i < src.c; i++) {					// number of irep children
    	guru_irep *irep_n = _load_irep(&sp);		// load a child irep recursively
        v[i] = U8POFF(irep_n, irep);				// calculate offset
    }
	*pos = sp;

    return irep;
}

//================================================================
/*!@brief
  Parse LVAR section.

  @param  vm    A pointer of VM.
  @param  pos	A pointer of pointer of LVAR section.
  @return int	zero if no error.
*/
__HOST__ U32
_load_lvar(U8P *pos)
{
    U8P p      = *pos;
    U32 sec_sz = bin_to_u32(p+sizeof(U32));

    // TODO: local variable is not supported yet

    *pos += sec_sz;

    return NO_ERROR;
}

//================================================================
/*!@brief
  Load the VM bytecode.

  @param  vm    Pointer to VM.
  @param  ptr	Pointer to bytecode.

*/
__HOST__ void
guru_parse_bytecode(guru_vm *vm, U8P src)
{
	U8P *sp = (U8P *)&src;
	int ret = _check_header(sp);

    while (ret==NO_ERROR) {
        if (memcmp(*sp, "IREP", 4)==0) {
        	*sp += 4 + sizeof(U32);								// skip "IREP", irep_sz
            if (memcmp(*sp, "0000", 4) != 0) break;				// IREP version
            *sp += 4;											// skip "0000"

        	ret = (vm->irep = _load_irep(sp))==NULL
        			? LOAD_FILE_IREP_ERROR_ALLOCATION
        			: NO_ERROR;
        }
        else if (memcmp(*sp, "LVAR", 4)==0) {
            ret = _load_lvar(sp);
        }
        else if (memcmp(*sp, "END\0", 4)==0) {
            break;
        }
    }
}

#if GURU_DEBUG
__HOST__ void
_show_irep(guru_irep *irep, U32 ioff, char level, char *idx)
{
	printf("\tirep[%c]=%c%04x: size=%d, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			*idx, level, ioff,
			irep->size, irep->nr, irep->nv, irep->p, irep->s, irep->c, irep->i);

	// dump all children ireps
	U8  *base = (U8 *)irep;
	U32 *off  = (U32 *)U8PADD(base, irep->reps);		// pointer to irep offset array
	for (U32 i=0; i<irep->c; i++) {
		*idx += 1;
		_show_irep((guru_irep *)(base + off[i]), off[i], level+1, idx);
	}
}

__HOST__ void
guru_show_irep(guru_irep *irep)
{
	char idx = 'a';
	_show_irep(irep, 0, 'A', &idx);
}
#else
__HOST__ void guru_show_irep(guru_irep *irep) {}
#endif	// GURU_DEBUG
#endif 	// GURU_HOST_IMAGE
