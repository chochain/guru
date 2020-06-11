/*! @file
  @brief
  GURU instruction unit
    1. guru VM, host or cuda image, constructor and dispatcher
    2. dumpers for regfile and irep tree

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  initialize VM
  	  allocate vm cuda memory
  	  parse mruby bytecodee
  	  dump irep tree (optional)
  execute VM
  	  opcode execution loop (on GPU)
  	    ucode_prefetch
  	    ucode_exec
  	    flush output per step (optional)
  </pre>
*/
#include "guru.h"
#include "base.h"
#include "util.h"
#include "static.h"
#include "mmu.h"
#include "symbol.h"
#include "c_string.h"

#include "class.h"
#include "state.h"
#include "vm.h"
#include "vmx.h"
#include "load.h"

__GURU__ void
VM::init(int id_, int step_)
{
	id    = id_;
	step  = step_;
	xcp   = err  = 0;
	run   = VM_STATUS_FREE;					// VM not allocated
}

//================================================================
// Fetch a VM for operation
// Note: thread 0 is the master controller, no other thread can
//       modify the VM status
//
#if GURU_HOST_GRIT_IMAGE
__GURU__ void
VM::load_grit(U8 *u8_gr)
{
	if (run==VM_STATUS_FREE) {
		_transcode(u8_gr);
		_setup(MEMOFF(U8PADD(u8_gr, ((GRIT*)u8_gr)->reps)));
    }
}
#else
__GURU__ void
VM::load_grit(U8 *ibuf)
{
	if (vm->run==VM_STATUS_FREE) {
		U8 *u8_gr = parse_bytecode(ibuf);
		_transcode(u8_gr);
		_setup(MEMOFF(U8PADD(u8_gr, ((GRIT*)u8_gr)->reps)));
    }
}
#endif // GURU_HOST_GRIT_IMAGE

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GR[] (i.e. regfile)
//
__GURU__ void
VM::_transcode(U8 *u8_gr)
{
	GRIT *gr = (GRIT*)u8_gr;
	GR   *r0 = (GR*)U8PADD(gr, gr->pool), *r = r0;
	U32  n   = 0;
	for (int i=0; i < gr->psz; i++, r0++) {
		if (r0->gt==GT_STR) n++;
	}
	guru_str *str = n ? (guru_str*)guru_alloc(sizeof(guru_str)*n) : NULL;

	for (int i=0; i < gr->psz; i++, r++) {						// symbol table
		switch (r->gt) {
		case GT_SYM: guru_sym_transcode(r);			break;
		case GT_STR: guru_str_transcode(r, str++);	break;		// instantiate the string
		default:
			// do nothing
			break;
        }
    }
}

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
 */
__GURU__ void
VM::_setup(GP irep)
{
	GR v { GT_CLASS, 0, 0, guru_rom_get_class(GT_OBJ) };
	GR *rf = (GR*)guru_alloc(sizeof(GR) * VM_REGFILE_SIZE);
	GR *r  = rf;
    for (int i=0; rf && i<VM_REGFILE_SIZE; i++, r++) {	// wipe register
    	*r = (i==0) ? v : EMPTY;
    }
	run     = VM_STATUS_READY;
    xcp     = err = (rf==NULL);
    state   = NULL;
	regfile = MEMOFF(rf);

	StateMgr *sm = new StateMgr(this);					// needs a helper
	sm->push_state(irep, 0, rf, 0);
}
