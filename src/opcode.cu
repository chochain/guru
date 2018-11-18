/*! @file
  @brief
  Guru bytecode executor.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/

#include "alloc.h"
#include "instance.h"
#include "static.h"
#include "symbol.h"
#include "global.h"

#include "console.h"

#include "opcode.h"
#include "class.h"

#if GURU_USE_STRING
#include "c_string.h"
#endif
#if GURU_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

#define _RA_X(r)    do { mrbc_release(&regs[ra]); regs[ra] = *(r); mrbc_retain(r);  } while (0)
#define _RA_V(v)    do { mrbc_release(&regs[ra]); regs[ra] = (v); } while (0)
#define _RA_T(t, e) do { mrbc_release(&regs[ra]); regs[ra].tt = (t); regs[ra].e; } while (0)

//================================================================
/*!@brief

 */
//================================================================
/*! get sym[n] from symbol table in irep

  @param  p	Pointer to IREP SYMS section.
  @param  n	n th
  @return	symbol name string
*/
__GURU__ const char*
_get_symbol(const uint8_t *p, int n)
{
    int cnt = _bin_to_uint32(p);			p += sizeof(uint32_t);
    if (n >= cnt) return NULL;

    for (; n>0; n--) {	// advance to n'th symbol
        uint16_t s = _bin_to_uint16(p);		p += sizeof(uint16_t)+s+1;	// symbol len + '\0'
    }
    return (char *)p+2;  // skip size(2 bytes)
}

__GURU__ mrbc_sym
_get_symid(const uint8_t *p, int n)
{
  	const char *name = _get_symbol(p, n);
    return name2symid(name);
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
_push_state(mrbc_vm *vm, int argc)
{
    mrbc_state *ci  = (mrbc_state *)mrbc_alloc(sizeof(mrbc_state));
	mrbc_state *top = vm->state;


    ci->reg   = top->reg;			// pass register file
    ci->irep  = top->irep;
    ci->pc 	  = top->pc;
    ci->argc  = argc;
    ci->klass = top->klass;
    ci->prev  = top;

    vm->state = ci;
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
_pop_state(mrbc_vm *vm, mrbc_value *regs)
{
    mrbc_state *ci = vm->state;
    
    vm->state = ci->prev;
    
    // clear stacked arguments
    for (int i = 1; i <= ci->argc; i++) {
        mrbc_release(&regs[i]);
    }
    mrbc_free(ci);
}

__GURU__ void
_vm_proc_call(mrbc_vm *vm, mrbc_value v[], int argc)
{
	_push_state(vm, argc);			// check _funcall which is not used

	vm->state->pc   = 0;
	vm->state->irep = v[0].proc->irep;	// switch into callee context
	vm->state->reg  = v;					// shift register file pointer (for local stack)

	v[0].proc->refc++;				// CC: 20181027 added to track proc usage
}

// Object.new
__GURU__ void
_vm_object_new(mrbc_vm *vm, mrbc_value v[], int argc)
{
    mrbc_value obj = mrbc_instance_new(v[0].cls, 0);
    char sym[] = "______initialize";

    _uint32_to_bin(1, (uint8_t*)&sym[0]);		// setup symbol table
    _uint16_to_bin(10,(uint8_t*)&sym[4]);

    uint32_t code[2] = {
        (uint32_t)(MKOPCODE(OP_SEND) | MKARG_A(0) | MKARG_B(0) | MKARG_C(argc)),
        (uint32_t)(MKOPCODE(OP_ABORT))
    };
    mrbc_irep irep = {
        0,     				// nlv
        0,     				// nreg
        0,     				// rlen
        2,     				// ilen
        0,     				// plen
        0,					// slen
        code,   			// iseq
        (uint8_t *)sym,  	// ptr_to_sym
        NULL,  				// object pool
        NULL,  				// irep list
    };
    mrbc_release(&v[0]);
    v[0] = obj;
    mrbc_retain(&obj);

    // context switch, which is not multi-thread ready
    // TODO: create a vm context object with separate regfile
    mrbc_state  *ci    = vm->state;

    uint16_t    pc0    = ci->pc;
    mrbc_value  *reg0  = ci->reg;
    mrbc_irep 	*irep0 = ci->irep;

    ci->pc 	 = 0;
    ci->irep = &irep;
    ci->reg  = v;		   // new register file (shift for call stack)

    while (guru_op(vm)==0); // run til ABORT, or exception

    ci->pc 	 = pc0;
    ci->reg  = reg0;
    ci->irep = irep0;

    SET_RETURN(obj);
}

//================================================================
/*!@brief
  Execute OP_NOP

  No operation

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_nop(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MOVE

  R(A) := R(B)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_move(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    _RA_X(&regs[rb]);                  // swap ra <= rb

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADL

  R(A) := Pool(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadl(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    // regs[ra] = vm->pc_irep->pool[rb];

    mrbc_object *obj = GET_IREP(vm)->pool[rb];
    
    _RA_V(*obj);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADI

  R(A) := sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_FIXNUM, i=GETARG_sBx(code));

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADSYM

  R(A) := Syms(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadsym(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    _RA_T(GURU_TT_SYMBOL, i=_get_symid(GET_IREP(vm)->sym, rb));

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADNIL

  R(A) := nil

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadnil(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_NIL, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADSELF

  R(A) := self

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadself(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_X(&regs[0]);                   // ra <= vm

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADT

  R(A) := true

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_TRUE, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LOADF

  R(A) := false

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_loadf(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_FALSE, i=0);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_getglobal(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_value v = global_object_get(_get_symid(GET_IREP(vm)->sym, rb));
    _RA_V(v);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_setglobal(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_sym sym_id = _get_symid(GET_IREP(vm)->sym, rb);

    global_object_add(sym_id, regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETIV

  R(A) := ivget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_getiv(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const char *name  = _get_symbol(GET_IREP(vm)->sym, rb);
    mrbc_sym   sym_id = name2symid(name+1);	// skip '@'

    mrbc_value ret = mrbc_instance_getiv(&regs[0], sym_id);

    _RA_V(ret);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETIV

  ivset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_setiv(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const char *name  = _get_symbol(GET_IREP(vm)->sym, rb);
    mrbc_sym   sym_id = name2symid(name+1);	// skip '@'

    mrbc_instance_setiv(&regs[0], sym_id, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETCONST

  R(A) := constget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_getconst(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_value v = const_object_get(_get_symid(GET_IREP(vm)->sym, rb));

    _RA_X(&v);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETCONST

  constset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_setconst(mrbc_vm *vm, uint32_t code, mrbc_value *regs) {
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const_object_add(_get_symid(GET_IREP(vm)->sym, rb), &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETUPVAR

  R(A) := uvget(B,C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_getupvar(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   // UP

    mrbc_state *ci = vm->state;

    // find callinfo
    int n = rc * 2 + 1;
    while (n > 0){
        ci = ci->prev;
        n--;
    }
    mrbc_value *uregs = ci->reg;

    _RA_X(&uregs[rb]);             // ra <= up[rb]

    return 0;
}

//================================================================
/*!@brief
  Execute OP_SETUPVAR

  uvset(B,C,R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_setupvar(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   				// UP level

    mrbc_state *ci = vm->state;

    int n = (rc+1) << 1;					// 2 per outer scope level
    while (n > 0){
        ci = ci->prev;
        n--;
    }
    mrbc_value *up_regs = ci->reg;

    mrbc_release(&up_regs[rb]);
    up_regs[rb] = regs[ra];                // update outer-scope vars
    mrbc_retain(&regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMP

  pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_jmp(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    vm->state->pc += GETARG_sBx(code) - 1;
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPIF

  if R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_jmpif (mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt > GURU_TT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPNOT

  if not R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_jmpnot(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt <= GURU_TT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_send(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);  // index of method sym
    int rc = GETARG_C(code);  // number of params
    mrbc_value rcv = regs[ra];

    // Clear block param (needed ?)
    int bidx = ra + rc + 1;
    switch(GET_OPCODE(code)) {
    case OP_SEND:
        mrbc_release(&regs[bidx]);
        regs[bidx].tt = GURU_TT_NIL;
        break;
    case OP_SENDB:						// set Proc object
        if (regs[bidx].tt != GURU_TT_NIL && regs[bidx].tt != GURU_TT_PROC){
            // TODO: fix the following behavior
            // convert to Proc ?
            // raise exceprion in mruby/c ?
            return 0;
        }
        break;
    default: break;
    }

	mrbc_sym  sym_id = _get_symid(GET_IREP(vm)->sym, rb);
    mrbc_proc *m 	 = (mrbc_proc *)mrbc_get_class_method(rcv, sym_id);
#ifdef GURU_DEBUG
	const char *name = _get_symbol(GET_IREP(vm)->sym, rb);
#endif

    if (m==0) {
    	console_na(name);							// dump error, bail out
    	return 0;
    }

    if (m->flag & GURU_PROC_C_FUNC) {				// m is a C function
        if (m->func==c_proc_call) {
        	_vm_proc_call(vm, regs+ra, rc);
        }
        else if (m->func==c_object_new) {
        	_vm_object_new(vm, regs+ra, rc);
        }
        else {
        	m->func(regs+ra, rc);					// call the C-func
        	for (int i=ra+1; i<=bidx; i++) {		// clean up block parameters
                mrbc_release(&regs[i]);
            }
        }
    }
    else {								// m is a Ruby function
    	_push_state(vm, rc);			// append callinfo list

    	vm->state->irep = m->irep;		// call into target context
    	vm->state->pc 	= 0;			// call into target context
    	vm->state->reg 	+= ra;			// add call stack (new register)
    }

    return 0;
}

//================================================================
/*!@brief
  Execute OP_CALL

  R(A) := self.call(frame.argc, frame.argv)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_call(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    _push_state(vm, 0);

    // jump to proc
    vm->state->pc 	= 0;
    vm->state->irep = regs[0].proc->irep;

    return 0;
}



//================================================================
/*!@brief
  Execute OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_enter(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    uint32_t enter_param = GETARG_Ax(code);

    int def_args = (enter_param >> 13) & 0x1f;  // default args
    int argc     = (enter_param >> 18) & 0x1f;  // given args

    if (def_args > 0){
        vm->state->pc += vm->state->argc - argc;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_RETURN

  return R(A) (B=normal,in-block return/break)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_return(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    // return value
    int ra = GETARG_A(code);
    mrbc_value ret = regs[ra];

    mrbc_release(&regs[0]);
    regs[0]     = ret;
    regs[ra].tt = GURU_TT_EMPTY;

    _pop_state(vm, regs);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_BLKPUSH

  R(A) := block (16=6:1:5:4)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_blkpush(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_value *stack = regs + 1;       // call stack: push 1 mrbc_value

    if (stack[0].tt==GURU_TT_NIL){
        return vm->err = -1;  			// EYIELD
    }
    _RA_X(stack);                       // ra <= stack[0]

    return 0;
}

//================================================================
/*!@brief
  Execute OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_add(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i += regs[ra+1].i;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {	// in case of Fixnum, Float
            regs[ra].tt = GURU_TT_FLOAT;
            regs[ra].f = regs[ra].i + regs[ra+1].f;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f += regs[ra+1].i;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {	// in case of Float, Float
            regs[ra].f += regs[ra+1].f;
        }
#endif
    }
    else {    	// other case
    	op_send(vm, code, regs);		// should have already released regs[ra + n], ...
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_addi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        regs[ra].i += GETARG_C(code);
    }
#if GURU_USE_FLOAT
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        regs[ra].f += GETARG_C(code);
    }
#else
    else {
    	console_na("Float class");
    }
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_sub(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i -= regs[ra+1].i;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {		// in case of Fixnum, Float
            regs[ra].tt = GURU_TT_FLOAT;
            regs[ra].f = regs[ra].i - regs[ra+1].f;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	        // in case of Float, Fixnum
            regs[ra].f -= regs[ra+1].i;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {		// in case of Float, Float
            regs[ra].f -= regs[ra+1].f;
        }
#endif
    }
    else {  // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
	return 0;
}

//================================================================
/*!@brief
  Execute OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_subi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        regs[ra].i -= GETARG_C(code);
    }
#if GURU_USE_FLOAT
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        regs[ra].f -= GETARG_C(code);
    }
#else
    else {
    	console_na("Float class");
    }
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_mul(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i *= regs[ra+1].i;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {	// in case of Fixnum, Float
            regs[ra].tt = GURU_TT_FLOAT;
            regs[ra].f = regs[ra].i * regs[ra+1].f;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f *= regs[ra+1].i;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {	// in case of Float, Float
            regs[ra].f *= regs[ra+1].f;
        }
#endif
    }
    else {   // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_div(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i /= regs[ra+1].i;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {		// in case of Fixnum, Float
            regs[ra].tt = GURU_TT_FLOAT;
            regs[ra].f = regs[ra].i / regs[ra+1].f;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f /= regs[ra+1].i;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {		// in case of Float, Float
            regs[ra].f /= regs[ra+1].f;
        }
#endif
    }
    else {   // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_eq(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result = mrbc_compare(&regs[ra], &regs[ra+1]);

    _RA_T(result ? GURU_TT_FALSE : GURU_TT_TRUE, i=0);
    mrbc_release(&regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_lt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].i < regs[ra+1].i;	// in case of Fixnum, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].i < regs[ra+1].f;	// in case of Fixnum, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].f < regs[ra+1].i;	// in case of Float, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].f < regs[ra+1].f;	// in case of Float, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#endif
    }
    else {	// other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_le(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].i <= regs[ra+1].i;	// in case of Fixnum, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].i <= regs[ra+1].f;	// in case of Fixnum, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].f <= regs[ra+1].i;	// in case of Float, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].f <= regs[ra+1].f;	// in case of Float, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#endif
    }
    else {    // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_gt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].i > regs[ra+1].i;	// in case of Fixnum, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].i > regs[ra+1].f;	// in case of Fixnum, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].f > regs[ra+1].i;	// in case of Float, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].f > regs[ra+1].f;	// in case of Float, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#endif
    }
    else {    // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_ge(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==GURU_TT_FIXNUM) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].i >= regs[ra+1].i;	// in case of Fixnum, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#if GURU_USE_FLOAT
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].i >= regs[ra+1].f;	// in case of Fixnum, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
    }
    else if (regs[ra].tt==GURU_TT_FLOAT) {
        if (regs[ra+1].tt==GURU_TT_FIXNUM) {
            result = regs[ra].f >= regs[ra+1].i;	// in case of Float, Fixnum
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
        else if (regs[ra+1].tt==GURU_TT_FLOAT) {
            result = regs[ra].f >= regs[ra+1].f;	// in case of Float, Float
            regs[ra].tt = result ? GURU_TT_TRUE : GURU_TT_FALSE;
        }
#endif
    }
    else { // other case
    	op_send(vm, code, regs);
    }
    mrbc_release(&regs[ra+1]);
    return 0;
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_string(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if GURU_USE_STRING
	int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_object *obj = GET_IREP(vm)->pool[rb];

    /* CAUTION: pool_obj->sym - 2. see IREP POOL structure. */
    int         len = _bin_to_uint16(obj->sym - 2);
    const char *str = (const char *)obj->sym;			// 20181025
    mrbc_value  ret = mrbc_string_new(str);

    if (ret.str==NULL) return vm->err = -1;				// ENOMEM

    _RA_V(ret);
#else
    console_na("String class");
#endif
    return 0;
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_strcat(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if GURU_USE_STRING
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    // call "to_s"
    mrbc_sym sym_id = name2symid("to_s");
    mrbc_proc *m    = mrbc_get_class_method(regs[ra], sym_id);
    if (m && IS_C_FUNC(m)){
        m->func(regs+ra, 0);
    }
    m = mrbc_get_class_method(regs[rb], sym_id);
    if (m && IS_C_FUNC(m)){
        m->func(regs+rb, 0);
    }
    mrbc_value ret = mrbc_string_add(&regs[ra], &regs[rb]);

    _RA_V(ret);

#else
    console_na("String class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_array(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);
    int sz = sizeof(mrbc_value) * rc;

    mrbc_value ret = (mrbc_value)mrbc_array_new(rc);
    mrbc_array *h  = ret.array;
    if (h==NULL) return vm->err = -1;	// ENOMEM

    MEMCPY((uint8_t *)h->data, (uint8_t *)&regs[rb], sz);
    MEMSET((uint8_t *)&regs[rb], 0, sz);
    h->n = rc;

    _RA_V(ret);
#else
    console_na("Array class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_hash(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);
    int sz = sizeof(mrbc_value) * (rc<<1);				// size of k,v pairs

    mrbc_value ret = mrbc_hash_new(rc);
    mrbc_hash  *h  = ret.hash;
    if (h==NULL) return vm->err = -1;					// ENOMEM

    mrbc_value  *p = &regs[rb];
    MEMCPY((uint8_t *)h->data, (uint8_t *)p, sz);		// copy k,v pairs

    for (int i=0; i<(h->n=(rc<<1)); i++, p++) {
    	p->tt = GURU_TT_EMPTY;							// clean up call stack
    }
    _RA_V(ret);						                // set return value on stack top
#else
    console_na("Hash class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_range(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if GURU_USE_ARRAY
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);

    mrbc_value ret = mrbc_range_new(&regs[rb], &regs[rb+1], rc);
    if (ret.range==NULL) return vm->err = -1;		// ENOMEM

    _RA_V(ret);						// release and  reassign
    mrbc_retain(&regs[rb]);
    mrbc_retain(&regs[rb+1]);

#else
    console_na("Range class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_lambda(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_b(code);      		// sequence position in irep list
    // int c = GETARG_C(code);    		// TODO: Add flags support for OP_LAMBDA

    mrbc_proc *prc = (mrbc_proc *)mrbc_proc_alloc("(lambda)");

    prc->flag &= ~GURU_PROC_C_FUNC;	// IREP
    prc->irep = GET_IREP(vm)->ilist[rb];

    _RA_T(GURU_TT_PROC, proc=prc);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_CLASS

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1): super class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_class(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    mrbc_class *super = (regs[ra+1].tt==GURU_TT_CLASS) ? regs[ra+1].cls : mrbc_class_object;

    mrbc_irep  *cur_irep = GET_IREP(vm);
    const char *name     = _get_symbol(cur_irep->sym, rb);
    mrbc_class *cls 	 = (mrbc_class *)mrbc_define_class(name, super);

    mrbc_value ret = {.tt = GURU_TT_CLASS};
    ret.cls = cls;

    regs[ra] = ret;

    return 0;
}

//================================================================
/*!@brief
  Execute OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_exec(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_value rcv = regs[ra];						// receiver

    _push_state(vm, 0);								// push call stack

    vm->state->pc 	 = 0;							// switch context to callee
    vm->state->irep  = vm->irep->ilist[rb];			// fetch designated irep
    vm->state->reg 	 += ra;							// shift regfile (for local stack)
    vm->state->klass = mrbc_get_class_by_object(&rcv);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_method(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    if (regs[ra].tt != GURU_TT_CLASS) {
    	console_str("?op_method");
    	return 0;
    }

    mrbc_class 	*cls 	= regs[ra].cls;
    mrbc_proc 	*proc   = regs[ra+1].proc;
    mrbc_irep  	*irep 	= GET_IREP(vm);
    mrbc_sym   	sym_id  = _get_symid(irep->sym, rb);

    // check same name method
    mrbc_proc 	*p  = cls->procs;
    void 		*pp = &cls->procs;
    while (p != NULL) {
    	if (p->sym_id==sym_id) break;
    	pp = &p->next;
    	p  = p->next;
    }
    if (p) {	// found?
    	*((mrbc_proc**)pp) = p->next;
    	if (!IS_C_FUNC(p)) {
    		mrbc_value v = {.tt = GURU_TT_PROC};
    		v.proc = p;
    		mrbc_release(&v);
        }
    }

    // add proc to class
    proc->flag   &= ~GURU_PROC_C_FUNC;
    proc->sym_id = sym_id;
#ifdef GURU_DEBUG
    proc->name  = _get_symbol(irep->sym, rb);
#endif
    proc->next   = cls->procs;
    cls->procs   = proc;

    regs[ra+1].tt = GURU_TT_EMPTY;

    return 0;
}

//================================================================
/*!@brief
  Execute OP_TCLASS

  R(A) := target_class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__ int
op_tclass(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    _RA_T(GURU_TT_CLASS, cls=vm->state->klass);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_ABORT)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval -1  No error and exit from vm.
*/
__GURU__ int
op_stop(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
	vm->run = 0;
    return -1;		// exit guru_op loop
}

__GURU__ int
op_abort(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
	return -1;		// exit guru_op loop
}

__GURU__ int
guru_op(mrbc_vm *vm)
{
	uint32_t   code   = GET_BYTECODE(vm);
	int        opcode = GET_OPCODE(code);
	mrbc_value *regs  = vm->state->reg;
	int 	   ret;

    vm->state->pc++;		// advance program counter, ready for next cycle
    switch (opcode) {
    // LOAD,STORE
    case OP_LOADL:      ret = op_loadl     (vm, code, regs); break;
    case OP_LOADI:      ret = op_loadi     (vm, code, regs); break;
    case OP_LOADSYM:    ret = op_loadsym   (vm, code, regs); break;
    case OP_LOADNIL:    ret = op_loadnil   (vm, code, regs); break;
    case OP_LOADSELF:   ret = op_loadself  (vm, code, regs); break;
    case OP_LOADT:      ret = op_loadt     (vm, code, regs); break;
    case OP_LOADF:      ret = op_loadf     (vm, code, regs); break;
    case OP_GETGLOBAL:  ret = op_getglobal (vm, code, regs); break;
    case OP_SETGLOBAL:  ret = op_setglobal (vm, code, regs); break;
    case OP_GETIV:      ret = op_getiv     (vm, code, regs); break;
    case OP_SETIV:      ret = op_setiv     (vm, code, regs); break;
    case OP_GETCONST:   ret = op_getconst  (vm, code, regs); break;
    case OP_SETCONST:   ret = op_setconst  (vm, code, regs); break;
    case OP_GETUPVAR:   ret = op_getupvar  (vm, code, regs); break;
    case OP_SETUPVAR:   ret = op_setupvar  (vm, code, regs); break;
    // BRANCH
    case OP_JMP:        ret = op_jmp       (vm, code, regs); break;
    case OP_JMPIF:      ret = op_jmpif     (vm, code, regs); break;
    case OP_JMPNOT:     ret = op_jmpnot    (vm, code, regs); break;
    case OP_SEND:       ret = op_send      (vm, code, regs); break;
    case OP_SENDB:      ret = op_send      (vm, code, regs); break;  // reuse
    case OP_CALL:       ret = op_call      (vm, code, regs); break;
    case OP_ENTER:      ret = op_enter     (vm, code, regs); break;
    case OP_RETURN:     ret = op_return    (vm, code, regs); break;
    case OP_BLKPUSH:    ret = op_blkpush   (vm, code, regs); break;
    // ALU
    case OP_MOVE:       ret = op_move      (vm, code, regs); break;
    case OP_ADD:        ret = op_add       (vm, code, regs); break;
    case OP_ADDI:       ret = op_addi      (vm, code, regs); break;
    case OP_SUB:        ret = op_sub       (vm, code, regs); break;
    case OP_SUBI:       ret = op_subi      (vm, code, regs); break;
    case OP_MUL:        ret = op_mul       (vm, code, regs); break;
    case OP_DIV:        ret = op_div       (vm, code, regs); break;
    case OP_EQ:         ret = op_eq        (vm, code, regs); break;
    case OP_LT:         ret = op_lt        (vm, code, regs); break;
    case OP_LE:         ret = op_le        (vm, code, regs); break;
    case OP_GT:         ret = op_gt        (vm, code, regs); break;
    case OP_GE:         ret = op_ge        (vm, code, regs); break;
#if GURU_USE_STRING
    case OP_STRING:     ret = op_string    (vm, code, regs); break;
    case OP_STRCAT:     ret = op_strcat    (vm, code, regs); break;
#endif
#if GURU_USE_ARRAY
    case OP_ARRAY:      ret = op_array     (vm, code, regs); break;
    case OP_HASH:       ret = op_hash      (vm, code, regs); break;
    case OP_RANGE:      ret = op_range     (vm, code, regs); break;
#endif
    // BRANCH
    case OP_LAMBDA:     ret = op_lambda    (vm, code, regs); break;
    case OP_CLASS:      ret = op_class     (vm, code, regs); break;
    case OP_EXEC:       ret = op_exec      (vm, code, regs); break;
    case OP_METHOD:     ret = op_method    (vm, code, regs); break;
    case OP_TCLASS:     ret = op_tclass    (vm, code, regs); break;
    // EXEC
    case OP_STOP:       ret = op_stop      (vm, code, regs); break;
    case OP_ABORT:      ret = op_abort     (vm, code, regs); break;  // reuse
    case OP_NOP:        ret = op_nop       (vm, code, regs); break;
    default:
    	console_str("?OP=");
    	console_int(opcode);
    	console_str("\n");
    	ret = 0;
    	break;
    }
    return ret;
}



