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
#include <stdio.h>

#include "alloc.h"
#include "instance.h"
#include "static.h"
#include "symbol.h"
#include "global.h"

#include "console.h"

#include "opcode.h"
#include "load.h"
#include "vm.h"
#include "class.h"

#if MRBC_USE_STRING
#include "c_string.h"
#endif
#if MRBC_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

//================================================================
/*! get sym[n] from symbol table in irep

  @param  p	Pointer to IREP SYMS section.
  @param  n	n th
  @return	symbol name string
*/
__GURU__
const char * mrbc_get_symbol(const uint8_t *p, int n)
{
    int cnt = _bin_to_uint32(p);
    if (n >= cnt) return 0;
    p += 4;
    while (n > 0) {
        uint16_t s = _bin_to_uint16(p);
        p += 2+s+1;   // size(2 bytes) + symbol len + '\0'
        n--;
    }
    return (char *)p+2;  // skip size(2 bytes)
}

__GURU__
mrbc_sym mrbc_get_symid(const uint8_t *p, int n)
{
	const char *sym_name = mrbc_get_symbol(p, n);
    return name2symid(sym_name);
}

//================================================================
/*! get callee name

  @param  vm	Pointer to VM
  @return	string
*/
__GURU__
const char *mrbc_get_callee_name(mrbc_vm *vm)
{
    uint32_t code = _bin_to_uint32(vm->pc_irep->code + (vm->pc - 1) * 4);
    
    int rb = GETARG_B(code);  // index of method sym

    return mrbc_get_symbol(vm->pc_irep->sym, rb);
}

//================================================================
/*!@brief

 */
__GURU__
void not_supported(void)
{
    console_str("Not supported!\n");
}

// Call a method
// v[0]: receiver
// v[1..]: params
//================================================================
/*!@brief
  call a method with params

  @param  vm		pointer to vm
  @param  name		method name
  @param  v		receiver and params
  @param  argc		num of params
*/
__GURU__
void mrbc_funcall(mrbc_vm *vm, const char *name, mrbc_value *v, int argc)
{
    mrbc_sym  sym_id = name2symid(name);
    mrbc_proc *m     = mrbc_get_class_method(v[0], sym_id);

    if (m==0) return;   	// no method

    mrbc_callinfo *ci = (mrbc_callinfo *)mrbc_alloc(sizeof(mrbc_callinfo));

    ci->reg		= vm->reg;
    ci->pc_irep = vm->pc_irep;
    ci->pc   	= vm->pc;
    ci->argc 	= 0;
    ci->klass	= vm->klass;

    ci->prev = vm->calltop;	// push call stack
    vm->calltop = ci;

    vm->pc_irep = m->irep;	// target irep
    vm->pc 		= 0;
    // new regs
    vm->reg 	+= 2;   	// recv and symbol
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__
void mrbc_push_callinfo(mrbc_vm *vm, int argc)
{
    mrbc_callinfo *ci = (mrbc_callinfo *)mrbc_alloc(sizeof(mrbc_callinfo));

    ci->reg 	= vm->reg;
    ci->pc_irep = vm->pc_irep;
    ci->pc 		= vm->pc;
    ci->argc 	= argc;
    ci->klass 	= vm->klass;

    // push call stack
    ci->prev 	= vm->calltop;
    vm->calltop = ci;
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__
void mrbc_pop_callinfo(mrbc_vm *vm)
{
    mrbc_callinfo *ci = vm->calltop;
    
    vm->calltop = ci->prev;
    vm->reg  	= ci->reg;
    vm->pc_irep = ci->pc_irep;
    vm->pc      = ci->pc;
    vm->klass  	= ci->klass;
    
    // free callinfo
    mrbc_free(ci);
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
__GURU__
int op_nop(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
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
__GURU__
int op_move(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    mrbc_release(&regs[ra]);

    mrbc_dup(&regs[rb]);
    regs[ra] = regs[rb];

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
__GURU__
int op_loadl(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_release(&regs[ra]);

    // regs[ra] = vm->pc_irep->pools[rb];

    mrbc_object *pool_obj = vm->pc_irep->pools[rb];
    regs[ra] = *pool_obj;

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
__GURU__
int op_loadi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    regs[ra].tt = MRBC_TT_FIXNUM;
    regs[ra].i = GETARG_sBx(code);

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
__GURU__
int op_loadsym(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    regs[ra].tt = MRBC_TT_SYMBOL;
    regs[ra].i = mrbc_get_symid(vm->pc_irep->sym, rb);

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
__GURU__
int op_loadnil(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    regs[ra].tt = MRBC_TT_NIL;

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
__GURU__
int op_loadself(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_release(&regs[ra]);
    mrbc_dup(&regs[0]);       // TODO: Need?
    regs[ra] = regs[0];

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
__GURU__
int op_loadt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    regs[ra].tt = MRBC_TT_TRUE;

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
__GURU__
int op_loadf(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    regs[ra].tt = MRBC_TT_FALSE;

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
__GURU__
int op_getglobal(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_release(&regs[ra]);

    regs[ra] = global_object_get(mrbc_get_symid(vm->pc_irep->sym, rb));

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
__GURU__
int op_setglobal(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_sym sym_id = mrbc_get_symid(vm->pc_irep->sym, rb);

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
__GURU__
int op_getiv(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const char *sym_name = mrbc_get_symbol(vm->pc_irep->sym, rb);
    mrbc_sym sym_id = name2symid(sym_name+1);	// skip '@'

    mrbc_value val = mrbc_instance_getiv(&regs[0], sym_id);

    mrbc_release(&regs[ra]);
    regs[ra] = val;

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
__GURU__
int op_setiv(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const char *sym_name = mrbc_get_symbol(vm->pc_irep->sym, rb);
    mrbc_sym sym_id = name2symid(sym_name+1);	// skip '@'

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
__GURU__
int op_getconst(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_release(&regs[ra]);
    regs[ra] = const_object_get(mrbc_get_symid(vm->pc_irep->sym, rb));

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
__GURU__
int op_setconst(mrbc_vm *vm, uint32_t code, mrbc_value *regs) {
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    const_object_add(mrbc_get_symid(vm->pc_irep->sym, rb), &regs[ra]);

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
__GURU__
int op_getupvar(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   // UP
    mrbc_callinfo *ci = vm->calltop;

    // find callinfo
    int n = rc * 2 + 1;
    while (n > 0){
        ci = ci->prev;
        n--;
    }

    mrbc_value *up_regs = ci->reg;

    mrbc_release(&regs[ra]);
    mrbc_dup(&up_regs[rb]);
    regs[ra] = up_regs[rb];

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
__GURU__
int op_setupvar(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);   // UP
    mrbc_callinfo *ci = vm->calltop;

    // find callinfo
    int n = rc * 2 + 1;
    while (n > 0){
        ci = ci->prev;
        n--;
    }

    mrbc_value *up_regs = ci->reg;

    mrbc_release(&up_regs[rb]);
    mrbc_dup(&regs[ra]);
    up_regs[rb] = regs[ra];

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
__GURU__
int op_jmp(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    vm->pc += GETARG_sBx(code) - 1;
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
__GURU__
int op_jmpif (mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt > MRBC_TT_FALSE) {
        vm->pc += GETARG_sBx(code) - 1;
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
__GURU__
int op_jmpnot(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    if (regs[GETARG_A(code)].tt <= MRBC_TT_FALSE) {
        vm->pc += GETARG_sBx(code) - 1;
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
__GURU__
int op_send(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);  // index of method sym
    int rc = GETARG_C(code);  // number of params
    mrbc_value recv = regs[ra];

    // Block param
    int bidx = ra + rc + 1;
    switch(GET_OPCODE(code)) {
    case OP_SEND:
        // set nil
        mrbc_release(&regs[bidx]);
        regs[bidx].tt = MRBC_TT_NIL;
        break;

    case OP_SENDB:
        // set Proc object
        if (regs[bidx].tt != MRBC_TT_NIL && regs[bidx].tt != MRBC_TT_PROC){
            // TODO: fix the following behavior
            // convert to Proc ?
            // raise exceprion in mruby/c ?
            return 0;
        }
        break;

    default:
        break;
    }

	mrbc_sym  sym_id = mrbc_get_symid(vm->pc_irep->sym, rb);
    mrbc_proc *m 	 = (mrbc_proc *)mrbc_get_class_method(recv, sym_id);
#ifdef MRBC_DEBUG
	const char *sym_name = mrbc_get_symbol(vm->pc_irep->sym, rb);
#endif

    if (m==0) {
    	console_str("func?:");
    	console_str(sym_name);
    	console_str("\n");
    	return 0;		// method not found
    }

    if (m->c_func) {				// m is a C function
        m->func(regs + ra, rc);

        if ((void (*))m->func==(void (*))c_proc_call) return 0;

        int release_reg = ra+1;
        while (release_reg <= bidx) {
            mrbc_release(&regs[release_reg]);
            release_reg++;
        }
    }
    else {							// m is a Ruby function
    	mrbc_push_callinfo(vm, rc);	// append callinfo list

    	vm->pc_irep = m->irep;		// call into target context
    	vm->pc = 0;					// call into target context
    	vm->reg += ra;				// add call stack (new register)
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
__GURU__
int op_call(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    mrbc_push_callinfo(vm, 0);

    // jump to proc
    vm->pc = 0;
    vm->pc_irep = regs[0].proc->irep;

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
__GURU__
int op_enter(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    mrbc_callinfo *callinfo = vm->calltop;
    uint32_t enter_param = GETARG_Ax(code);
    int def_args = (enter_param >> 13) & 0x1f;  // default args
    int argc = (enter_param >> 18) & 0x1f;      // given args
    if (def_args > 0){
        vm->pc += callinfo->argc - argc;
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
__GURU__
int op_return(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    // return value
    int ra = GETARG_A(code);

    mrbc_release(&regs[0]);
    regs[0] = regs[ra];
    regs[ra].tt = MRBC_TT_EMPTY;

    // restore irep,pc,regs
    mrbc_callinfo *ci = vm->calltop;

    vm->calltop	= ci->prev;
    vm->reg 	= ci->reg;
    vm->pc_irep = ci->pc_irep;
    vm->pc 		= ci->pc;
    vm->klass 	= ci->klass;

    // clear stacked arguments
    for (int i = 1; i <= ci->argc; i++) {
        mrbc_release(&regs[i]);
    }

    // release callinfo
    mrbc_free(ci);

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
__GURU__
int op_blkpush(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_value *stack = regs + 1;

    if (stack[0].tt==MRBC_TT_NIL){
        return -1;  // EYIELD
    }

    mrbc_release(&regs[ra]);
    mrbc_dup(stack);
    regs[ra] = stack[0];

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
__GURU__
int op_add(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i += regs[ra+1].i;
            return 0;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {	// in case of Fixnum, Float
            regs[ra].tt = MRBC_TT_FLOAT;
            regs[ra].f = regs[ra].i + regs[ra+1].f;
            return 0;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f += regs[ra+1].i;
            return 0;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {	// in case of Float, Float
            regs[ra].f += regs[ra+1].f;
            return 0;
        }
#endif
    }
    // other case
    op_send(vm, code, regs);
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
__GURU__
int op_addi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        regs[ra].i += GETARG_C(code);
        return 0;
    }

#if MRBC_USE_FLOAT
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        regs[ra].f += GETARG_C(code);
        return 0;
    }
#endif

    not_supported();
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
__GURU__
int op_sub(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i -= regs[ra+1].i;
            return 0;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {		// in case of Fixnum, Float
            regs[ra].tt = MRBC_TT_FLOAT;
            regs[ra].f = regs[ra].i - regs[ra+1].f;
            return 0;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f -= regs[ra+1].i;
            return 0;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {		// in case of Float, Float
            regs[ra].f -= regs[ra+1].f;
            return 0;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
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
__GURU__
int op_subi(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        regs[ra].i -= GETARG_C(code);
        return 0;
    }

#if MRBC_USE_FLOAT
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        regs[ra].f -= GETARG_C(code);
        return 0;
    }
#endif

    not_supported();
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
__GURU__
int op_mul(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i *= regs[ra+1].i;
            return 0;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {	// in case of Fixnum, Float
            regs[ra].tt = MRBC_TT_FLOAT;
            regs[ra].f = regs[ra].i * regs[ra+1].f;
            return 0;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f *= regs[ra+1].i;
            return 0;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {	// in case of Float, Float
            regs[ra].f *= regs[ra+1].f;
            return 0;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
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
__GURU__
int op_div(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Fixnum, Fixnum
            regs[ra].i /= regs[ra+1].i;
            return 0;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {		// in case of Fixnum, Float
            regs[ra].tt = MRBC_TT_FLOAT;
            regs[ra].f = regs[ra].i / regs[ra+1].f;
            return 0;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {	// in case of Float, Fixnum
            regs[ra].f /= regs[ra+1].i;
            return 0;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {		// in case of Float, Float
            regs[ra].f /= regs[ra+1].f;
            return 0;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
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
__GURU__
int op_eq(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result = mrbc_compare(&regs[ra], &regs[ra+1]);

    mrbc_release(&regs[ra+1]);
    mrbc_release(&regs[ra]);
    regs[ra].tt = result ? MRBC_TT_FALSE : MRBC_TT_TRUE;

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
__GURU__
int op_lt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].i < regs[ra+1].i;	// in case of Fixnum, Fixnum
            goto DONE;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].i < regs[ra+1].f;	// in case of Fixnum, Float
            goto DONE;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].f < regs[ra+1].i;	// in case of Float, Fixnum
            goto DONE;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].f < regs[ra+1].f;	// in case of Float, Float
            goto DONE;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
    mrbc_release(&regs[ra+1]);
    return 0;

DONE:
    regs[ra].tt = result ? MRBC_TT_TRUE : MRBC_TT_FALSE;
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
__GURU__
int op_le(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].i <= regs[ra+1].i;	// in case of Fixnum, Fixnum
            goto DONE;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].i <= regs[ra+1].f;	// in case of Fixnum, Float
            goto DONE;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].f <= regs[ra+1].i;	// in case of Float, Fixnum
            goto DONE;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].f <= regs[ra+1].f;	// in case of Float, Float
            goto DONE;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
    mrbc_release(&regs[ra+1]);
    return 0;

DONE:
    regs[ra].tt = result ? MRBC_TT_TRUE : MRBC_TT_FALSE;
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
__GURU__
int op_gt(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].i > regs[ra+1].i;	// in case of Fixnum, Fixnum
            goto DONE;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].i > regs[ra+1].f;	// in case of Fixnum, Float
            goto DONE;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].f > regs[ra+1].i;	// in case of Float, Fixnum
            goto DONE;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].f > regs[ra+1].f;	// in case of Float, Float
            goto DONE;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
    mrbc_release(&regs[ra+1]);
    return 0;

DONE:
    regs[ra].tt = result ? MRBC_TT_TRUE : MRBC_TT_FALSE;
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
__GURU__
int op_ge(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int result;

    if (regs[ra].tt==MRBC_TT_FIXNUM) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].i >= regs[ra+1].i;	// in case of Fixnum, Fixnum
            goto DONE;
        }
#if MRBC_USE_FLOAT
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].i >= regs[ra+1].f;	// in case of Fixnum, Float
            goto DONE;
        }
    }
    if (regs[ra].tt==MRBC_TT_FLOAT) {
        if (regs[ra+1].tt==MRBC_TT_FIXNUM) {
            result = regs[ra].f >= regs[ra+1].i;	// in case of Float, Fixnum
            goto DONE;
        }
        if (regs[ra+1].tt==MRBC_TT_FLOAT) {
            result = regs[ra].f >= regs[ra+1].f;	// in case of Float, Float
            goto DONE;
        }
#endif
    }

    // other case
    op_send(vm, code, regs);
    mrbc_release(&regs[ra+1]);
    return 0;

DONE:
    regs[ra].tt = result ? MRBC_TT_TRUE : MRBC_TT_FALSE;
    return 0;
}

#if MRBC_USE_STRING
//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__
int op_string(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if MRBC_USE_STRING
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);
    mrbc_object *pool_obj = vm->pc_irep->pools[rb];

    /* CAUTION: pool_obj->sym - 2. see IREP POOL structure. */
    int len = _bin_to_uint16(pool_obj->sym - 2);
    mrbc_value value = mrbc_string_new(pool_obj->sym, len);
    if (value.str==NULL) return -1;		// ENOMEM

    mrbc_release(&regs[ra]);
    regs[ra] = value;

#else
    not_supported();
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
__GURU__
int op_strcat(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
#if MRBC_USE_STRING
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    // call "to_s"
    mrbc_sym sym_id = name2symid("to_s");
    mrbc_proc *m;
    m = mrbc_get_class_method(regs[ra], sym_id);
    if (m && m->c_func){
        m->func(regs+ra, 0);
    }
    m = mrbc_get_class_method(regs[rb], sym_id);
    if (m && m->c_func){
        m->func(regs+rb, 0);
    }

    mrbc_value v = mrbc_string_add(&regs[ra], &regs[rb]);
    mrbc_release(&regs[ra]);
    regs[ra] = v;

#else
    not_supported();
#endif
    return 0;
}
#endif

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
#if MRBC_USE_ARRAY
__GURU__
int op_array(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);

    mrbc_value value = mrbc_array_new(vm, rc);
    if (value.array==NULL) return -1;	// ENOMEM

    MEMCPY(value.array->data, &regs[rb], sizeof(mrbc_value) * rc);
    MEMSET(&regs[rb], 0, sizeof(mrbc_value) * rc);
    value.array->n_stored = rc;

    mrbc_release(&regs[ra]);
    regs[ra] = value;

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
__GURU__
int op_hash(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);

    mrbc_value value = mrbc_hash_new(vm, rc);
    if (value.hash==NULL) return -1;	// ENOMEM

    rc *= 2;
    MEMCPY(value.hash->data, &regs[rb], sizeof(mrbc_value) * rc);
    MEMSET(&regs[rb], 0, sizeof(mrbc_value) * rc);
    value.hash->n_stored = rc;

    mrbc_release(&regs[ra]);
    regs[ra] = value;

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
__GURU__
int op_range(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    int rc = GETARG_C(code);

    mrbc_dup(&regs[rb]);
    mrbc_dup(&regs[rb+1]);

    mrbc_value value = mrbc_range_new(vm, &regs[rb], &regs[rb+1], rc);
    if (value.range==NULL) return -1;		// ENOMEM

    mrbc_release(&regs[ra]);
    regs[ra] = value;

    return 0;
}
#endif

//================================================================
/*!@brief
  Execute OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  vm->regs + vm->reg_top
  @retval 0  No error.
*/
__GURU__
int op_lambda(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_b(code);      	// sequence position in irep list
    // int c = GETARG_C(code);    	// TODO: Add flags support for OP_LAMBDA
    mrbc_proc *proc = (mrbc_proc *)mrbc_proc_alloc("(lambda)");

    proc->c_func = 0;				// IREP
    proc->irep = vm->pc_irep->reps[rb];

    mrbc_release(&regs[ra]);
    regs[ra].tt = MRBC_TT_PROC;
    regs[ra].proc = proc;

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
__GURU__
int op_class(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);

    mrbc_class *super = (regs[ra+1].tt==MRBC_TT_CLASS) ? regs[ra+1].cls : mrbc_class_object;

    mrbc_irep *cur_irep  = vm->pc_irep;
    const char *sym_name = mrbc_get_symbol(cur_irep->sym, rb);
    mrbc_class *cls 	 = (mrbc_class *)mrbc_define_class(sym_name, super);

    mrbc_value ret = {.tt = MRBC_TT_CLASS};
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
__GURU__
int op_exec(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_Bx(code);

    mrbc_value recv = regs[ra];

    // prepare callinfo
    mrbc_push_callinfo(vm, 0);

    // target irep
    vm->pc = 0;
    vm->pc_irep = vm->irep->reps[rb];

    // new regs
    vm->reg += ra;
    vm->klass = mrbc_get_class_by_object(&recv);

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
__GURU__
int op_method(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);
    int rb = GETARG_B(code);
    mrbc_proc *proc = regs[ra+1].proc;

    if (regs[ra].tt==MRBC_TT_CLASS) {
        mrbc_class *cls = regs[ra].cls;

        // sym_id : method name
        mrbc_irep *cur_irep = vm->pc_irep;
        mrbc_sym sym_id = mrbc_get_symid(cur_irep->sym, rb);

        // check same name method
        mrbc_proc *p = cls->procs;
        void *pp = &cls->procs;
        while (p != NULL) {
            if (p->sym_id==sym_id) break;
            pp = &p->next;
            p = p->next;
        }
        if (p) {
            // found it.
            *((mrbc_proc**)pp) = p->next;
            if (!p->c_func) {
                mrbc_value v = {.tt = MRBC_TT_PROC};
                v.proc = p;
                mrbc_release(&v);
            }
        }

        // add proc to class
        proc->c_func = 0;
        proc->sym_id = sym_id;
#ifdef MRBC_DEBUG
        proc->names = mrbc_get_symbol(cur_irep->sym, rb);
#endif
        proc->next = cls->procs;
        cls->procs = proc;

        regs[ra+1].tt = MRBC_TT_EMPTY;
    }

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
__GURU__
int op_tclass(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    int ra = GETARG_A(code);

    mrbc_release(&regs[ra]);
    regs[ra].tt = MRBC_TT_CLASS;
    regs[ra].cls = vm->klass;

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
__GURU__
int op_stop(mrbc_vm *vm, uint32_t code, mrbc_value *regs)
{
    if (GET_OPCODE(code)==OP_STOP) {
        int i;
        for(i = 0; i < MAX_REGS_SIZE; i++) {
            mrbc_release(&vm->regfile[i]);
        }
    }
    vm->run = 0;

    return -1;
}

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__
void _mrbc_vm_begin(mrbc_vm *vm)
{
    vm->pc_irep = vm->irep;
    vm->pc 		= 0;
    vm->reg 	= vm->regfile;

    MEMSET((uint8_t *)vm->regfile, 0, sizeof(vm->regfile));	// clean up registers

    // set self to reg[0]
    vm->regfile[0].tt  	= MRBC_TT_CLASS;
    vm->regfile[0].cls 	= mrbc_class_object;
    vm->calltop  		= NULL;

    // target_class
    vm->klass = mrbc_class_object;
    vm->run   = 1;
}

//================================================================
/*!@brief
  VM finalizer.

  @param  vm  Pointer to VM
*/
__GURU__
void _mrbc_vm_end(mrbc_vm *vm)
{
    mrbc_free_all();
}

//================================================================
/*!@brief
  Fetch a bytecode and execute

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__
int _mrbc_vm_exec(mrbc_vm *vm)
{
    int ret       = 0;
    int opcode    = 0;
    uint32_t code = 0;
    mrbc_value *regs;

    console_str("vm*start...\n");
    do {
        code   = _bin_to_uint32(vm->pc_irep->code + vm->pc * 4);	// get next bytecode
        opcode = GET_OPCODE(code);
        regs   = vm->reg;

        vm->pc++;

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
#if MRBC_USE_STRING
        case OP_STRING:     ret = op_string    (vm, code, regs); break;
        case OP_STRCAT:     ret = op_strcat    (vm, code, regs); break;
#endif
#if MRBC_USE_ARRAY      
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
        case OP_ABORT:      ret = op_stop      (vm, code, regs); break;  // reuse
        case OP_NOP:        ret = op_nop       (vm, code, regs); break;
        default:
            console_str("Skip OP=");
            console_int(opcode);
            console_str("\n");
            break;
        }
    } while (vm->run);

    console_str("vm*done!\n");
    return ret;
}

//================================================================
/*!@brief
  release mrbc_irep holds memory
*/
__GURU__
void _mrbc_free_ireplist(mrbc_irep *irep)
{
    // release pools.
    for(int i = 0; i < irep->plen; i++) {
        mrbc_free(irep->pools[i]);
    }
    if (irep->plen) mrbc_free(irep->pools);

    // release child ireps.
    for(int i = 0; i < irep->rlen; i++) {
        _mrbc_free_ireplist(irep->reps[i]);
    }
    if (irep->rlen) mrbc_free(irep->reps);

    mrbc_free(irep);
}

__global__
void _run_vm(mrbc_vm *vm)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	_mrbc_vm_begin(vm);

	int ret = _mrbc_vm_exec(vm);

	_mrbc_vm_end(vm);

	__syncthreads();
}

int guru_vm_init(guru_ses *ses)
{
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return -4;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);		// can also be done on host?
	cudaDeviceSynchronize();

#ifdef MRBC_DEBUG
	printf("guru loader:\n");
	dump_irep(vm->irep);
#endif
	ses->vm = (uint8_t *)vm;
	return 0;
}

int guru_vm_run(guru_ses *ses)
{
	int sz;
	cudaDeviceGetLimit((size_t *)&sz, cudaLimitStackSize);
	printf("defaultStackSize %d => %d\n", sz, sz*4);

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz*4);
	_run_vm<<<1,1>>>((mrbc_vm *)ses->vm);
	cudaDeviceSynchronize();

	return 0;
}

#ifdef MRBC_DEBUG
void dump_irep(mrbc_irep *irep)
{
	printf("\tnlocals=%d, nregs=%d, rlen=%d, ilen=%d, plen=%d\n",
			irep->nlocals,	irep->nregs, irep->rlen, irep->ilen, irep->plen);
	for (int i=0; i<irep->rlen; i++) {
		dump_irep(irep->reps[i]);
	}
}
#endif

