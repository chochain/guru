* NSight project rebuild
+ rename ~/lib/guru to ~/lib/guru.bak
+ NSight
  + create new CUDA C++ project guru
  + project->build->separate compilation (instead of entire project)
+ copy ~/lib/guru.bak to ~/lib/guru
  + refresh project
+ NSight
  + set workspace to ~/lib
  + Project->Properties->Build->Setting: CUDA tab-> select "Separate compilation"
  + Project->Properties->Build->Setting: Tool Setting tab-> tick Enable C++11 support
  + Project->Properties->Build: Behaviour tab-> tick Enable parallel build
  + set to "exclude" directory property to ~/lib/guru/ext, ~/lib/guru/orig, ~/lib/guru/test
  + set to "exclude" legacy code
    - alloc.cu, c_str_ascii.cu, cuda.cu, puts.cu, symbol_orig.cu
  + set to "exclude" console formatter
    - console.*, sprintf.*
  + set to "exclude" GPU-based RITE code loader
    - load_gpu.cu
  C only codebase: guru_config.h#GURU_CXX_CODEBASE == 0, to exclude 
    - state_cxx.cu, ucode_cxx.cu, vmx.cu, vm_cxx.cu, class_cxx.cu, debug_cxx.cu
  C++ codebase:    guru_config.h#GURU_CXX_CODEBASE == 1, to exclude
    - state.cu, ucode.cu, vm.cu, class.cu, debug.cu

