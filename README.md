* NSight project rebuild
+ rename ~/lib/guru to ~/lib/guru.bak
+ NSight
  + create new CUDA C++ project guru
  + project->build->separate compilation (instead of entire project)
+ copy ~/lib/guru.bak to ~/lib/guru
  + refresh project
+ NSight
  + set workspace to ~/lib
  + set to "exclude" directory property to ~/lib/guru/ext, ~/lib/guru/orig, ~/lib/guru/test
  + set to "exclude" src/alloc.cu, c_str_ascii.cu, cuda.cu, load_gpu.cu, puts.cu, symbol_orig.cu
  + Project->Properties->Build->Setting: CUDA tab-> select "Separate compilation"
  + Project->Properties->Build->Setting: Tool Setting tab-> tick Enable C++11 support
  + Project->Properties->Build: Behaviour tab-> tick Enable parallel build

