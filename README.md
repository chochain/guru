* NSight project rebuild
+ rename ~/lib/guru to ~/lib/guru.bak
+ NSight
  + create new CUDA C++ project guru
  + project->build->separate compilation (instead of entire project)
+ copy ~/lib/guru.bak to ~/lib/guru
  + refresh project
+ NSight
  + set workspace to ~/lib
  + set to "exclude" directory property to ~/lib/guru/ext, ~/lib/guru/orig
  
