#%Module######################################################################
##
##      Project Module
##
proc ModulesHelp { } {
  puts stderr "\tThe TGFS-ISS Project Module\n"
  puts stderr "\tThis module loads PATHs and variables for accessing Verilator."
}


set distro [exec /bin/lsb_release -i -s]
if { $distro == "CentOS" && ![info exists ::env(PROJECT)] && ![info exists ::env(PCP_DIR)] } {
    puts stderr     "Don't forget to execute 'scl enable devtoolset-7 bash'"
}

module load tools/utilities
module load tools/cmake
module load tools/clang
module load tools/gcc-riscv64-unknown-elf/13

setenv PROJECT TGFS-ISS

