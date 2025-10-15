#!/bin/bash

gcc -o passim.out -I./include -I/usr/include/libxml2 -fopenacc src/main.c src/config.c src/rf.c src/gis.c src/path.c src/pvt.c src/threadpool.c -Wno-implicit-function-declaration -lxml2 -lm -lpthread -lbsd -O3 && echo "built passim.out"

build_array() {
  [[ -z $1 ]] && { echo "Usage: build_array <name>"; return 1; }
  src="arrays/$1/impl.c"; out="$1.so"
  [[ -f $src ]] || { echo "Missing: $src"; return 1; }
  gcc -I./include -fPIC -shared -o "$out" arrays/common.c "$src" -lm && echo "+ built $out"
}

build_array ula
build_array uca