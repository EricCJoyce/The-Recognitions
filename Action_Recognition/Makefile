all: align dtw.cpython-35m-x86_64-linux-gnu.so
.PHONY: all

align: align.c
	gcc -Wall align.c -lm -o align

dtw.cpython-35m-x86_64-linux-gnu.so: DTW.FOR
	f2py3 -m dtw -c DTW.FOR