all: ../clicker
.PHONY: all

../clicker: clicker.cpp
	g++ clicker.cpp -o ../clicker `pkg-config --cflags --libs opencv`
