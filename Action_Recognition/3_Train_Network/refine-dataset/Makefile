all: ../clicker ../edit_artifacts
.PHONY: all

../clicker: clicker.cpp
	g++ clicker.cpp -o ../clicker `pkg-config --cflags --libs opencv`

../edit_artifacts: edit_artifacts.cpp
	g++ edit_artifacts.cpp -o ../edit_artifacts `pkg-config --cflags --libs opencv`
