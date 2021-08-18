../../make_signatures/extract: extract.o
	g++ -o ../../make_signatures/extract descriptor.o extractor.o pnp_config.o extract.o `pkg-config --cflags --libs opencv`

descriptor.o: descriptor.cpp descriptor.h
	g++ -std=c++11 -c -Wall descriptor.cpp

extractor.o: extractor.cpp extractor.h
	g++ -std=c++11 -c -Wall extractor.cpp

pnp_config.o: pnp_config.cpp pnp_config.h
	g++ -std=c++11 -c -Wall pnp_config.cpp

extract.o: extract.cpp
	g++ -std=c++11 -c -Wall extract.cpp
