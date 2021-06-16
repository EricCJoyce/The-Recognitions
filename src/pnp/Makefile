../../pnp: camera.o descriptor.o extractor.o matcher.o pnp_config.o pnp_solver.o signature.o pnp.o
	g++ -o ../../pnp camera.o descriptor.o extractor.o matcher.o pnp_config.o pnp_solver.o signature.o pnp.o `pkg-config --cflags --libs opencv`

camera.o: camera.cpp camera.h
	g++ -std=c++11 -c -Wall camera.cpp

descriptor.o: descriptor.cpp descriptor.h
	g++ -std=c++11 -c -Wall descriptor.cpp

extractor.o: extractor.cpp extractor.h descriptor.cpp descriptor.h pnp_config.cpp pnp_config.h
	g++ -std=c++11 -c -Wall descriptor.cpp
	g++ -std=c++11 -c -Wall pnp_config.cpp
	g++ -std=c++11 -c -Wall extractor.cpp

matcher.o: extractor.cpp extractor.h descriptor.cpp descriptor.h pnp_config.cpp pnp_config.h signature.cpp signature.h matcher.cpp matcher.h
	g++ -std=c++11 -c -Wall descriptor.cpp
	g++ -std=c++11 -c -Wall pnp_config.cpp
	g++ -std=c++11 -c -Wall extractor.cpp
	g++ -std=c++11 -c -Wall signature.cpp
	g++ -std=c++11 -c -Wall matcher.cpp

pnp_config.o: descriptor.cpp descriptor.h pnp_config.cpp pnp_config.h
	g++ -std=c++11 -c -Wall descriptor.cpp
	g++ -std=c++11 -c -Wall pnp_config.cpp

pnp_solver.o: pnp_solver.cpp pnp_solver.h
	g++ -std=c++11 -c -Wall pnp_solver.cpp

signature.o: descriptor.cpp descriptor.h signature.cpp signature.h
	g++ -std=c++11 -c -Wall descriptor.cpp
	g++ -std=c++11 -c -Wall signature.cpp

pnp.o: pnp.cpp
	g++ -std=c++11 -c -Wall pnp.cpp
