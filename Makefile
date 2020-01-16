CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0 -no-pie
CXXFLAGS_SANITIZE=$(CXXFLAGS) -O0 -g3 \
				  -fsanitize=address,undefined -fno-omit-frame-pointer \
				  -fno-optimize-sibling-calls
CXXFLAGS_RELEASE=$(CXXFLAGS) -O3 -march=native -g3
INC=-I picojson/ -I TFHEpp/include/ \
	-I TFHEpp/thirdparties/randen/ -I TFHEpp/thirdparties/spqlios/
LIB=-lpthread \
	-L TFHEpp/build/src/ -ltfhe++ \
	-L TFHEpp/build/thirdparties/randen/ -lranden \
	-L TFHEpp/build/thirdparties/spqlios/ -lspqlios

main: main.cpp main.hpp plain.hpp tfhepp.hpp
	clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB)
