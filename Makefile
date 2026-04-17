CXX=g++
CXXFLAGS= -I. -IRaP -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2
TARGET=dynamic_test
SRC=dynamic_test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET) 