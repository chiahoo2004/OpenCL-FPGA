g: clean gaussian
b: clean bilateral

bilateral:
	g++ -std=c++11 -O3 -o bilateral bilateral.cpp bilateralfilter.cpp -lglog -lIL
gaussian:
	g++ -std=c++11 -O3 -o gaussian gaussian.cpp gaussianfilter.cpp -lglog -lIL
clean:
	rm -f *.o *.bmp bilateral gaussian
