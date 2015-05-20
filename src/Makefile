gu: clean guided
ga: clean gaussian
b: clean bilateral

guided:
	g++ -std=c++11 -O2 -o guided guided.cpp guidedfilter.cpp enhancement.cpp utilities.cpp filter.cpp -lglog -lIL
bilateral:
	g++ -std=c++11 -O2 -o bilateral bilateral.cpp bilateralfilter.cpp enhancement.cpp utilities.cpp filter.cpp -lglog -lIL
gaussian:
	g++ -std=c++11 -O2 -o gaussian gaussian.cpp gaussianfilter.cpp enhancement.cpp utilities.cpp filter.cpp -lglog -lIL
clean:
	rm -f *.o *.bmp guided bilateral gaussian
