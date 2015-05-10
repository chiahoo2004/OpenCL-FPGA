gu: clean guided
ga: clean gaussian
b: clean bilateral

guided:
	g++ -std=c++11 -O3 -o guided guided.cpp guidedfilter.cpp -lglog -lIL
bilateral:
	g++ -std=c++11 -O3 -o bilateral bilateral.cpp bilateralfilter.cpp -lglog -lIL
gaussian:
	g++ -std=c++11 -O3 -o gaussian gaussian.cpp gaussianfilter.cpp -lglog -lIL
clean:
	rm -f *.o *.bmp guided bilateral gaussian
