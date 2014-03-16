#include <Magick++.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>

#include "Hough.h"

using namespace std;
using namespace Magick;

int usage(string argv0) {
	cerr << "USAGE: " << argv0 << " <input_image> <output_image> [angles]" << endl;
	return 1;
}

int main(int argc, char **argv)
{
	if (argc < 3 || argc > 4)
		return usage(argv[0]);

	int angles = 20;
	if (argc > 3)
		angles = atoi(argv[3]);

	cout << "Processing `" << argv[1] << "' into `" << argv[2] << "' using " << angles << " directions" << endl;

	InitializeMagick(*argv);
	Image input;
	Color black("black");

	try {
		input.read(argv[1]);
		input.quantizeColorSpace(GRAYColorspace);
		input.quantizeColors(256);
		input.quantize();

		int width = input.size().width();
		int height = input.size().height();

		int n = width + height - 1;
		vector<int> v(n * angles * 2);

		hough_transform(width, height, input.getConstPixels(0, 0, width, height), angles, v.data());

		Image out(Geometry(2 * angles, n), "red");

		PixelPacket *pp = out.getPixels(0, 0, 2 * angles, n);

		for (int j = 0; j < n; j++) {
			for (int ia = 0; ia < 2 * angles; ia++)
				pp[j * 2 * angles + ia] = ColorGray(static_cast<double>(v[ia * n + j]) / n / 65535);
		}
		out.syncPixels();
		out.autoLevel();
		out.write(argv[2]);
	} catch (Exception &e) { 
		cerr << "Caught exception: " << e.what() << endl;
		return 1;
	}

	return 0;
}
