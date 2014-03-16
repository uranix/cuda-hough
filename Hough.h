#ifndef __HOUGH_H__
#define __HOUGH_H__

#include <Magick++.h>

void hough_transform(int width, int height, const Magick::PixelPacket *image, int angles, int *res);

#endif 
