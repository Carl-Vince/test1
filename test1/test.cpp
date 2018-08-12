/*#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>

using namespace  cv;

using namespace  std;



int main()
{
	Mat img = imread("earth.jpg", CV_LOAD_IMAGE_UNCHANGED);
	if(img.empty())
	{
		cout << "图像加载失败！" << endl;

		//system("pause");

		return -1;

	}

	//创建一个名字为MyWindow的窗口

	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

	//在MyWindow的窗中中显示存储在img中的图片

	imshow("MyWindow", img);

	//等待直到有键按下

	waitKey(0);

	//销毁MyWindow的窗口

	destroyWindow("MyWindow");
	cin.get( );
	cin.get();
	return 0;

}*/
/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 1 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

/*#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// test function that creates an image
cv::Mat function() {

	// create image
	cv::Mat ima(500, 500, CV_8U, 50);
	// return it
	return ima;
}

int main() {

	// create a new image made of 240 rows and 320 columns
	cv::Mat image1(240, 320, CV_8U, 100);
	// or:
	// cv::Mat image1(240,320,CV_8U,cv::Scalar(100));

	cv::imshow("Image", image1); // show the image
	cv::waitKey(0); // wait for a key pressed

					// re-allocate a new image
					// (only if size or type are different)
	image1.create(200, 200, CV_8U);
	image1 = 200;

	cv::imshow("Image", image1); // show the image
	cv::waitKey(0); // wait for a key pressed

					// create a red color image
					// channel order is BGR
	cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0,0,0));

	// or:
	// cv::Mat image2(cv::Size(320,240),CV_8UC3);
	// image2= cv::Scalar(0,0,255);

	cv::imshow("Image", image2); // show the image
	cv::waitKey(0); // wait for a key pressed

					// read an image
	cv::Mat image3 = cv::imread("earth.jpg");

	// all these images point to the same data block
	cv::Mat image4(image3);
	image1 = image3;

	// these images are new copies of the source image
	image3.copyTo(image2);
	cv::Mat image5 = image3.clone();

	// transform the image for testing
	cv::flip(image3, image3, 1);

	// check which images have been affected by the processing
	cv::imshow("Image 3", image3);
	cv::imshow("Image 1", image1);
	cv::imshow("Image 2", image2);
	cv::imshow("Image 4", image4);
	cv::imshow("Image 5", image5);
	cv::waitKey(0); // wait for a key pressed

					// get a gray-level image from a function
	cv::Mat gray = function();

	cv::imshow("Image", gray); // show the image
	cv::waitKey(0); // wait for a key pressed

					// read the image in gray scale
	image1 = cv::imread("puppy.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	// convert the image into a floating point image [0,1]
	image1.convertTo(image2, CV_32F, 1 / 255.0, 0.0);

	cv::imshow("Image", image2); // show the image

								 // Test cv::Matx
								 // a 3x3 matrix of double-precision
	cv::Matx33d matrix(3.0, 2.0, 1.0,
		2.0, 1.0, 3.0,
		1.0, 2.0, 3.0);
	// a 3x1 matrix (a vector)
	cv::Matx31d vector(5.0, 1.0, 3.0);
	// multiplication
	cv::Matx31d result = matrix*vector;

	std::cout << result;

	cv::waitKey(0); // wait for a key pressed
	return 0;
}*/
/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 1 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

/*#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main() {

	// define an image window
	cv::namedWindow("Image");

	// read the image 
	cv::Mat image = cv::imread("earth.jpg");

	// read the logo
	cv::Mat logo = cv::imread("timg.jfif");

	// define image ROI at image bottom-right
	cv::Mat imageROI(image,
		cv::Rect(image.cols - logo.cols, //ROI coordinates
			image.rows - logo.rows,
			logo.cols, logo.rows));// ROI size

								   // insert logo
	logo.copyTo(imageROI);

	cv::imshow("Image", image); // show the image
	cv::waitKey(0); // wait for a key pressed

					// re-read the original image
	image = cv::imread("puppy.bmp");

	// define image ROI at image bottom-right
	imageROI = image(cv::Rect(image.cols - logo.cols, image.rows - logo.rows,
		logo.cols, logo.rows));
	// or using ranges:
	// imageROI= image(cv::Range(image.rows-logo.rows,image.rows), 
	//                 cv::Range(image.cols-logo.cols,image.cols));

	// use the logo as a mask (must be gray-level)
	cv::Mat mask(logo);

	// insert by copying only at locations of non-zero mask
	logo.copyTo(imageROI, mask);

	cv::imshow("Image", image); // show the image
	cv::waitKey(0); // wait for a key pressed

	return 0;
}*/
/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 1 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

/*#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void onMouse(int event, int x, int y, int flags, void* param) {

	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);

	switch (event) {	// dispatch the event

	case cv::EVENT_LBUTTONDOWN: // mouse button down event

								// display pixel value at (x,y)
		std::cout << "at (" << x << "," << y << ") value is: "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
		break;
	}
}

int main() {

	cv::Mat image; // create an empty image
	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;

	// read the input image as a gray-scale image
	image = cv::imread("earth.jpg", cv::IMREAD_GRAYSCALE);

	if (image.empty()) {  // error handling
						  // no image has been created...
						  // possibly display an error message
						  // and quit the application 
		std::cout << "Error reading image..." << std::endl;
		return 0;
	}

	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;
	std::cout << "This image has "
		<< image.channels() << " channel(s)" << std::endl;

	// create image window named "My Image"
	cv::namedWindow("Original Image"); // define the window (optional)
	cv::imshow("Original Image", image); // show the image

										 // set the mouse callback for this image
	cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void*>(&image));

	cv::Mat result; // we create another empty image
	cv::flip(image, result, 1); // positive for horizontal
								// 0 for vertical,                     
								// negative for both

	cv::namedWindow("Output Image"); // the output window
	cv::imshow("Output Image", result);

	cv::waitKey(0); // 0 to indefinitely wait for a key pressed
					// specifying a positive value will wait for
					// the given amount of msec

	cv::imwrite("output.bmp", result); // save result

									   // create another image window named
	cv::namedWindow("Drawing on an Image"); // define the window

	cv::circle(image,              // destination image 
		cv::Point(155, 110), // center coordinate
		65,                 // radius  
		0,                  // color (here black)
		3);                 // thickness

	cv::putText(image,                   // destination image
		"This is a dog.",        // text
		cv::Point(40, 200),       // text position
		cv::FONT_HERSHEY_PLAIN,  // font type
		2.0,                     // font scale
		255,                     // text color (here white)
		2);                      // text thickness

	cv::imshow("Drawing on an Image", image); // show the image

	cv::waitKey(0); // 0 to indefinitely wait for a key pressed

	return 0;
}*/
/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 2 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

/*#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>

// Add salt noise to an image
void salt(cv::Mat image, int n) {

	// C++11 random number generator
	std::default_random_engine generator;
	std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
	std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

	int i, j;
	for (int k = 0; k<n; k++) {

		// random image coordinate
		i = randomCol(generator);
		j = randomRow(generator);

		if (image.type() == CV_8UC1) { // gray-level image

									   // single-channel 8-bit image
			image.at<uchar>(j, i) = 255;

		}
		else if (image.type() == CV_8UC3) { // color image

										// 3-channel image
			image.at<cv::Vec3b>(j, i)[0] = 46;
			image.at<cv::Vec3b>(j, i)[1] = 49;
			image.at<cv::Vec3b>(j, i)[2] = 57;

			// or simply:
			 //image.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 255, 255);
		}
	}
}

// This is an extra version of the function
// to illustrate the use of cv::Mat_
// works only for a 1-channel image
void salt2(cv::Mat image, int n) {

	// must be a gray-level image
	CV_Assert(image.type() == CV_8UC1);

	// C++11 random number generator
	std::default_random_engine generator;
	std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
	std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

	// use image with a Mat_ template
	cv::Mat_<uchar> img(image);

	//  or with references:
	//	cv::Mat_<uchar>& im2= reinterpret_cast<cv::Mat_<uchar>&>(image);

	int i, j;
	for (int k = 0; k<n; k++) {

		// random image coordinate
		i = randomCol(generator);
		j = randomRow(generator);

		// add salt
		img(j, i) = 255;
	}
}


int main()
{
	// open the image
	cv::Mat image = cv::imread("earth.jpg", 1);

	// call function to add noise
	salt(image, 66666);

	// display result
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	// write on disk
	cv::imwrite("salted.bmp", image);

	cv::waitKey();

	// test second version
	image = cv::imread("earth.jpg", 0);

	salt2(image, 300000);

	cv::namedWindow("Image");
	cv::imshow("Image", image);

	cv::waitKey();

	return 0;
}*/
/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 2 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// 1st version
// see recipe Scanning an image with pointers
void colorReduce(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line

	for (int j = 0; j<nl; j++) {

		// get the address of row j
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			data[i] = data[i] / div*div + div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// version with input/ouput images
// see recipe Scanning an image with pointers
void colorReduceIO(const cv::Mat &image, // input image
	cv::Mat &result,      // output image
	int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols; // number of columns
	int nchannels = image.channels(); // number of channels

									  // allocate output image if necessary
	result.create(image.rows, image.cols, image.type());

	for (int j = 0; j<nl; j++) {

		// get the addresses of input and output row j
		const uchar* data_in = image.ptr<uchar>(j);
		uchar* data_out = result.ptr<uchar>(j);

		for (int i = 0; i<nc*nchannels; i++) {

			// process each pixel ---------------------

			data_out[i] = data_in[i] / div*div + div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 1
// this version uses the derefere nce operator *
void colorReduce1(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {


			// process each pixel ---------------------

			*data++ = *data / div*div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 2
// this version uses the modulo operator
void colorReduce2(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			int v = *data;
			*data++ = v - v%div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 3
// this version uses a binary mask
void colorReduce3(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = 1 << (n - 1); // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++) {

			// process each pixel ---------------------

			*data &= mask;     // masking
			*data++ |= div2;   // add div/2

							   // end of pixel processing ----------------

		} // end of line
	}
}


// Test 4
// this version uses direct pointer arithmetic with a binary mask
void colorReduce4(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	int step = image.step; // effective width
						   // mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // get the pointer to the image buffer
	uchar *data = image.data;

	for (int j = 0; j<nl; j++) {

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*(data + i) &= mask;
			*(data + i) += div2;

			// end of pixel processing ----------------

		} // end of line

		data += step;  // next line
	}
}

// Test 5
// this version recomputes row size each time
void colorReduce5(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<image.cols * image.channels(); i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 6
// this version optimizes the case of continuous image
void colorReduce6(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line

	if (image.isContinuous()) {
		// then no padded pixels
		nc = nc*nl;
		nl = 1;  // it is now a 1D array
	}

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // this loop is executed only once
						   // in case of continuous images
	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 7
// this versions applies reshape on continuous image
void colorReduce7(cv::Mat image, int div = 64) {

	if (image.isContinuous()) {
		// no padded pixels
		image.reshape(1,   // new number of channels
			1); // new number of rows
	}
	// number of columns set accordingly

	int nl = image.rows; // number of lines
	int nc = image.cols*image.channels(); // number of columns

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 8
// this version processes the 3 channels inside the loop with Mat_ iterators
void colorReduce8(cv::Mat image, int div = 64) {

	// get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
	uchar div2 = div >> 1; // div2 = div/2

	for (; it != itend; ++it) {

		// process each pixel ---------------------

		(*it)[0] = (*it)[0] / div*div + div2;
		(*it)[1] = (*it)[1] / div*div + div2;
		(*it)[2] = (*it)[2] / div*div + div2;

		// end of pixel processing ----------------
	}
}

// Test 9
// this version uses iterators on Vec3b
void colorReduce9(cv::Mat image, int div = 64) {

	// get iterators
	cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itend = image.end<cv::Vec3b>();

	const cv::Vec3b offset(div / 2, div / 2, div / 2);

	for (; it != itend; ++it) {

		// process each pixel ---------------------

		*it = *it / div*div + offset;
		// end of pixel processing ----------------
	}
}

// Test 10
// this version uses iterators with a binary mask
void colorReduce10(cv::Mat image, int div = 64) {

	// div must be a power of 2
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();

	// scan all pixels
	for (; it != itend; ++it) {

		// process each pixel ---------------------

		(*it)[0] &= mask;
		(*it)[0] += div2;
		(*it)[1] &= mask;
		(*it)[1] += div2;
		(*it)[2] &= mask;
		(*it)[2] += div2;

		// end of pixel processing ----------------
	}
}

// Test 11
// this versions uses ierators from Mat_ 
void colorReduce11(cv::Mat image, int div = 64) {

	// get iterators
	cv::Mat_<cv::Vec3b> cimage = image;
	cv::Mat_<cv::Vec3b>::iterator it = cimage.begin();
	cv::Mat_<cv::Vec3b>::iterator itend = cimage.end();
	uchar div2 = div >> 1; // div2 = div/2

	for (; it != itend; it++) {

		// process each pixel ---------------------

		(*it)[0] = (*it)[0] / div*div + div2;
		(*it)[1] = (*it)[1] / div*div + div2;
		(*it)[2] = (*it)[2] / div*div + div2;

		// end of pixel processing ----------------
	}
}


// Test 12
// this version uses the at method
void colorReduce12(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols; // number of columns
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {
		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			image.at<cv::Vec3b>(j, i)[0] = image.at<cv::Vec3b>(j, i)[0] / div*div + div2;
			image.at<cv::Vec3b>(j, i)[1] = image.at<cv::Vec3b>(j, i)[1] / div*div + div2;
			image.at<cv::Vec3b>(j, i)[2] = image.at<cv::Vec3b>(j, i)[2] / div*div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}


// Test 13
// this version uses Mat overloaded operators
void colorReduce13(cv::Mat image, int div = 64) {

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

							// perform color reduction
	image = (image&cv::Scalar(mask, mask, mask)) + cv::Scalar(div / 2, div / 2, div / 2);
}

// Test 14
// this version uses a look up table速度最快
void colorReduce14(cv::Mat image, int div = 64) {

	cv::Mat lookup(1, 256, CV_8U);

	for (int i = 0; i<256; i++) {

		lookup.at<uchar>(i) = i / div*div + div / 2;
	}

	cv::LUT(image, lookup, image);
}

#define NTESTS 15
#define NITERATIONS 10

int main()
{
	// read the image
	cv::Mat image = cv::imread("earth.jpg");

	// time and process the image
	const int64 start = cv::getTickCount();
	colorReduce(image, 64);
	//Elapsed time in seconds
	double duration = (cv::getTickCount() - start) / cv::getTickFrequency();

	// display the image
	std::cout << "Duration= " << duration << "secs" << std::endl;
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	cv::waitKey();

	// test different versions of the function

	int64 t[NTESTS], tinit;
	// timer values set to 0
	for (int i = 0; i<NTESTS; i++)
		t[i] = 0;

	cv::Mat images[NTESTS];
	cv::Mat result;

	// the versions to be tested
	typedef void(*FunctionPointer)(cv::Mat, int);
	FunctionPointer functions[NTESTS] = { colorReduce, colorReduce1, colorReduce2, colorReduce3, colorReduce4,
		colorReduce5, colorReduce6, colorReduce7, colorReduce8, colorReduce9,
		colorReduce10, colorReduce11, colorReduce12, colorReduce13, colorReduce14 };
	// repeat the tests several times
	int n = NITERATIONS;
	for (int k = 0; k<n; k++) {

		std::cout << k << " of " << n << std::endl;

		// test each version
		for (int c = 0; c < NTESTS; c++) {

			images[c] = cv::imread("earth.jpg");

			// set timer and call function
			tinit = cv::getTickCount();
			functions[c](images[c], 64);
			t[c] += cv::getTickCount() - tinit;

			std::cout << ".";
		}

		std::cout << std::endl;
	}

	// short description of each function
	std::string descriptions[NTESTS] = {
		"original version:",
		"with dereference operator:",
		"using modulo operator:",
		"using a binary mask:",
		"direct ptr arithmetic:",
		"row size recomputation:",
		"continuous image:",
		"reshape continuous image:",
		"with iterators:",
		"Vec3b iterators:",
		"iterators and mask:",
		"iterators from Mat_:",
		"at method:",
		"overloaded operators:",
		"look-up table:",
	};

	for (int i = 0; i < NTESTS; i++) {

		cv::namedWindow(descriptions[i]);
		cv::imshow(descriptions[i], images[i]);
	}

	// print average execution time
	std::cout << std::endl << "-------------------------------------------" << std::endl << std::endl;
	for (int i = 0; i < NTESTS; i++) {

		std::cout << i << ". " << descriptions[i] << 1000.*t[i] / cv::getTickFrequency() / n << "ms" << std::endl;
	}

	cv::waitKey();
	return 0;
}








