#include <iostream>
#include "face_detector.hpp"
#include "helpers.hpp"

int main(int argc, char**argv) {
	Timer timer;
    bool step4 = argc>1 ? atoi(argv[1]) : false;
	mtcnn::FaceDetector fd("./model/", 0.6f, 0.7f, 0.7f, step4);
	//cv::Mat img = cv::imread(argc<2?"test.jpg":argv[1]);
	cv::VideoCapture cap(0);
	timer.start();
	while(1) {
		cv::Mat img;
		if (!cap.read(img)) break;

		std::vector<mtcnn::Face> faces = fd.detect(img, 40.f, 0.709f);
		for (size_t i = 0; i < faces.size(); ++i) {
			std::vector<cv::Point> pts;
			for (int p = 0; p < mtcnn::NUM_PTS; ++p) {
				pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
			}
			drawFace(img, faces[i].bbox.getRect(), pts);
		}
		cv::imshow("mtcnn", img);
		if (cv::waitKey(20) >=0) break;
	}
	return 0;
}
