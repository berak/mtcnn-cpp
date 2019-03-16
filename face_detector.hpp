#ifndef _FACE_DETECTOR_HPP_
#define _FACE_DETECTOR_HPP_

#include <string>

//#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace mtcnn {

const int NUM_REGRESSIONS = 4;
const int NUM_PTS = 5;

struct BBox {
	float x1;
	float y1;
	float x2;
	float y2;
	cv::Rect getRect() const;
	BBox getSquare() const;
};

struct Face {
	BBox bbox;
	float regression[NUM_REGRESSIONS];
	float score;
	float ptsCoords[2 * NUM_PTS];

	static void applyRegression(std::vector<Face>& faces, bool addOne = false);
	static void bboxes2Squares(std::vector<Face>& faces);
};

class FaceDetector {
private:

	cv::dnn::Net pNet_,rNet_,oNet_,lNet_;

	float pThreshold_;
	float rThreshold_;
	float oThreshold_;
	bool useLNet_;
	void initNetInput(cv::dnn::Net &net, cv::Mat img);
	void initNetInput(cv::dnn::Net &net, std::vector<cv::Mat>& imgs);
	std::vector<Face> step1(cv::Mat img, float minFaceSize, float scaleFactor);
	std::vector<Face> step2(cv::Mat img, const std::vector<Face>& faces);
	std::vector<Face> step3(cv::Mat img, const std::vector<Face>& faces);
	std::vector<Face> step4(cv::Mat img, const std::vector<Face>& faces);
	std::vector<Face> composeFaces(const cv::Mat &regressionsBlob,
								   const cv::Mat &scoresBlob,
								   float scale);
	static std::vector<Face> nonMaximumSuppression(std::vector<Face> faces, float threshold, bool useMin = false);
public:
	FaceDetector(const std::string& modelDir,
				 float pThreshold = 0.6f,
				 float rThreshold = 0.7f,
				 float oThreshold = 0.7f,
				 bool useLNet = true);

	std::vector<Face> detect(cv::Mat img, float minFaceSize, float scaleFactor);
};

} // namespace mtcnn

#endif // _FACE_DETECTOR_HPP_
