#pragma once
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

typedef struct DBOX {
	Rect rect;
	float score;
}DBOX;


class Centerface {
public:
	Centerface(string model_path, vector<string> out_names);
	vector<Mat> Inference(Mat image, float thresh);
	vector<DBOX> Postprocess(vector<Mat> outs);


private:
	dnn::Net net_;
	vector<string> out_names_;
	Size img_size_new_;
	float scale_h_;
	float scale_w_;

};


bool int_cmp(int a, int b);

float iou(DBOX box1, DBOX box2);

vector<DBOX> nms(std::vector<DBOX>&vec_boxs, float threshold);

