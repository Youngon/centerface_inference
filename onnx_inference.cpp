// onnx_inference.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "util.h"

int main()
{
	
	float threshold = 0.5;

	string model_path = "C:\\deep_learnning\\centerface\\centerface.onnx";

	string output_tensor_names[] = { "537", "538", "539", "540" };
	vector<string> out_names;
	for (auto name : output_tensor_names)
	{
		out_names.push_back(name);
		cout << name << endl;
	}

	Centerface cfnet(model_path, out_names);

	Mat image;
	image = imread("C:\\deep_learnning\\centerface\\000388.jpg");

	vector<Mat> outs = cfnet.Inference(image, threshold);


	vector<DBOX> ret = cfnet.Postprocess(outs);

	for (auto r : ret)
	{
		rectangle(image, Point(r.rect.x, r.rect.y), Point(r.rect.x+r.rect.width, r.rect.y+r.rect.height), Scalar(0, 255, 0));

		std::string label = format("%.2f", r.score);
		putText(image, label, Point(r.rect.x+2, r.rect.y+2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
	}

	imshow("image", image);
	waitKey(0);
	system("pause");

    return 0;
}

