#include "util.h"


Centerface::Centerface(string model_path, vector<string> out_names)
{
	net_ = dnn::readNetFromONNX(model_path);
	out_names_ = out_names;
	img_size_new_ = Size(0, 0);
	scale_h_ = 1;
	scale_w_ = 1;

}

vector<Mat> Centerface::Inference(Mat image, float thresh)
{
	static Mat blob;

	int img_h_new = ceil(image.rows*1.0 / 32) * 32;
	int img_w_new = ceil(image.cols*1.0 / 32) * 32;

	scale_h_ = img_h_new * 1.0 / image.rows;
	scale_w_ = img_w_new*1.0 / image.cols;


	img_size_new_ = Size(img_w_new, img_h_new);

	dnn::blobFromImage(image, blob, 1.0, img_size_new_, Scalar(0, 0, 0), true, false, CV_8U);

	net_.setInput(blob, "");

	vector<Mat> outs;

	net_.forward(outs, out_names_);

	return outs;

}

vector<DBOX> Centerface::Postprocess(vector<Mat> outs)
{
	Mat headmap(outs[0].size[2], outs[0].size[3], CV_32F, outs[0].ptr<float>());

	int data_seq;

	vector<DBOX> boxes;

	float *s0, *s1, *o0, *o1, s;

	for (int r = 0; r < headmap.rows; r++)
	{
		float * row_data = headmap.ptr<float>(r);
		for (int c = 0; c < headmap.cols; c++)
		{
			s = row_data[c];
			if (s < 0.5)
				continue;
			data_seq = r*headmap.cols + c;

			int s0_id = outs[1].step[2] * r + outs[1].step[3] * c;
			s0 = (float *)(outs[1].data + s0_id);
			int s1_id = outs[1].step[1] + outs[1].step[2] * r + outs[1].step[3] * c;
			s1 = (float *)(outs[1].data + s1_id);

			int o0_id = outs[2].step[2] * r + outs[2].step[3] * c;
			o0 = (float *)(outs[2].data + o0_id);
			int o1_id = outs[2].step[1] + outs[2].step[2] * r + outs[2].step[3] * c;
			o1 = (float *)(outs[2].data + o1_id);

			//circle(image, Point((c + *o1 + 0.5) * 4/ scale_w, (r + *o0 + 0.5) * 4/ scale_h), 15, Scalar(255, 0, 255), 1, 8);

			float x1 = fmin(img_size_new_.width, fmax(0, (c + *o1 + 0.5) * 4 - exp(*s1) * 4 / 2)) / scale_w_;
			float y1 = fmin(img_size_new_.height, fmax(0, (r + *o0 + 0.5) * 4 - exp(*s0) * 4 / 2)) / scale_h_;
			float box_width = (fmin(x1 + exp(*s1) * 4, img_size_new_.width) - x1) / scale_w_;
			float box_height = (fmin(y1 + exp(*s0) * 4, img_size_new_.height) - y1) / scale_h_;
			cout << x1 << " " << y1 << " " << box_height << " " << box_height << endl;
			Rect rect(x1, y1, box_width, box_height);
			DBOX box = { rect, s };
			boxes.push_back(box);
		}
	}

	vector<DBOX> ret = nms(boxes, 0.5);

	return ret;
}


bool sort_score(DBOX box1, DBOX box2)
{
	return (box1.score > box2.score);

}

bool int_cmp(int a, int b)
{
	return a>b;
}

float iou(DBOX box1, DBOX box2)
{
	return (box1.rect & box2.rect).area()*1.0 / (box1.rect | box2.rect).area();
}

vector<DBOX> nms(std::vector<DBOX>&vec_boxs, float threshold)
{
	vector<DBOX>results;

	while (vec_boxs.size() > 0)
	{
		vector<int> remove_indexs;
		remove_indexs.push_back(0);
		std::sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
		results.push_back(vec_boxs[0]);

		for (int i = 0; i <vec_boxs.size() - 1; i++)
		{
			float iou_value = iou(vec_boxs[0], vec_boxs[i + 1]);
			if (iou_value >threshold)
			{
				remove_indexs.push_back(i + 1);
			}
		}

		std::sort(remove_indexs.begin(), remove_indexs.end(), int_cmp);

		for (auto i : remove_indexs)
			vec_boxs.erase(std::begin(vec_boxs) + i);
	}
	return results;
}
