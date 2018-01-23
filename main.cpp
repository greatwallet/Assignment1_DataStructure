#include <opencv2\highgui.hpp>
#include <opencv\cv.h>

#include <iostream>
#include <string>
#include <conio.h>

#include "init.h"
#include "global_var.h"

using namespace std;
using namespace cv;



//通过视图观测结果
void watch_result(vector<ImgData>& imgdata_vec, int interpolation_density)	//interpolation_density表示每两张照片里插值多少图片
{
	//进行路径的和世界中心的初始化和每个观测点最近四张图片的确定
	vector<Mat> pos_path;
	vector<Mat> center_path;
	vector<vector<int>> near_img_id;		//二维矩阵,存储
	for (int i = 1; i < imgdata_vec.size() - 2; i++)		//从第二张图到倒数第二张图之间插值
	{
		Mat &pos_before = imgdata_vec[i].cam.cam_pos;
		Mat &pos_after = imgdata_vec[i + 1].cam.cam_pos;
		Mat &center_before = imgdata_vec[i].world_center;
		Mat &center_after = imgdata_vec[i + 1].world_center;
		for (int j = 0; j < interpolation_density; j++)
		{
			//插值处理
			pos_path.push_back(((interpolation_density - j)*pos_before + j * pos_after) / interpolation_density);
			center_path.push_back(((interpolation_density - j)*center_before + j * center_after) / interpolation_density);
			//根据j保存最近的四张照片的id
			{
				//对于每个j都会生成一个temp_near_img_id
				vector<int> temp_near_img_id;
				if (1.0 * j / interpolation_density < 0.5)
				{
					temp_near_img_id.push_back(i);
					temp_near_img_id.push_back(i + 1);
					temp_near_img_id.push_back(i - 1);
					temp_near_img_id.push_back(i + 2);
				}
				else
				{
					temp_near_img_id.push_back(i + 1);
					temp_near_img_id.push_back(i);
					temp_near_img_id.push_back(i + 2);
					temp_near_img_id.push_back(i - 1);
				}
				
				near_img_id.push_back(temp_near_img_id);
			}
		}

	}
	/*
	//输出观测结果
	for(int i = 0; i < pos_path.size(); i++)
	{
		Camera now_cam = imgdata_vec[0].cam.generate_novel_cam(pos_path[i], center_path[i]);
		cout << "nearest cam id:";
		for (int j = 0; j < near_img_id[i].size(); j++)
		{
			cout << near_img_id[i][j] << "->";
		}
		cout << endl;

		Mat output_img;
		mix_pic(imgdata_vec, now_cam, near_img_id[i], output_img);
		imwrite(PATH_MY_OUTPUT + "\\images\\" + to_string(i) + ".jpg", output_img);
	}
	*/

	//用左右键查看观测结果
	int i = 0;
	//int key_value;
	//死循环？
	while (true)
	{
		//产生虚拟像
		Camera now_cam = imgdata_vec[0].cam.generate_novel_cam(pos_path[i], center_path[i]);
		cout << "view in id:" << i << ", nearest cam id:";
		for (int j = 0; j < near_img_id[i].size(); j++)
		{
			cout << near_img_id[i][j] << "->";
		}
		cout << endl;
		Mat output_img;
		//////////
		/////////
		///////
		/////
		///
		//
		mix_pic(imgdata_vec, now_cam, near_img_id[i], output_img);
		imshow("blend_pic", output_img);
		//等待时间？
		int key_value = waitKey();
		switch (key_value)
		{
		case 97:
			if (i > 0)
				i--;
			break;
		case 100:
			if (i < pos_path.size() - 1)
				i++;
			break;
		default:
			break;
		}
	}
}

int main()
{
	//已做完的超像素分割、mve重建的图片数组
	vector<ImgData> imgdata_vec;
	bool with_file = true;	//false:	重新超像素分割
	bool run_mve = false;	//true:		重新做mve重建
	init(imgdata_vec, run_mve, with_file);

	//std::ofstream fout("RGB information.txt");
	//cv::Mat img(imgdata_vec[0].origin_img);
	//for (int y = 0; y < HEIGHT; y++)
	//{
	//	for (int x = 0; x < WIDTH; x++)
	//	{
	//		cv::Vec3b v = img.at<cv::Vec3b>(cv::Point(x, y));
	//		unsigned char B = v[0];
	//		unsigned char G = v[1];
	//		unsigned char R = v[2];
	//		fout << " x = " << x << " y = " << y;
	//		fout << " Red = " << int(R) << " Green = " << int(G) << " Blue = " << int(B) << std::endl;
	//	}
	//}
	//fout.close();
	/*imgdata_vec[0].show_sky();*/
	imgdata_vec[0].depth_synthesis();
	/*std::ofstream fout("img count.txt");
	for (int i = 0; i < imgdata_vec.size(); i++)
	{
		fout << "IMG NO. = " << i << " ";
		imgdata_vec[i].show_sky();
		imgdata_vec[i].depth_synthesis();
		if ((i + 1) % 8 == 0)fout << std::endl;
	}
	fout.close();*/

	//此处应做depth synthesis
	{
		
		//ofstream fout("adjacency list.txt");

		//for (int i = 0; i < imgdata_vec[0].adjacency_list.size(); i++)
		//{
		//	fout << "Rank " << i << " node:" << std::endl;
		//	for (int j = 0; j < imgdata_vec[0].adjacency_list[i].size(); j++)
		//	{
		//		fout << imgdata_vec[0].adjacency_list[i][j].sp_rank << " ";
		//	}
		//	std::cout << endl;
		//	for (int j = 0; j < imgdata_vec[0].adjacency_list[i].size(); j++)
		//	{
		//		fout << imgdata_vec[0].adjacency_list[i][j].edgeCost << " ";
		//	}
		//	std::cout << endl;
		//}


		//fout.close();
	}
	watch_result(imgdata_vec, 5);	//主要函数


	return 0;
}