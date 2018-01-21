#include <opencv2\highgui.hpp>
#include <opencv\cv.h>

#include <iostream>
#include <string>
#include <conio.h>

#include "init.h"
#include "global_var.h"

using namespace std;
using namespace cv;



//ͨ����ͼ�۲���
void watch_result(vector<ImgData>& imgdata_vec, int interpolation_density)	//interpolation_density��ʾÿ������Ƭ���ֵ����ͼƬ
{
	//����·���ĺ��������ĵĳ�ʼ����ÿ���۲���������ͼƬ��ȷ��
	vector<Mat> pos_path;
	vector<Mat> center_path;
	vector<vector<int>> near_img_id;		//��ά����,�洢
	for (int i = 1; i < imgdata_vec.size() - 2; i++)		//�ӵڶ���ͼ�������ڶ���ͼ֮���ֵ
	{
		Mat &pos_before = imgdata_vec[i].cam.cam_pos;
		Mat &pos_after = imgdata_vec[i + 1].cam.cam_pos;
		Mat &center_before = imgdata_vec[i].world_center;
		Mat &center_after = imgdata_vec[i + 1].world_center;
		for (int j = 0; j < interpolation_density; j++)
		{
			//��ֵ����
			pos_path.push_back(((interpolation_density - j)*pos_before + j * pos_after) / interpolation_density);
			center_path.push_back(((interpolation_density - j)*center_before + j * center_after) / interpolation_density);
			//����j���������������Ƭ��id
			{
				//����ÿ��j��������һ��temp_near_img_id
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
	//����۲���
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

	//�����Ҽ��鿴�۲���
	int i = 0;
	//int key_value;
	//��ѭ����
	while (true)
	{
		//����������
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
		//�ȴ�ʱ�䣿
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
	//������ĳ����طָmve�ؽ���ͼƬ����
	vector<ImgData> imgdata_vec;
	bool with_file = true;	//false:	���³����طָ�
	bool run_mve = false;	//true:		������mve�ؽ�
	init(imgdata_vec, run_mve, with_file);
	//�˴�Ӧ��depth synthesis

	watch_result(imgdata_vec, 5);	//��Ҫ����


	return 0;
}