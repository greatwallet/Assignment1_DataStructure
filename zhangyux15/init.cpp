#include "global_var.h"
#include "imgdata.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <windows.h>
#include <cmath>

#include <string>
#include <regex>
#include <vector>

#include <iostream>
#include<fstream>

using namespace std;
using namespace cv;

string PATH_PROJECT;
string PATH_MY_OUTPUT;
string PATH_MVE_OUTPUT;
string PATH_MY_DATA_SAVED;

int HEIGHT;
int WIDTH;


void run_mve_reconstruction()
{
	string mve_path = PATH_PROJECT + "\\program_lib\\mve\\";
	string command = mve_path + "makescene -i " + PATH_PROJECT + "\\dataset " + PATH_MVE_OUTPUT;
	system(command.c_str());
	command = mve_path + "sfmrecon " + PATH_MVE_OUTPUT;
	system(command.c_str());
	command = mve_path + "dmrecon -s2 " + PATH_MVE_OUTPUT;
	system(command.c_str());
	command = mve_path + "scene2pset -F2 " + PATH_MVE_OUTPUT + " " + PATH_MVE_OUTPUT + "\\pset-L2.ply";
	system(command.c_str());
	command = mve_path + "fssrecon " + PATH_MVE_OUTPUT + "\\pset-L2.ply " + PATH_MVE_OUTPUT + "\\surface-L2.ply";
	system(command.c_str());
	command = mve_path + "meshclean -t10 " + PATH_MVE_OUTPUT + "\\surface-L2.ply " + PATH_MVE_OUTPUT + "\\surface-L2-clean.ply";
	system(command.c_str());

}

// ��mve����ļ��ж�ȡ��Ϣ��Imgdata��
void read_view_data(ImgData& imgdata, string path_view, bool with_file)
{
	// ��ʼ�������VIEW��ID
	int id;
	float focal_length, pixel_aspect;
	float principal_point[2];
	Mat camera_R, camera_T, camera_K;
	ifstream mera_ini(path_view + "\\meta.ini");
	string s;
	while (getline(mera_ini, s))//���ж�ȡ���ݲ�����s�У�ֱ������ȫ����ȡ
	{
		smatch m;
		if (std::regex_search(s, m, regex("id = (.+)")))
		{
			id = stoi(m[1].str());
		}

		if (std::regex_search(s, m, regex("focal_length = (.+)")))
		{
			focal_length = stof(m[1].str());
		}

		if (std::regex_search(s, m, regex("pixel_aspect = (.+)")))
		{
			pixel_aspect = stof(m[1].str());
		}

		if (std::regex_search(s, m, regex("principal_point = (.+)")))
		{
			vector<string> result;
			split_string(m[1].str(), result, " ");
			principal_point[0] = stof(result[0]);
			principal_point[1] = stof(result[1]);
		}

		if (std::regex_search(s, m, regex("rotation = (.+)")))
		{
			vector<string> result_str;
			float result_f[9];
			split_string(m[1].str(), result_str, " ");
			for (int i = 0; i < 9; i++)
			{
				result_f[i] = stof(result_str[i]);
			}
			camera_R = Mat(3, 3, CV_32F, result_f);
		}

		if (std::regex_search(s, m, regex("translation = (.+)")))
		{
			vector<string> result_str;
			float result_f[3];
			split_string(m[1].str(), result_str, " ");
			for (int i = 0; i < 3; i++)
			{
				result_f[i] = stof(result_str[i]);
			}
			camera_T = Mat(3, 1, CV_32F, result_f);
		}

	}
	mera_ini.close();
	// ���������K����ֱ����MVE�ĺ���ʵ�֣�
	{
		float dim_aspect = (float)WIDTH / HEIGHT;
		float image_aspect = dim_aspect * pixel_aspect;
		float ax, ay;
		if (image_aspect < 1.0f)/* Portrait. */
		{
			ax = focal_length * HEIGHT / pixel_aspect;
			ay = focal_length * HEIGHT;
		}
		else /* Landscape. */
		{
			ax = focal_length * WIDTH;
			ay = focal_length * WIDTH * pixel_aspect;
		}
		float mat[9];
		mat[0] = ax; mat[1] = 0.0f; mat[2] = WIDTH * principal_point[0];
		mat[3] = 0.0f; mat[4] = ay; mat[5] = HEIGHT * principal_point[1];
		mat[6] = 0.0f; mat[7] = 0.0f; mat[8] = 1.0f;
		camera_K = Mat(3, 3, CV_32F, mat);
	}
	Camera cam(camera_K, camera_R, camera_T);
	// ��ȡ���ͼ
	ifstream depth_mvei(path_view + "\\depth-L2.mvei", ios::binary);
	depth_mvei.seekg(0, std::ios::end);    // go to the end  
	int fill_size = (int)(depth_mvei.tellg());           // report location (this is the length) 
	char* buffer = new char[fill_size];    // allocate memory for a buffer of appropriate dimension  
	depth_mvei.seekg(0, std::ios::beg);    // go back to the beginning  
	depth_mvei.read(buffer, fill_size);       // read the whole file into the buffer 
	depth_mvei.close();
	int data_size = WIDTH * HEIGHT;
	float* depth_vec = new float[data_size];
	int pos = 27;
	for (int i = 0; i < data_size; i++)
	{
		depth_vec[i] = *((float*)&buffer[pos]);
		pos += 4;
	}
	// �õ����ͼ
	Mat depth_mat(HEIGHT, WIDTH, CV_32F, depth_vec);
	// ��ȡԭͼ
	Mat origin_img = imread(path_view + "\\undist-L2.png");
	// �õ�������
	Mat sp_label, sp_contour; int sp_num;
	if (!with_file)
	{
		calc_sp(origin_img, sp_label, sp_contour, sp_num);		//���¼��㳬���ش����ļ�
		string filename = PATH_MY_DATA_SAVED + "//" + to_string(id) + ".yml";
		FileStorage fs(filename, FileStorage::WRITE);
		//store id,width,height
		fs << "sp_num" << sp_num;
		fs << "sp_label" << sp_label;
		fs << "sp_contour" << sp_contour;
	}
	else
	{// ���ļ���ȡ��������Ϣ
		string filename = PATH_MY_DATA_SAVED + "//" + to_string(id) + ".yml";
		FileStorage fs(filename, FileStorage::READ);
		sp_num = fs["sp_num"];
		fs["sp_label"] >> sp_label;
		fs["sp_contour"] >> sp_contour;
	}

	// ����imgdata
	imgdata = ImgData(id, cam, origin_img, depth_mat, sp_label, sp_contour, sp_num);
}

//��ʼ�����е�Imgdata
void init(vector<ImgData>& imgdata_vec, bool run_mve, bool with_file)
{
	char buf[1000];
	GetCurrentDirectory(1000, buf);
	PATH_PROJECT = string(buf);
	cout << "project path is " << PATH_PROJECT << endl;
	PATH_MY_OUTPUT = PATH_PROJECT + "\\my_output";
	PATH_MVE_OUTPUT = PATH_PROJECT + "\\mve_output";
	PATH_MY_DATA_SAVED = PATH_PROJECT + "\\my_data_saved";
	// �����ʹ���ļ���ʼ���������path my data saved
	if (run_mve)
	{
		string command = "rd /s/q " + PATH_MVE_OUTPUT;
		system(command.c_str());
		//_mkdir(PATH_MVE_OUTPUT.c_str());
		run_mve_reconstruction();
	}

	if (!with_file)
	{
		string command = "rd /s/q " + PATH_MY_DATA_SAVED;
		system(command.c_str());
		_mkdir(PATH_MY_DATA_SAVED.c_str());
	}
	// ��ȡmve�����view��Ŀ¼
	vector<string> dir_view;
	{
		// ��MVE���Ŀ¼�µ�����VIEW��DIR���浽TXT
		string txt_mve_view = PATH_MY_OUTPUT + "\\dir_mve_view.txt";
		string command = "dir " + PATH_MVE_OUTPUT + "\\views" + " > " + txt_mve_view;
		cout << command << endl;
		system(command.c_str());
		// ��TXT��ȡVIEW��Ŀ¼
		std::ifstream in(txt_mve_view);
		string s;
		while (getline(in, s))//���ж�ȡ���ݲ�����s�У�ֱ������ȫ����ȡ
		{
			std::smatch m;
			std::regex e("(view_.+\.mve)");   // matches words beginning by "sub"  
			if (std::regex_search(s, m, e))
			{
				dir_view.push_back(PATH_MVE_OUTPUT + "\\views\\" + m[1].str());
			}
		}
	}
	// ��ʼ���߶ȺͿ��
	Mat img = imread(dir_view[0] + "\\undist-L2.png");
	HEIGHT = img.rows;
	WIDTH = img.cols;
	// ��MVE���ݳ�ʼ��imgdata
    imgdata_vec.resize(dir_view.size());
	for (int i = 0; i < imgdata_vec.size(); i++)
	{
		read_view_data(imgdata_vec[i], dir_view[i], with_file);
	}
}



