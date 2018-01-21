#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.h"


cv::Mat labHist(const cv::Mat& src, int lbins = 20, int abins = 20, int bbins = 20);

//��������
class SuperPixel
{
public:
	//���ĵ�
	cv::Point center;
	//��������ļ��ϣ�
	std::vector<cv::Point> contour;
	//�ڲ����أ���ļ��ϣ�
	std::vector<cv::Point> pixels;
	//������ȣ����飩
	std::vector<float> pixels_depth;

	//ֱ��ͼ
	cv::Mat hist;

	int pixel_num;
	int depth_num;
	float depth_average;
	float depth_max;
	float depth_min;
	SuperPixel() 
	{
		hist = cv::Mat();
	}
	~SuperPixel() {}

	//��ȡ��������
	double chiSquareDist(const SuperPixel &right);

	//��ȡĳ������أ�����õ���ȣ�
	cv::Point& get_pixel(int i);
	//
	void create(cv::Mat &origin_img);
	//�ж���ǰ�����ص��Ƿ��������Ϣ
	bool have_depth();

private:
};


//ͼƬ��Ϣ��
class ImgData
{
public:
	//����
	int id;
	//�������
	Camera cam;
	//�������Ϣ��ͼƬ
	cv::Mat depth_mat;
	//
	cv::Mat world_center;
	//ԭʼͼƬ
	cv::Mat origin_img;
	//�����ص��ǩ������
	cv::Mat sp_label;
	//�����ص�����������
	cv::Mat sp_contour;
	//�����ص����Ŀ
	int sp_num;
	std::vector<SuperPixel> data;
	std::string path_output;
	//����
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	//��ȡ�����ص�
	SuperPixel& get_superpixel(int i);
	//��ȡĳ��������Ϣ
	float& get_pixel_depth(cv::Point& point);

private:
	//�������ƶ�·��������
	void create_path();
	//�����������ģ�����
	void calc_world_center();
	//���������ص�
	void generate_sp();
	//���������Ϣ
	float find_depth(cv::Point& p_begin);
	//���������Ϣ��ͼƬ
	void save_depth_image();
	//���泬���ص�ͼƬ
	void save_sp_image();
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);
