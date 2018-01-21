#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.h"


cv::Mat labHist(const cv::Mat& src, int lbins = 20, int abins = 20, int bbins = 20);

//超像素类
class SuperPixel
{
public:
	//中心点
	cv::Point center;
	//轮廓（点的集合）
	std::vector<cv::Point> contour;
	//内部像素（点的集合）
	std::vector<cv::Point> pixels;
	//像素深度（数组）
	std::vector<float> pixels_depth;

	//直方图
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

	//获取卡方距离
	double chiSquareDist(const SuperPixel &right);

	//获取某点的像素（输入该点的秩）
	cv::Point& get_pixel(int i);
	//
	void create(cv::Mat &origin_img);
	//判定当前超像素点是否有深度信息
	bool have_depth();

private:
};


//图片信息类
class ImgData
{
public:
	//变量
	int id;
	//所属相机
	Camera cam;
	//有深度信息的图片
	cv::Mat depth_mat;
	//
	cv::Mat world_center;
	//原始图片
	cv::Mat origin_img;
	//超像素点标签（矩阵）
	cv::Mat sp_label;
	//超像素点轮廓（矩阵）
	cv::Mat sp_contour;
	//超像素点的数目
	int sp_num;
	std::vector<SuperPixel> data;
	std::string path_output;
	//函数
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	//获取超像素点
	SuperPixel& get_superpixel(int i);
	//获取某点的深度信息
	float& get_pixel_depth(cv::Point& point);

private:
	//创建相似度路径（？）
	void create_path();
	//计算世界中心（？）
	void calc_world_center();
	//产生超像素点
	void generate_sp();
	//查找深度信息
	float find_depth(cv::Point& p_begin);
	//保存深度信息的图片
	void save_depth_image();
	//保存超像素点图片
	void save_sp_image();
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);
