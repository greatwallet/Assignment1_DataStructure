#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.h"





class SuperPixel
{
public:
	cv::Point center;
	std::vector<cv::Point> contour;
	std::vector<cv::Point> pixels;
	std::vector<float> pixels_depth;
	int pixel_num;
	int depth_num;
	float depth_average;
	float depth_max;
	float depth_min;
	SuperPixel() {}
	~SuperPixel() {}

	cv::Point& get_pixel(int i);
	void create();
	bool have_depth();

private:
};



class ImgData
{
public:
	//±äÁ¿
	int id;
	Camera cam;
	cv::Mat depth_mat;
	cv::Mat world_center;
	cv::Mat origin_img;
	cv::Mat sp_label;
	cv::Mat sp_contour;
	int sp_num;
	std::vector<SuperPixel> data;
	std::string path_output;
	//º¯Êý
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	SuperPixel& get_superpixel(int i);
	float& get_pixel_depth(cv::Point& point);

private:
	void create_path();
	void calc_world_center();
	void generate_sp();
	float find_depth(cv::Point& p_begin);
	void save_depth_image();
	void save_sp_image();
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);
