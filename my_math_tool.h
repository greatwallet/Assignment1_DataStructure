#pragma once
#include <opencv2/opencv.hpp>

bool is_zero(float x);

bool check_range(cv::Point& point);

void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);

//计算超像素点的（？）
void calc_sp(cv::Mat &origin_img, cv::Mat &output_sp_label, cv::Mat &output_sp_contour, int &output_sp_num);

//三重插值（？）
void tri_interpolation(std::vector<cv::Point2f>& triangle, cv::Point& point, std::vector<float>& coefficient);

cv::Point inv_tri_interpolation(std::vector<cv::Point2f>& triangle, std::vector<float>& coefficient);

//点间距
float point_distance(cv::Point2f p1, cv::Point2f p2);

void contour_to_set(std::vector<cv::Point>& contour, std::vector<cv::Point>& points_set);

//计算三角形面积
float calc_triangle_area(std::vector<cv::Point2f>& triangle);

std::vector<float> fit_plane(std::vector<cv::Mat> point_set);