#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <string>


class Camera
{
public:
	//member
	//相机参数
	//K-内参(3*3)，R-外参中旋转矩阵(3*3),T-外参中平移向量（3*1）
	cv::Mat K, R, T, Ki, Ri;
	//左乘因子，可将世界坐标与相机坐标进行转换
	cv::Mat world_to_cam, cam_to_world;
	cv::Mat cam_pos;
	//function

	Camera() {}
	Camera(const Camera& cam);
	//构造函数
	Camera(cv::Mat& _K, cv::Mat& _R, cv::Mat& _T);
	//将相机坐标转化为世界坐标（需要深度信息）
	cv::Mat get_world_pos(cv::Point point, float depth);
	//获取相机位置
	cv::Point get_cam_pos(cv::Mat xyz);
	//产生虚拟相机
	Camera generate_novel_cam(cv::Mat cam_pos_novel, cv::Mat world_center);
	//填充二次投影
	void fill_reprojection(Camera& des_cam, cv::Mat& mat, cv::Mat& vec);

	void debug();
};

//计算二次投影
cv::Point cal_reprojection(cv::Point origin_point, float depth, cv::Mat& mat, cv::Mat& vec);


#endif // !CAMERA_H