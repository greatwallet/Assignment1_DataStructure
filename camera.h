#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <string>


class Camera
{
public:
	//member
	//�������
	//K-�ڲ�(3*3)��R-�������ת����(3*3),T-�����ƽ��������3*1��
	cv::Mat K, R, T, Ki, Ri;
	//������ӣ��ɽ���������������������ת��
	cv::Mat world_to_cam, cam_to_world;
	cv::Mat cam_pos;
	//function

	Camera() {}
	Camera(const Camera& cam);
	//���캯��
	Camera(cv::Mat& _K, cv::Mat& _R, cv::Mat& _T);
	//���������ת��Ϊ�������꣨��Ҫ�����Ϣ��
	cv::Mat get_world_pos(cv::Point point, float depth);
	//��ȡ���λ��
	cv::Point get_cam_pos(cv::Mat xyz);
	//�����������
	Camera generate_novel_cam(cv::Mat cam_pos_novel, cv::Mat world_center);
	//������ͶӰ
	void fill_reprojection(Camera& des_cam, cv::Mat& mat, cv::Mat& vec);

	void debug();
};

//�������ͶӰ
cv::Point cal_reprojection(cv::Point origin_point, float depth, cv::Mat& mat, cv::Mat& vec);


#endif // !CAMERA_H