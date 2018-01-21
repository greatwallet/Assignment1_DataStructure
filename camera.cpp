#include "camera.h"

#include <iostream>
#include "global_var.h"



using namespace std;
using namespace cv;

Camera::Camera(const Camera& cam)
{
	cam.K.copyTo(K);
	cam.R.copyTo(R);
	cam.T.copyTo(T);
	cam.Ki.copyTo(Ki);
	cam.Ri.copyTo(Ri);
	cam.world_to_cam.copyTo(world_to_cam);
	cam.cam_to_world.copyTo(cam_to_world);
	cam.cam_pos.copyTo(cam_pos);
}

Camera::Camera(Mat& _K, Mat& _R, Mat& _T)
{
	_K.copyTo(K);
	_R.copyTo(R);
	_T.copyTo(T);
	//Ki=K^(-1),Ri=R^T;
	Ki = K.inv();
	Ri = R.t();
	//cam pos
	cam_pos = -Ri * T;

	//calculate world to cam
	{
		/*	��R��T����������������Ϊ���
		*	temp = [R,T],Ϊ3*4����
		*/
		Mat temp; hconcat(R, T, temp);
		//����γ˻���Ϊ��������������Ϊ������ӣ�����������ת��Ϊ�������
		//world_to_camΪ3*3���󣬳�Ϊ�����֮�˻�
		//world_to_cam=K*[R,T]
		world_to_cam = K * temp;
	}
	//calculate cam to world
	{
		//temp=[K^(-1),-T]Ϊ3*4����
		Mat temp; hconcat(Ki, -T, temp);
		//��Ϊ������ӣ����������ת��Ϊ��������
		//cam_to_world=R^T*[K^(-1),-T]
		cam_to_world = Ri * temp;
	}
}

//���������ת��Ϊ�������꣨��Ҫ�����Ϣ��
Mat Camera::get_world_pos(Point point, float depth)
{
	int x = point.x; 
	int y = point.y;
	Mat xy = (Mat_<float>(4, 1) << depth * x, depth * y, depth, 1);
	return cam_to_world * xy;
}

//����������ת��Ϊ�������,xyzΪ3*1������
Point Camera::get_cam_pos(Mat xyz)
{
	
	//	(x			(x
	//	 y		=	 y
	//	 z)		   	 z
	//			   	 1)
	vconcat(xyz, Mat(Mat_<float>(1, 1) << 1), xyz);
	//temp_resultΪxyz������ж�Ӧ�����١�һ��ϵ��
	Mat temp_result = world_to_cam * xyz;
	//�õ��������
	return Point(temp_result.at<float>(0, 0) / temp_result.at<float>(2, 0),
		temp_result.at<float>(1, 0) / temp_result.at<float>(2, 0));
}

//dst->destinationĿ��	src->source��Դ
//
void Camera::fill_reprojection(Camera &dst_cam, Mat& mat, Mat& vec)
{
	Mat &dst_K = dst_cam.K, &dst_R = dst_cam.R;
	Mat &src_Ri = Ri, &src_Ki = Ki;
	Mat &dst_t = dst_cam.T, &src_t = T;
	//mat=d.K*d.R*(s.R^Tran)*s.K^(-1);	Ϊ3*3����
	mat = dst_K * dst_R * src_Ri * src_Ki;
	//vec=d.K*(d.T-d.R*s.R^Tran*s.T),�õ�3*1������
	vec = dst_K * (dst_t - dst_R * src_Ri * src_t);
}

//�õ���һ��camera��origin��->��ͶӰ��*this camera�����꣬mat,vec���Ǽ�����м�����
Point cal_reprojection(Point origin_point, float depth, Mat& mat, Mat& vec)
{
	//3*1��������x,y,1��
	Mat point = (Mat_<float>(3, 1) << origin_point.x, origin_point.y, 1);
	Mat result = mat * point * depth + vec;
	Point dst_point;
	dst_point.x = result.at<float>(0, 0) / result.at<float>(2, 0);
	dst_point.y = result.at<float>(1, 0) / result.at<float>(2, 0);
	return dst_point;
}

//�����������
Camera Camera::generate_novel_cam(Mat cam_pos_novel, Mat world_center)
{
	Mat novel_z = world_center - cam_pos_novel; 
	normalize(novel_z, novel_z);	//ʹnovel_z��׼������һ����
	Mat origin_x = R.row(0); origin_x = origin_x.t();
	Mat origin_y = R.row(1); origin_y = origin_y.t();
	Mat origin_z = R.row(2); origin_z = origin_z.t();

	//Rodrigues' rotation: calculate transform R to R_novel: use origin_z -> novel_z
	//�޵������ת
	Mat axis = origin_z.cross(novel_z); normalize(axis, axis);
	double theta = acos(origin_z.dot(novel_z));

	float kx = axis.at<float>(0, 0);
	float ky = axis.at<float>(1, 0);
	float kz = axis.at<float>(2, 0);

	Mat Rodrigues_K = (Mat_<float>(3, 3) << 0, -kz, ky, kz, 0, -kx, -ky, kx, 0);
	Mat Rodrigues_I = Mat::eye(3, 3, CV_32F);	//����һ����λ���󣨺�ͬ����I��
	Mat Rodrigues_R = cos(theta) * Rodrigues_I + (1 - cos(theta))*Rodrigues_K*Rodrigues_K + sin(theta) * Rodrigues_K;

	Mat origin_xyz = R.t();
	Mat Rn = Rodrigues_R * origin_xyz; Rn = Rn.t();
	Mat Tn = -Rn * cam_pos_novel;

	return Camera(K, Rn, Tn);
}

void Camera::debug()
{
	cout << cam_pos << endl;
	cout << K << endl;
	cout << Ki << endl;
	cout << R << endl;
	cout << Ri << endl;
	cout << T << endl;
}
