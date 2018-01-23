#include "imgdata.h"

#include "global_var.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <thread>

#include <string>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>
#include "my_tool.h"
#include <cmath>

using namespace std;
using namespace cv;

int inX[4] = { 0,0,1,-1 };
int inY[4] = { 1,-1,0,0 };

//获取卡方距离
double SuperPixel::chiSquareDist(const SuperPixel & right)
{
	double sum = 0.0;
	double d1, d2;
	for (int r = 0; r < hist.rows; r++)
	{
		d1 = hist.at<double>(r, 0);
		d2 = right.hist.at<double>(r, 0);
		if (d1 == 0 && d2 == 0)
			continue;
		else
			sum += (pow(d1 - d2, 2) / (d1 + d2))*0.5;
	}
	return sum;
}

Point& SuperPixel::get_pixel(int i)
{
	return pixels[i];
}

bool SuperPixel::have_depth()
{
	//若有深度信息的像素点 占 像素总数超过5%,则说明其有深度
	return 1.0 * depth_num / pixel_num > 0.05;
}

bool SuperPixel::neighbor(const SuperPixel & neigh, int rank_neigh,cv::Mat sp_label)
{
	return posneigh(neigh);
	//if (posneigh(neigh.center))
	//{
	//	int contotal = contour.size();
	//		for (int i = 0; i<contotal; i++)   //边界点是会记到两块超像素点吗？
	//		{
	//			Point p_con = contour[i];
	//			int conrank = sp_label.at<int>(p_con);
	//			if (conrank == rank_neigh) return true;
	//		}
	//}
	//else return false;
}
const int minDIST = 30;
bool SuperPixel::posneigh(cv::Point cen_neigh)
{
	int dist = abs(center.x - cen_neigh.x) + abs(center.y - cen_neigh.y);
	if (dist > minDIST) return false;
	else return true;
}

bool SuperPixel::posneigh(const SuperPixel & neigh)
{
	return posneigh(neigh.center);
}

int ImgData::get_sp_rank(const SuperPixel & sp)
{
	cv::Point p(sp.center);
	return sp_label.at<int>(p);
}

ImgData::ImgData(int _id, Camera& _cam, Mat& _origin_img, Mat& _depth_mat, Mat& _sp_label, Mat& _sp_contour, int _sp_num)
{
	cout << "--Init ImgData with file" << to_string(_id) << "..." << endl;			// 不需要重新计算超像素分割
	id = _id;
	cam = _cam;
	sp_num = _sp_num;
	_sp_label.copyTo(sp_label);
	_sp_contour.copyTo(sp_contour);
	_origin_img.copyTo(origin_img);
	_depth_mat.copyTo(depth_mat);
	path_output = PATH_MY_OUTPUT + "\\" + to_string(id);
	//创建路径
	create_path();
	//计算图上反映的 世界中心
	calc_world_center();
	generate_sp();

	//存储深度信息
	save_depth_image();
	//存储超像素图片
	save_sp_image();
}

//超像素产生函数(产生hist)
void SuperPixel::create(cv::Mat &origin_img)
{
	pixel_num = pixels.size();
	// 计算中心点和平均深度
	Point point_sum(0, 0);
	float depth_sum = 0;
	depth_min = FLT_MAX;
	depth_max = 0;
	depth_num = 0;

	//生成RGB三通道列向量，用于存储蓝、绿、红信息。
	cv::Mat v_bgr(pixel_num, 1, CV_8UC3);
	//将bgr信息转化为lab信息
	cv::Mat v_lab(pixel_num, 1, CV_8UC3);
	//
	hist.resize(pixel_num * 3);

	for (int i = 0, j = 0; i < pixel_num; i++)
	{
		point_sum += pixels[i];

		//用于更新v_bgr

			//获取当前像素坐标位置
		cv::Point p = pixels[i];
		cv::Vec3b v = origin_img.at<cv::Vec3b>(p);

		//存储超像素点的BGR信息
		v_bgr.at<cv::Vec3b>(i)[0] = v[0];
		v_bgr.at<cv::Vec3b>(i)[1] = v[1];
		v_bgr.at<cv::Vec3b>(i)[2] = v[2];
		{
			//v_bgr = v;
			/*unsigned char rgb[3];
			rgb[0] = unsigned char(v[0]);
			rgb[1] = unsigned char(v[1]);
			rgb[2] = unsigned char(v[2]);
			std::cout << int(rgb[0]) << " " << int(rgb[1]) << " " << int(rgb[2]) << endl ;
			double lab[3];*/
			/*cv::cvtColor(v_bgr, v_lab, CV_RGB2Lab);

			cout << int(v_bgr.at<unsigned char>(0)) << " " << int(v_bgr.at<unsigned char>(1)) << " " << int(v_bgr.at<unsigned char>(2)) << " " << endl;
			cout << int(v_lab.at<unsigned char>(0))*100.0/255.0 << " " << int(v_lab.at<char>(1))+128 << " " << int(v_lab.at<char>(2))+128 << " " << endl << endl;*/


			//获得该像素点坐标


			/*RGB2Lab(rgb, lab);*/
			//
			/*std::cout << lab[0] <<" "<< lab[1] <<" " << lab[2] << endl << endl;*/
			//将b g r信息赋值
			////此处错误！应用指针赋值
			//v_bgr.at<cv::Vec3b>(j)[0] = v[0]; j++;
			///*std::cout << v_bgr.at<int>(j - 1)<<" " ;*/
			//v_bgr.at<int>(j) = v[1]; j++;
			///*std::cout << v_bgr.at<int>(j - 1)<<" " ;*/
			//v_bgr.at<int>(j) = v[2]; j++;
			///*std::cout << v_bgr.at<int>(j - 1) << std::endl;*/
		}

		float depth = pixels_depth[i];
		if (!is_zero(depth))
		{
			depth_num++;
			depth_sum += depth;
			depth_min = depth < depth_min ? depth : depth_min;
			depth_max = depth > depth_max ? depth : depth_max;
		}
	}

	cv::cvtColor(v_bgr, v_lab, CV_BGR2Lab);
	//RGB2Lab()
	//cv::Mat v_lab(pixel_num * 3, 1, CV_8UC3);
	//hist.resize(pixel_num * 3);
	//cv::cvtColor(v_bgr, v_lab, CV_BGR2Lab, 3);

	{/*std::ofstream fout("bgr_lab.txt");
	fout << "bgr" << std::endl;
	for (int i = 0; i < pixel_num; i++)
	{
		fout << int(v_bgr.at<cv::Vec3b>(i)[0]) << " " << int(v_bgr.at<cv::Vec3b>(i)[1]) << " " << int(v_bgr.at<cv::Vec3b>(i)[2]) << std::endl;
	}
	fout << std::endl;
	fout << "lab" << std::endl;
	for (int i = 0; i < pixel_num; i++)
	{
		fout << int(v_lab.at<cv::Vec3b>(i)[0])*100.0/255.0 << " " << int(v_lab.at<cv::Vec3b>(i)[1])-128 << " " << int(v_lab.at<cv::Vec3b>(i)[2])-128 << std::endl;
	}


	fout.close();*/}
	//cv::Mat hist;
	cv::Mat Lab_planes[3];
	//将三个通道分割开
	split(v_lab, Lab_planes);

	// 设定bin数目
	int histSize = 20;

	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat L_hist, a_hist, b_hist;
	Mat L_hist_normal, a_hist_normal, b_hist_normal;
	/// 计算直方图:
	calcHist(&Lab_planes[0], 1, 0, Mat(), L_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&Lab_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&Lab_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);


	/// 将直方图归一化到范围
	normalize(L_hist, L_hist_normal, 1, 0, NORM_L1);
	normalize(a_hist, a_hist_normal, 1, 0, NORM_L1);
	normalize(b_hist, b_hist_normal, 1, 0, NORM_L1);

	cv::Mat normals[] = { L_hist_normal,a_hist_normal,b_hist_normal };
	cv::vconcat(normals, 3, hist);

	center = point_sum / pixel_num;
	if (depth_num == 0)
		depth_average = 0;
	else
		depth_average = depth_sum / depth_num;

	


	//生成mask
	Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	for (int i = 0; i < pixels.size(); i++)
	{
		Point& temp_pixel = get_pixel(i);
		mask.at<uchar>(temp_pixel) = 1;
	}
	// 计算轮廓
	vector<vector<Point>> temp_contour;
	findContours(mask, temp_contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	contour = temp_contour[0];
	
}
//超像素产生函数(不产生hist)
void SuperPixel::create()
{
	pixel_num = pixels.size();
	// 计算中心点和平均深度
	Point point_sum(0, 0);
	float depth_sum = 0;
	depth_min = FLT_MAX;
	depth_max = 0;
	depth_num = 0;

	for (int i = 0, j = 0; i < pixel_num; i++)
	{
		point_sum += pixels[i];
		float depth = pixels_depth[i];
		if (!is_zero(depth))
		{
			depth_num++;
			depth_sum += depth;
			depth_min = depth < depth_min ? depth : depth_min;
			depth_max = depth > depth_max ? depth : depth_max;
		}
	}


	center = point_sum / pixel_num;
	if (depth_num == 0)
		depth_average = 0;
	else
		depth_average = depth_sum / depth_num;

	//生成mask
	Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	for (int i = 0; i < pixels.size(); i++)
	{
		Point& temp_pixel = get_pixel(i);
		mask.at<uchar>(temp_pixel) = 1;
	}
	// 计算轮廓
	vector<vector<Point>> temp_contour;
	findContours(mask, temp_contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	contour = temp_contour[0];
}


void ImgData::save_depth_image()
{
	cout << "--save_depth_image..." << std::flush;
	// 用彩色表示深度图
	Mat hue_mat;		// 映射到色相0~360
	normalize(depth_mat, hue_mat, 255.0, 0, NORM_MINMAX);

	Mat hsv_pic(HEIGHT, WIDTH, CV_8UC3);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Vec3b color{unsigned char(int(hue_mat.at<float>(Point(x, y)))), 100, 255 };
			hsv_pic.at<Vec3b>(Point(x, y)) = color;
		}
	}
	cvtColor(hsv_pic, hsv_pic, CV_HSV2BGR);			// 转换为BGR空间
	imwrite(path_output + "\\depth_map.png", hsv_pic);
	cout << "OK" << endl;
}

void ImgData::save_sp_image()
{
	cout << "--save_superpixel_image..." << std::flush;
	Mat sp_img = origin_img.clone();
	sp_img.setTo(Scalar(255, 255, 255), sp_contour);
	imwrite(path_output + "\\superpixel.png", sp_img);
	cout << "OK" << endl;
}

void ImgData::reset()
{
	for (int i = 0; i < sp_num; i++)
	{
		get_superpixel(i).discovered = false;
		get_superpixel(i).priority = DBL_MAX / 1024;
	}
}

//产生天空
void ImgData::show_sky()
{
	//初始化为false
	vector<bool>discover(sp_num, false);
	//存储sp的秩
	queue<int>Queue;
	for (int x = 0; x < WIDTH; x++)
	{
		//若其5上的那一行没有深度且未被遍历过，且没有深度
		if (!discover[sp_label.at<int>(Point(x, 5))]
			&& !get_superpixel(sp_label.at<int>(Point(x, 5))).have_depth())
		{
			discover[sp_label.at<int>(Point(x, 5))] = true;
			Queue.push(sp_label.at<int>(Point(x, 5)));
		}
	}

	while (!Queue.empty())
	{
		int sp_rank = Queue.front();//目前superpixel的编号
		Queue.pop();
		SuperPixel& sp = get_superpixel(sp_rank);
		for (int i = 0; i < sp.contour.size(); i++)
		{
			int current_sp_rank;
			//边缘点
			Point ct = sp.contour[i];
			//push进去的都是没深度点
			for (int j = 0; j < 4; j++)  //邻域
			{
				if ((check_range(ct + Point(inX[j], inY[j])))
					&& (!discover[current_sp_rank = sp_label.at<int>(ct + Point(inX[j], inY[j]))])  //没找过
					&& (!get_superpixel(current_sp_rank).have_depth()) //没深度
					&& (sky_color(current_sp_rank)))//是蓝色
				{
					discover[current_sp_rank] = true;
					//将其标记为天空
					/*get_superpixel(current_sp_rank).state = SKY;*/
					Queue.push(current_sp_rank);
				}
			}
		}
		//将本SPnumber的深度置为无穷
		for (int m = 0; m < sp.pixel_num; m++)
		{
			sp.pixels_depth[m] = WQT_DEPTH; //FLT_MAX//1117586350
			depth_mat.at<int>(Point(sp.pixels[m].x, sp.pixels[m].y)) = WQT_DEPTH;
		}
		sp.create();
	}
	save_depth_image();
	//output(image);
}

void ImgData::debug_depth(const cv::Point &pos)
{
	std::ofstream fout(PATH_DEBUG_DEPTH_OUTPUT);
	SuperPixel SPxl = get_superpixel(sp_label.at<int>(pos));
	for (int i = 0; i < SPxl.pixel_num; i++)
		fout << SPxl.pixels_depth[i] << endl;
	fout.close();
}

bool ImgData::sky_color(const SuperPixel & sp)
{
	cv::Vec3b v = origin_img.at<cv::Vec3b>(sp.center);
	unsigned char B = v[0]; //B
	unsigned char G = v[1]; //G
	unsigned char R = v[2]; //R
	return (((B - G) > unsigned char(25)) && ((B - R) > unsigned char(25)));
}

void ImgData::create_path()
{
	cout << "--create_path " << path_output << "..." << std::flush;
	if (_access(path_output.c_str(), 0) == 0)
	{
		string command = "rd /s/q " + path_output;
		system(command.c_str());
	}
	_mkdir(path_output.c_str());
	cout << "OK" << endl;
}

//产生超像素
void ImgData::generate_sp()
{
	// 把每个点加入到超像素对象中
	data.resize(sp_num);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			//sp_label存储每个点的秩
			int superpixel_rank = sp_label.at<int>(Point(x, y));
			get_superpixel(superpixel_rank).pixels.push_back(Point(x, y));
			get_superpixel(superpixel_rank).pixels_depth.push_back(depth_mat.at<float>(Point(x, y)));
		}
	}
	for (int i = 0; i < data.size(); i++)
	{
		data[i].create(origin_img);
	}
}

//从p_begin点出发寻找最近的有深度的点 
float ImgData::find_depth(Point& p_begin)
{
	//visited相当于布尔型二阶矩阵，标记每个点是否已经discovered
	Mat visited(HEIGHT, WIDTH, CV_8UC1);
	queue<Point> queue_p;
	queue_p.push(p_begin);
	while (true)
	{
		Point p_next = queue_p.front();
		queue_p.pop();
		if (!check_range(p_next))
			continue;
		if (visited.at<int>(p_next) != 0)
			continue;

		visited.at<int>(p_next) = 1;	//记录该节点已访问
		float depth = get_pixel_depth(p_next);

		if (!is_zero(depth))
		{
			p_begin = p_next;		//更新输入顶点的值
			cout << "find depth point: " << p_next << " depth = " << depth << endl;
			return depth;
		}
		//把相邻的顶点加入队列
		queue_p.push(p_next + Point(-1, 0));
		queue_p.push(p_next + Point(1, 0));
		queue_p.push(p_next + Point(0, -1));
		queue_p.push(p_next + Point(0, 1));
	}
}

void ImgData::calc_world_center()
{
	//创建零向量（3*1）
	Mat temp = Mat::zeros(3, 1, CV_32F);
	int depth_num = 0;
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			//查看该点的深度信息
			float depth = get_pixel_depth(Point(x, y));
			//将有深度信息者加入temp中
			if ((depth > 1e-6) && ((depth - WQT_DEPTH) > 1e-6) && ((WQT_DEPTH - depth) > 1e-6))
			{
				temp += cam.get_world_pos(Point(x, y), depth);
				depth_num++;
			}
		}
	}
	//世界中心 为 所有有深度信息的像素点的 平均坐标值 （x,y,z）
	world_center = temp / depth_num;
}

SuperPixel& ImgData::get_superpixel(int i) 
{ 
	return data[i]; 
}

float& ImgData::get_pixel_depth(Point& point)
{
	return depth_mat.at<float>(point);
}

//第二步改进函数（？）
void shape_preserve_warp(ImgData& imgdata, Camera& novel_cam, Mat& output_img, int thread_rank)
{
	cout << "--thread--" << thread_rank << "--begin shape_preserve_warp..." << endl;
	clock_t start;
	clock_t end;
	start = clock();

	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Mat wrap_img_depth = Mat::zeros(HEIGHT, WIDTH, CV_32F);			//记录wrap后img的深度图
	Mat reproject_mat, reproject_vec;
	//计算重投影到novel_cam的mat,vec
	imgdata.cam.fill_reprojection(novel_cam, reproject_mat, reproject_vec);

	// 计算每个超像素在新视点下的深度
	vector<float> depth_dict(imgdata.sp_num);
	for (int i = 0; i < imgdata.sp_num; i++)
	{
		SuperPixel& superpixel = imgdata.get_superpixel(i);
		if (!superpixel.have_depth())
			continue;
		//temp_mat指 超像素点-->相机所在空间点的向量
		Mat temp_mat = novel_cam.cam_pos - imgdata.cam.get_world_pos(superpixel.center, superpixel.depth_average);
		//超像素点-->相机所在空间点的距离
		depth_dict[i] = sqrt(temp_mat.dot(temp_mat));
	}


	clock_t t1 = 0;


	// 逐个超像素进行warp
	for (int i = 0; i < imgdata.sp_num; i++)
	{
		SuperPixel& superpixel = imgdata.get_superpixel(i);
		if (!superpixel.have_depth())
			continue;
		vector<Point2f> triangle;
		minEnclosingTriangle(superpixel.pixels, triangle);			// 计算最小外接三角形

																	//calculate ep
        Eigen::Matrix<float, 6, 6> ep_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1 >ep_vec_eigen = Eigen::Matrix<float, 6, 1>::Zero();

        Eigen::MatrixXf A_mat = Eigen::MatrixXf::Zero(superpixel.pixel_num * 2, 6);
        Eigen::MatrixXf b_mat = Eigen::MatrixXf::Zero(superpixel.pixel_num * 2, 1);

		//对超像素点的每个像素点进行处理
        vector<float> coefficient(3);
		{
			for (int j = 0; j < superpixel.pixel_num; j++)
			{
				Point& origin_point = superpixel.get_pixel(j);
				float point_depth = superpixel.pixels_depth[j];
				// 检验是否有深度
				if (point_depth < 1e-6)
					continue;

				// 插值到外接三角形里，得到用三个定点表示的参数

				tri_interpolation(triangle, origin_point, coefficient);

				// 计算在新视点下的像素坐标	
				Point destination_point = cal_reprojection(origin_point, point_depth, reproject_mat, reproject_vec);

                for (int i = 0; i < 3; i++) A_mat(2 * j, i) = A_mat(2 * j + 1, i + 3) = coefficient[i % 3];
                b_mat(2 * j, 0) = destination_point.x;
                b_mat(2 * j + 1, 0) = destination_point.y;
			}
		}

        ep_mat_eigen = A_mat.transpose()*A_mat;
        ep_vec_eigen = A_mat.transpose()*b_mat;

		// 计算es_mat，衡量三角形的形变量
        Eigen::Matrix<float, 6, 6> es_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
		{
			int j, k, l;
			for (int iter_time = 0; iter_time < 3; iter_time++)
			{
				switch (iter_time)
				{
				case 0:
					j = 0; k = 1; l = 2;
					break;
				case 1:
					j = 1; k = 2; l = 0;
					break;
				case 2:
					j = 2; k = 0; l = 1;
					break;
				default:
					break;
				}

				Point2f& pj = triangle[j]; float xj = pj.x, yj = pj.y;
				Point2f& pk = triangle[k]; float xk = pk.x, yk = pk.y;
				Point2f& pl = triangle[l]; float xl = pl.x, yl = pl.y;

                Eigen::Vector2f p1(xk, yk);
                Eigen::Vector2f p2(xj, yj);
                Eigen::Vector2f p3(xl, yl);

                Eigen::Matrix2f R90;
                R90 << 
                    0, 1,
                    -1, 0;

                Eigen::Matrix<float, 2, 6> A = Eigen::Matrix<float, 2, 6>::Zero();
                float a = (p3 - p1).dot(p2 - p1) / (p2 - p1).squaredNorm();
                float b = (p3 - p1).dot(R90*(p2 - p1)) / (p2 - p1).squaredNorm();

                //E=|p3-p1-a(p2-p1)-bR90(p2-p1)|^2
                Eigen::Matrix2f Aj, Ak, Al; 
                Ak <<
                    (-1 + a), b,
                    -b, (-1 + a);
                Aj <<
                    -a, -b,
                    b, -a;
                Al << 
                    1, 0,
                    0, 1;
                A.col(j) = Aj.col(0);
                A.col(j + 3) = Aj.col(1);
                A.col(k) = Ak.col(0);
                A.col(k + 3) = Ak.col(1);
                A.col(l) = Al.col(0);
                A.col(l + 3) = Al.col(1);

                Eigen::Matrix<float, 6, 6 >ATA = A.transpose()*A;
                es_mat_eigen += ATA;
			}
		}

		// 求逆矩阵，计算在新视点下的外接三角形
		float es_weight = 1;

        Eigen::Matrix<float, 6, 6> temp_mat_eigen = ep_mat_eigen + es_mat_eigen*es_weight;
        if (temp_mat_eigen.determinant() < 1e-6) continue;
        Eigen::LDLT<Eigen::MatrixXf> ldlt(temp_mat_eigen);
        Eigen::VectorXf x = ldlt.solve(ep_vec_eigen);

		vector<Point2f> novel_triangle;
		novel_triangle.resize(3);
		novel_triangle[0] = Point2f(x(0), x(3));
		novel_triangle[1] = Point2f(x(1), x(4));
		novel_triangle[2] = Point2f(x(2), x(5));

		//如果面积之比大于4，则跳过
		float origin_area = calc_triangle_area(triangle);
		float new_area = calc_triangle_area(novel_triangle);
		if (new_area / origin_area > 4)
			continue;


		// 把原来超像素的轮廓用三角形插值投影到新视点下
        vector<Point> novel_contour(superpixel.contour.size());

        vector<Point> novel_points;
		for (int j = 0; j < superpixel.contour.size(); j++)
		{
			Point& origin_point = superpixel.contour[j];
			tri_interpolation(triangle, origin_point, coefficient);
            novel_contour[j] = (inv_tri_interpolation(novel_triangle, coefficient));
		}

		// 用投影后的轮廓得到投影后的超像素区域
		contour_to_set(novel_contour, novel_points);

		for (int j = 0; j < novel_points.size(); j++)
		{
			Point& novel_point = novel_points[j];
			vector<float> coefficient;
			tri_interpolation(novel_triangle, novel_point, coefficient);
			Point reproject_point = inv_tri_interpolation(triangle, coefficient);
			if (check_range(reproject_point) && check_range(novel_point))
			{
				float& before_depth = wrap_img_depth.at<float>(novel_point);
				if (abs(before_depth - 0) < 1e-6 || depth_dict[i] < before_depth)
				{
					before_depth = depth_dict[i];
					output_img.at<Vec3b>(novel_point) = imgdata.origin_img.at<Vec3b>(reproject_point);
				}
			}
		}
	}
	end = clock();
	cout << "--thread--" << thread_rank << "--end using time:..." << (end - start) << "ms" << endl;
}

//产生lab直方图
cv::Mat labHist(const cv::Mat &src, int lbins, int abins, int bbins)
{
	//Mat dst(src);
	//颜色空间的转换 BGR2Lab  
	//cvtColor(src,lab,CV_BGR2Lab);  

	//L,a,b三个通道分别为 4,14,14bins  
	int histSize[] = { lbins , abins , bbins };

	//L的取值范围 0-255  
	float lranges[] = { 0,256 };
	//a的取值范围   
	float aranges[] = { 0,256 };
	//b的取值范围  
	float branges[] = { 0,256 };
	const float* ranges[] = { lranges ,aranges , branges };

	Mat hist3D, hist3dNormal;
	Mat dst = Mat(lbins*abins*bbins, 1, CV_8UC1);

	const int channels[] = { 0,1,2 };
	calcHist(&src, 1, channels, Mat(), hist3D, 3, histSize, ranges, true, false);//hist3D是32F  
																				 //归一化,64F  
	normalize(hist3D, hist3dNormal, 1, 0, CV_L1, CV_8U);

	//第二种方法取得三维直方图中的值  
	//double* p = (double*)hist3D.data;  
	int row = 0;
	for (int l = 0; l < lbins; l++)
	{
		for (int a = 0; a < abins; a++)
		{
			for (int b = 0; b < bbins; b++)
			{
				dst.at<double>(row, 0) = *((double*)(hist3dNormal.data + l*hist3dNormal.step[0] + a*hist3dNormal.step[1] + b*hist3dNormal.step[2]));
				//hist.at<double>(row,0) = *(p+row);  
				row++;
			}
		}
	}
	return dst;
}

void mix_pic(vector<ImgData>& imgdata_vec, Camera& now_cam, vector<int>& img_id, Mat& output_img)
{
	cout << "--begin generate pic..." << endl;
	clock_t start;
	clock_t end;
	start = clock();
	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	vector<Mat> wrap_img(img_id.size());
	vector<thread> threads(img_id.size());
	for (int i = 0; i < img_id.size(); i++)
	{
		//对四个最近的照片分别wrap到同一个视点
		threads[i] = thread(shape_preserve_warp, ref(imgdata_vec[img_id[i]]), ref(now_cam), ref(wrap_img[i]), i);
	}
	//等待所有线程执行完毕
	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}
	//图像融合
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Point point(x, y);
			for (int i = 0; i < img_id.size(); i++)
			{
				//按照片远近优先级赋值
				if (wrap_img[i].at<Vec3b>(point) != Vec3b{ 0,0,0 })
				{
					output_img.at<Vec3b>(point) = wrap_img[i].at<Vec3b>(point);
					break;		//一旦找到一张wrap后图像有图像信息，则赋值完跳出循环
				}
			}

		}
	}

	end = clock();
	cout << "--OK using time:" << (end - start) << "ms" << endl;

}



double chiSquareDist(const cv::Mat &hist1, const cv::Mat & hist2)
{
	int rows = hist1.rows;
	double sum = 0.0;
	double d1, d2;
	for (int r = 0; r < rows; r++)
	{
		d1 = hist1.at<double>(r, 0);
		d2 = hist2.at<double>(r, 0);
		if (d1 == 0 && d2 == 0)
			;
		else
			sum +=  pow(d1 - d2, 2) / (d1 + d2);
	}
	return sum;
}

double sp_Pair::calcChiSquare()
{
	return chiSquareDist(sp_src.hist, sp_dst.hist);
}


void ImgData::depth_synthesis()
{
	ofstream fout("commands.txt");
	//sim_graph.resize(sp_num);
	Near_sp.resize(sp_num);

	//邻接表的纵向长度为sp_num
	adjacency_list.resize(sp_num);
	int depthless_num = 0;
	//fout << "start Graphing!" << std::endl;
	int index_Near_sp = 0;
	for (int i = 0; (i<sp_num); i++)
	{
		//sim_graph[i].resize(sp_num);
		//此处必须用引用，因为要作为左值使用
		SuperPixel &pixel_now = data[i];
		if (!data[i].have_depth())depthless_num++;
		//↓可删除
		if (!data[i].have_depth()) sp_no_depth.push_back(data[i]);
		//↑可删除

		// 构建一个小顶堆的向量；可以加到SuperPixel类里当成员
		std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp> current_nbr;
		
		/*vector<priority_queue<sp_Pair, vector<sp_Pair>, pair_cmp>> Near_sp;
		Near_sp.resize(sp_num);*/
		//建图
		
		for (int j = 0, k = 0;j<sp_num; j++)
		{
			if (i == j)continue;
			SuperPixel &pixel_neigh = data[j];
			//若是邻居
			if (pixel_now.neighbor(data[j], j, sp_label))
			{
				//fout << "neighbor found!" << std::endl;
				double similarity = chiSquareDist(data[i].hist, data[j].hist);
				//邻接表存储
				adjacency_list[i].push_back(adjacency_list_node(j, similarity));
				//sim_graph[i][j] = similarity;

				//将邻居存储至SP类中
				//pixel_now.neighbors_ranks.push_back(j);
			}
			/*else
				sim_graph[i][j] = DBL_MAX;*/

			//建立该元素的最小的40个,且data[j]!=SKY
			if ((!data[i].have_depth()) && (data[j].have_depth() && (data[j].depth_average < 100000)) /*&& (data[j].state != SKY)*/)
			{
				//fout << "depthless point neighbor found!" << std::endl;
				sp_Pair pair(data[i], data[j]);
				current_nbr.push(pair);
				if (current_nbr.size() > 40) current_nbr.pop();
				//fout << " data[i].have_depth()= " << data[i].have_depth() << "  data[j].have_depth()= " << data[j].have_depth() << std::endl;
				//把它压进去；
				//弹出小顶堆的第一个;
			}
		}
		if (!data[i].have_depth())
		{
			Near_sp[index_Near_sp++] = (current_nbr);
			/*fout << "one heap set up!" << std::endl;
			fout << " pixel_now.neighbors_ranks.size() " << pixel_now.neighbors_ranks.size() << std::endl;
			fout << " get_superpixel(i).neighbors_ranks.size() = " << get_superpixel(i).neighbors_ranks.size() << std::endl;*/
		}
		//如果它是无深度超像素，那么把大顶堆转化成向量，即为最相似的40个超像素
		
	}
	//fout << "sp_num = " << sp_num << std::endl;
	//fout << "depthless_num = " << depthless_num << std::endl;
	//fout << "start DIJKSTRA ing!" << std::endl;
	//此处应该进行Dijkstra
	Near_sp.resize(index_Near_sp);

	fout << Near_sp.size() << std::endl;
	for (int i = 0; (i < (Near_sp.size())); i++)
	{
		std::cout << "DEEPLESS No: " << i << std::endl;
		//fout << "DEEPLESS No: " << i << std::endl;
		int count = 0;
		//fout << "No " <<i<<" depthless point start dijkstra-ing "<< std::endl;
		//current_nbr指40个相似点,此处应该使用大顶堆
		std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp> &current_nbr = Near_sp[i];
		//fout << "1" << " ";
		//nbr_vec指40个相似点的秩
		std::vector<int> nbr_vec;

		nbr_vec.resize(40);
		//必须转化为引用
		//sp_Pair &current_pair = const_cast<sp_Pair &>(Near_sp[i].top());
		SuperPixel &current_sp = const_cast<SuperPixel &>(Near_sp[i].top().sp_src);
		SuperPixel source = Near_sp[i].top().sp_src;
		//将优先级队列中N[S]转存到vector中
		int index = 0;
		//fout << "2" << " ";
		while (!Near_sp[i].empty())
		{
			//fout << "3" << " ";
			sp_Pair p = Near_sp[i].top(); nbr_vec[index++] = get_sp_rank(p.sp_dst);
			/*nbr_vec.push_back(get_sp_rank(p.sp_dst));*/ 
			//fout << "nbr_vec [i]" << get_sp_rank(p.sp_dst) << std::endl;
			Near_sp[i].pop();
			//fout << "4" << " ";
		}
		//fout << "5" << " ";
		//fout << " queue to nbr_vec transformed " << std::endl;
		//重置,初始化
		reset();
		current_sp.discovered = false;
		current_sp.priority = 0;
		//优先级队列
		std::priority_queue<SuperPixel, std::vector<SuperPixel>, sp_cmp> q;
		//fout << "current_sp_rank = " << get_sp_rank(current_sp) << std::endl;
		//fout << " start traversing " << std::endl;
		int cur_rank = get_sp_rank(current_sp);
		q.push(current_sp);
		while (true)
		{
			if (q.empty())
			{
				//fout << "Queue empty!" << std::endl;
				break;
			}
			current_sp = q.top(); q.pop();
			if (current_sp.discovered)continue;
			get_superpixel(get_sp_rank(current_sp)).discovered = true;
			//fout << "current_sp_rank = " << get_sp_rank(current_sp) << std::endl;
			//fout << "current priority = " << current_sp.priority << std::endl;
			//检查当前节点是否为40个节点之一
			for (int i = 0; i < nbr_vec.size(); i++)
			{
				//fout << "nbr_vec: " << nbr_vec[i] << " center: " << get_superpixel(nbr_vec[i]).center;
				//fout << "current: " << cur_rank << " center: " << current_sp.center;
				if (get_superpixel(nbr_vec[i]).center == current_sp.center)
				{
					count++;
				}
			}
			//std::cout << " count =  " << count << std::endl;
			//fout << " count =  " << count << std::endl;
			//直至3个顶点被加入
			if (count >= 3)break;
			
			////更新优先级,先遍历所有的邻居
			//fout << " current_sp.neighbors_ranks.size() = " << adjacency_list[get_sp_rank(current_sp)].size() << std::endl;
			for (int j = 0,cur_rank = get_sp_rank(current_sp); j < adjacency_list[get_sp_rank(current_sp)].size(); j++)
			{
				//邻居点的秩
				int nbr_rank = adjacency_list[cur_rank][j].sp_rank; //fout << "j = " << j << " nbr_bank = " << nbr_rank << " ";
				//fout << "discovered? " << get_superpixel(nbr_rank).discovered << std::endl;
				//更新优先级
				if (!get_superpixel(nbr_rank).discovered)
				{
					
					//fout << " undiscovered neighbor found! " << std::endl;
					//SuperPixel &nbr = get_superpixel(adjacency_list[get_sp_rank(current_sp)][j].sp_rank);
					//fout << "nbr.priority=" << get_superpixel(nbr_rank).priority << " current_sp.priority = " << current_sp.priority << " edgeCost = " << adjacency_list[cur_rank][j].edgeCost << std::endl;
					
					//若邻居点的优先级					 大于 现结点的优先级	  +	两点之间的边权重
					//且其超像素不为天空
					if ((get_superpixel(nbr_rank).priority > (current_sp.priority + adjacency_list[cur_rank][j].edgeCost))
						/*&& (get_superpixel(nbr_rank).state != SKY)*/)
					{
						//fout << " priority updated! " << std::endl;
						get_superpixel(nbr_rank).priority = current_sp.priority + adjacency_list[cur_rank][j].edgeCost;
						//fout << "UPDATED::nbr.priority=" << get_superpixel(nbr_rank).priority << std::endl;
						//将更新后的邻居塞入队列中
						q.push(get_superpixel(nbr_rank));
					}
				}
			}
			//fout << " ALL priority updated! " << std::endl;
			//fout << "q.size() = " << q.size() << std::endl;
		}
		//找到最小的1个向量，并进行插值
		double shortest = DBL_MAX / 1024; int shortest_rank = 0;
		for (int j = 0; j < 40; j++)
		{
			//若该找到的点不为自己
			if (get_superpixel(nbr_vec[j]).center == source.center)continue;
			if ((shortest > get_superpixel(nbr_vec[j]).priority) 
				&& (get_superpixel(nbr_vec[j]).priority > 1e-6) 
				&& get_superpixel(nbr_vec[j]).have_depth()
				/*&& (get_superpixel(nbr_vec[j]).state != SKY)*/)
			{
				shortest = get_superpixel(nbr_vec[j]).priority;
				shortest_rank = j;
				//fout << " shortest updated !" << std::endl;
				//fout << " shortest =  " << shortest << std::endl;
			}
		}
		//为深度赋值
		//fout << " start  assigning depth!" << std::endl;
		for (int k = 0; k < source.pixel_num; k++)
		{
			get_pixel_depth(source.pixels[k]) = get_superpixel(nbr_vec[shortest_rank]).depth_average;
			//depth_mat.at<int>(source.pixels[k]) = get_superpixel(nbr_vec[shortest_rank]).depth_average;
			//fout << "pixel_num" << k << std::endl;
		}
		//get_superpixel(get_sp_rank(source)).create();
	}
	//fout << "UPDATING!!!!!!!!!" << endl;
	//fout << "UPDATING!!!!!!!!!" << endl;
	fout << "UPDATING!!!!!!!!!" << endl;

	save_depth_image();

	//fout << "new depth_image Saved!" << endl;

	fout.close();
	//以上构建了相邻图和无深度点的最相似超像素

	//接下来要做的是：从Pixel_no_depth队列里依次弹出超像素，求到相似超像素的最短路径的三个最小，并把这三个的深度信息存下来
}

