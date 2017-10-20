#include "imgdata.h"

#include "global_var.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <thread>

#include <string>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>


using namespace std;
using namespace cv;

Point& SuperPixel::get_pixel(int i)
{
	return pixels[i];
}

bool SuperPixel::have_depth()
{
	return 1.0 * depth_num / pixel_num > 0.05;
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
	create_path();
	calc_world_center();
	generate_sp();

	save_depth_image();
	save_sp_image();
}

void SuperPixel::create()
{
	pixel_num = pixels.size();
	// 计算中心点和平均深度
	Point point_sum(0, 0);
	float depth_sum = 0;
	depth_min = FLT_MAX;
	depth_max = 0;
	depth_num = 0;
	for (int i = 0; i < pixel_num; i++)
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

void ImgData::generate_sp()
{
	// 把每个点加入到超像素对象中
	data.resize(sp_num);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			int superpixel_rank = sp_label.at<int>(Point(x, y));
			get_superpixel(superpixel_rank).pixels.push_back(Point(x, y));
			get_superpixel(superpixel_rank).pixels_depth.push_back(depth_mat.at<float>(Point(x, y)));
		}
	}
	for (int i = 0; i < data.size(); i++)
	{
		data[i].create();
	}
}

float ImgData::find_depth(Point& p_begin)
{
	//从p_begin点出发寻找最近的有深度的点 
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
	Mat temp = Mat::zeros(3, 1, CV_32F);
	int depth_num = 0;
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			float depth = get_pixel_depth(Point(x, y));
			if (depth > 1e-6)
			{
				temp += cam.get_world_pos(Point(x, y), depth);
				depth_num++;
			}
		}
	}
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

void shape_preserve_wrap(ImgData& imgdata, Camera& novel_cam, Mat& output_img, int thread_rank)
{
	cout << "--thread--" << thread_rank << "--begin shape_preserve_wrap..." << endl;
	clock_t start;
	clock_t end;
	start = clock();

	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Mat wrap_img_depth = Mat::zeros(HEIGHT, WIDTH, CV_32F);			//记录wrap后img的深度图
	Mat reproject_mat, reproject_vec;
	imgdata.cam.fill_reprojection(novel_cam, reproject_mat, reproject_vec);

	// 计算每个超像素在新视点下的深度
	vector<float> depth_dict(imgdata.sp_num);
	for (int i = 0; i < imgdata.sp_num; i++)
	{
		SuperPixel& superpixel = imgdata.get_superpixel(i);
		if (!superpixel.have_depth())
			continue;
		Mat temp_mat = novel_cam.cam_pos - imgdata.cam.get_world_pos(superpixel.center, superpixel.depth_average);
		depth_dict[i] = sqrt(temp_mat.dot(temp_mat));
	}


	clock_t t1 = 0;


	// 逐个超像素进行wrap
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
		{
			for (int j = 0; j < superpixel.pixel_num; j++)
			{
				Point& origin_point = superpixel.get_pixel(j);
				float point_depth = superpixel.pixels_depth[j];
				// 检验是否有深度
				if (point_depth < 1e-6)
					continue;

				// 插值到外接三角形里，得到用三个定点表示的参数
				vector<float> coefficient;
				tri_interpolation(triangle, origin_point, coefficient);

				// 计算在新视点下的像素坐标	
				Point destination_point = cal_reprojection(origin_point, point_depth, reproject_mat, reproject_vec);

				// 对每个有深度的点，计算ep_mat

                Eigen::Matrix<float, 2, 6> A = Eigen::Matrix<float, 2, 6>::Zero();
                for (int i = 0; i < 3; i++) A(0, i) = A(1, i + 3) = coefficient[i % 3];
                Eigen::Matrix<float, 2, 1> b = Eigen::Matrix<float, 2, 1>::Zero();
                b(0, 0) = destination_point.x;
                b(1, 0) = destination_point.y;
                Eigen::Matrix<float, 6, 6 > ATA = A.transpose()*A;
                Eigen::Matrix<float, 6, 1 > ATb = A.transpose()*b;
                ep_mat_eigen += ATA;
                ep_vec_eigen += ATb;
			}
		}

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
		vector<Point> novel_contour, novel_points;
		for (int j = 0; j < superpixel.contour.size(); j++)
		{
			Point& origin_point = superpixel.contour[j];
			vector<float> coefficient;
			tri_interpolation(triangle, origin_point, coefficient);
			novel_contour.push_back(inv_tri_interpolation(novel_triangle, coefficient));
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
		threads[i] = thread(shape_preserve_wrap, ref(imgdata_vec[img_id[i]]), ref(now_cam), ref(wrap_img[i]), i);
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
