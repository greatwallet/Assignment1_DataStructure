#include "imgdata.h"

#include "global_var.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <thread>

#include <string>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>
#include <cmath>

using namespace std;
using namespace cv;


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

//超像素产生函数
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


		float depth = pixels_depth[i];
		if (!is_zero(depth))
		{
			depth_num++;
			depth_sum += depth;
			depth_min = depth < depth_min ? depth : depth_min;
			depth_max = depth > depth_max ? depth : depth_max;
		}
	}

	cv::cvtColor(v_bgr, v_lab, CV_RGB2Lab);

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
			if (depth > 1e-6)
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
void shape_preserve_wrap(ImgData& imgdata, Camera& novel_cam, Mat& output_img, int thread_rank)
{
	cout << "--thread--" << thread_rank << "--begin shape_preserve_wrap..." << endl;
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
	ofstream fout("commands");
	sim_graph.resize(sp_num);
	int depthless_num = 0;
	for (int i = 0; i<sp_num; i++)
	{	
		//sim_graph[i].resize(sp_num);
		SuperPixel &pixel_now = data[i];
		if (!pixel_now.have_depth())depthless_num++;
		std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp> current_nbr;

		//建图
		fout << "start Graphing!" << std::endl;
		for (int j = 0, k = 0;j<sp_num; j++)
		{
			SuperPixel &pixel_neigh = data[j];
			if (pixel_now.neighbor(pixel_neigh, j, sp_label))
			{
				//fout << "neighbor found!" << std::endl;
				double similarity = chiSquareDist(pixel_now.hist, pixel_neigh.hist);
				sp_nbr_similarity nbr_now(j,similarity);
				sim_graph[i].push_back(nbr_now);
				//将邻居存储至SP类中
				//pixel_now.neighbors_ranks.push_back(j);
			}
			/*
			else
				sim_graph[i][j] = DBL_MAX;*/

			//建立该元素的最小的40个
			if ((!pixel_now.have_depth())&&(pixel_neigh.have_depth()))
			{
				sp_Pair pair(pixel_now, pixel_neigh);
				current_nbr.push(pair);
				if (current_nbr.size() > 40) current_nbr.pop();
				//把它压进去；
				//弹出小顶堆的第一个；
			}
		}
		if (!pixel_now.have_depth())
		{
			Near_sp.push_back(current_nbr);
			fout << "one heap set up!" << std::endl;
			fout << " superpixel(i).neighbors_ranks = " << sim_graph[i].size() << std::endl;
		}
		//如果它是无深度超像素，那么把大顶堆转化成向量，即为最相似的40个超像素
		
	}
	fout << "sp_num = " << sp_num << std::endl;
	fout << "depthless_num = " << depthless_num << std::endl;
	fout << "start DIJKSTRA ing!" << std::endl;
	//此处应该进行Dijkstra

	for (int i = 0; i < Near_sp.size()&&i<3; i++)
	{
		fout << "No " <<i<<" depthless point start dijkstra-ing "<< std::endl;
		//current_nbr指40个相似点
		std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp> &current_nbr = Near_sp[i];
		//nbr_vec指40个相似点
		std::vector<SuperPixel> nbr_vec;
		nbr_vec.resize(40);
		//必须转化为引用
		sp_Pair &current_pair = const_cast<sp_Pair &>(Near_sp[i].top());
		SuperPixel &current_sp = current_pair.sp_src;
		//将优先级队列中N[S]转存到vector中
		while (!current_nbr.empty())
		{
			sp_Pair p = current_nbr.top();
			nbr_vec.push_back(p.sp_dst);
			current_nbr.pop();
		}
		fout << " queue to nbr_vec transformed " << std::endl;
		//重置,初始化
		reset();
		current_sp.discovered = true;
		current_sp.priority = 0;
		//优先级队列
		std::priority_queue<SuperPixel, std::vector<SuperPixel>, sp_cmp> q;
		int count = 1;
		fout << " start traversing " << std::endl;
		while (true)
		{
			//先遍历所有的邻居
			fout << " current_sp.neighbors_ranks.size() = " << sim_graph[get_sp_rank(current_sp)].size() << std::endl;
			for (int j = 0; j < sim_graph[get_sp_rank(current_sp)].size(); j++)
			{
				fout << "discovered? " << get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr).discovered << endl;
				//更新优先级
				if (!get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr).discovered)
				{
					get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr).discovered = false;
					fout << " undiscovered neighbor found! " << std::endl;
					SuperPixel &nbr = get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr);
					get_sp_rank(current_sp);
					fout << "nbr.priority=" << nbr.priority << " current_sp.priority = " << current_sp.priority << " edgeCost = " << sim_graph[get_sp_rank(current_sp)][j].nbr_similarity << std::endl;
					if (nbr.priority > (current_sp.priority + sim_graph[get_sp_rank(current_sp)][j].nbr_similarity)) 
					{
						fout << " priority updated! " << std::endl;
						get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr).priority = current_sp.priority + sim_graph[get_sp_rank(current_sp)][j].nbr_similarity;
						q.push(get_superpixel(sim_graph[get_sp_rank(current_sp)][j].Rank_nbr));
					}
				}
			}
			fout << " ALL priority updated! " << std::endl;
			current_sp = q.top(); q.pop();
			for (int i = 0; i < nbr_vec.size(); i++)
				if (nbr_vec[i].center == current_sp.center)count++;
			fout << " count =  " << count << std::endl;
			//直至40个顶点被加入
			if (count >= 3)break;
		}
		//找到最小的1个向量，并进行插值
		double shortest = DBL_MAX; int shortest_rank = 0;
		for (int i = 0; i < 40; i++)
		{
			//若该找到的点不为自己
			if (nbr_vec[i].center == current_sp.center)continue;
			fout << "nbr_vec[i].have_depth()" << nbr_vec[i].have_depth() << std::endl;
			fout << "nbr_vec[i].priority" << nbr_vec[i].priority << std::endl;
			if ((shortest > nbr_vec[i].priority) && (nbr_vec[i].priority > 1e-6) && nbr_vec[i].have_depth())
			{
				shortest = nbr_vec[i].priority;
				shortest_rank = i;
				fout << " shortest updated !" << std::endl;
				fout << " shortest =  " << shortest << std::endl;
			}
		}
		//为深度赋值
		fout << " start  assigning depth!" << std::endl;
		for (int i = 0; i < current_sp.pixel_num; i++)
		{
			get_pixel_depth(current_sp.pixels[i]) = nbr_vec[shortest_rank].depth_average;
			fout << "pixel_num" << i << std::endl;
		}





	}
	fout << "UPDATING!!!!!!!!!" << endl;
	fout << "UPDATING!!!!!!!!!" << endl;
	fout << "UPDATING!!!!!!!!!" << endl;

	save_depth_image();

	fout << "new depth_image Saved!" << endl;

	fout.close();
	//以上构建了相邻图和无深度点的最相似超像素

	//接下来要做的是：从Pixel_no_depth队列里依次弹出超像素，求到相似超像素的最短路径的三个最小，并把这三个的深度信息存下来
}

