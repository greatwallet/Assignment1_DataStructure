#pragma once
#ifndef IMG_DATA_H
#define IMG_DATA_H



#include "camera.h"
#include <functional>
#include <queue>
#include <vector>

//邻接矩阵存放邻域秩和相似度
struct sp_nbr_similarity
{
	int Rank_nbr;
	double nbr_similarity;
	sp_nbr_similarity(const int &rank, const double &similarity) :Rank_nbr(rank), nbr_similarity(similarity) {}
};
cv::Mat labHist(const cv::Mat& src, int lbins = 20, int abins = 20, int bbins = 20);

//邻接表节点类
struct adjacency_list_node
{
	//超像素在imgdata中的秩
	int sp_rank;
	//超像素的边的值
	float edgeCost;

	adjacency_list_node() {};
	adjacency_list_node(int sp_rank, float edgeCost)
		:sp_rank(sp_rank),edgeCost(edgeCost){};
};

enum sp_state { SKY, OTHERS };

class sp_Pair;
struct pair_cmp;
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

	//std::priority_queue <sp_Pair, std::vector<sp_Pair>, pair_cmp> Near_sp;

	int pixel_num;
	int depth_num;
	float depth_average;
	float depth_max;
	float depth_min;
	//储存该超像素邻居的秩
	std::vector<int> neighbors_ranks;

	//优先级
	double priority;
	//是否被发现
	bool discovered;
	SuperPixel() 
		:priority(DBL_MAX),discovered(true),state(OTHERS)
	{
		hist = cv::Mat();
	}
	~SuperPixel() {}

	//获取卡方距离
	double chiSquareDist(const SuperPixel &right);

	//获取某点的像素（输入该点的秩）
	cv::Point& get_pixel(int i);
	//创建超像素节点（产生hist）
	void create(cv::Mat &origin_img);
	void create();
	//判定当前超像素点是否有深度信息
	bool have_depth();
	//neighbor or not
	bool neighbor(const SuperPixel& neigh, int rank_neigh, cv::Mat sp_label);

	//计算与邻居的折线距离
	bool posneigh(cv::Point cen_neigh);
	bool posneigh(const SuperPixel &neigh);
	sp_state state;
private:
};

struct sp_cmp
{
	bool operator() (const SuperPixel &left, const SuperPixel &right)
	{
		return left.priority > right.priority;	//最小值优先
	}

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

	int get_sp_rank(const SuperPixel &sp);
	//无深度顶点
	std::vector <SuperPixel> sp_no_depth;

	std::vector<SuperPixel> data;
	std::string path_output;

	std::vector<std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp>> Near_sp;

	//函数
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	//获取超像素点
	SuperPixel& get_superpixel(int i);
	//获取某点的深度信息
	float& get_pixel_depth(cv::Point& point);
	//第一步
	void depth_synthesis();

	//邻接矩阵
	std::vector<std::vector<sp_nbr_similarity>> sim_graph;

	//邻接表
	std::vector<std::vector<adjacency_list_node>> adjacency_list;

	//产生天空
	void show_sky();

	void debug_depth(const cv::Point &pos);

	//某超像素点是否为天空色
	bool sky_color(const SuperPixel &sp);
	bool sky_color(int sp_rank)
	{
		return sky_color(get_superpixel(sp_rank));
	}
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
	//遍历重置
	void reset();
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);

class sp_Pair
{
public:
	SuperPixel sp_src;
	SuperPixel sp_dst;
	double similarity;
	sp_Pair()
		:similarity(-1)
	{};
	sp_Pair(const SuperPixel &src, const SuperPixel &dst)
		:sp_src(src),sp_dst(dst)
	{
		similarity = calcChiSquare();
	}
	double calcChiSquare();
};

struct pair_cmp
{
	bool operator() (sp_Pair &left, sp_Pair &right)
	{
		if (left.similarity < 0)left.similarity = left.calcChiSquare();
		if (right.similarity < 0)right.similarity = right.calcChiSquare();

		return left.similarity < right.similarity;	//大的优先
	}

};


struct sp_nbr_path
{
	SuperPixel sp_nbr;
	double priority;
};



#endif // !IMG_DATA_H