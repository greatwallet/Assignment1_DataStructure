#pragma once
#ifndef IMG_DATA_H
#define IMG_DATA_H



#include "camera.h"
#include <functional>
#include <queue>
#include <vector>

//�ڽӾ����������Ⱥ����ƶ�
struct sp_nbr_similarity
{
	int Rank_nbr;
	double nbr_similarity;
	sp_nbr_similarity(const int &rank, const double &similarity) :Rank_nbr(rank), nbr_similarity(similarity) {}
};
cv::Mat labHist(const cv::Mat& src, int lbins = 20, int abins = 20, int bbins = 20);

//�ڽӱ�ڵ���
struct adjacency_list_node
{
	//��������imgdata�е���
	int sp_rank;
	//�����صıߵ�ֵ
	float edgeCost;

	adjacency_list_node() {};
	adjacency_list_node(int sp_rank, float edgeCost)
		:sp_rank(sp_rank),edgeCost(edgeCost){};
};

enum sp_state { SKY, OTHERS };

class sp_Pair;
struct pair_cmp;
//��������
class SuperPixel
{
public:
	//���ĵ�
	cv::Point center;
	//��������ļ��ϣ�
	std::vector<cv::Point> contour;
	//�ڲ����أ���ļ��ϣ�
	std::vector<cv::Point> pixels;
	//������ȣ����飩
	std::vector<float> pixels_depth;

	//ֱ��ͼ
	cv::Mat hist;

	//std::priority_queue <sp_Pair, std::vector<sp_Pair>, pair_cmp> Near_sp;

	int pixel_num;
	int depth_num;
	float depth_average;
	float depth_max;
	float depth_min;
	//����ó������ھӵ���
	std::vector<int> neighbors_ranks;

	//���ȼ�
	double priority;
	//�Ƿ񱻷���
	bool discovered;
	SuperPixel() 
		:priority(DBL_MAX),discovered(true),state(OTHERS)
	{
		hist = cv::Mat();
	}
	~SuperPixel() {}

	//��ȡ��������
	double chiSquareDist(const SuperPixel &right);

	//��ȡĳ������أ�����õ���ȣ�
	cv::Point& get_pixel(int i);
	//���������ؽڵ㣨����hist��
	void create(cv::Mat &origin_img);
	void create();
	//�ж���ǰ�����ص��Ƿ��������Ϣ
	bool have_depth();
	//neighbor or not
	bool neighbor(const SuperPixel& neigh, int rank_neigh, cv::Mat sp_label);

	//�������ھӵ����߾���
	bool posneigh(cv::Point cen_neigh);
	bool posneigh(const SuperPixel &neigh);
	sp_state state;
private:
};

struct sp_cmp
{
	bool operator() (const SuperPixel &left, const SuperPixel &right)
	{
		return left.priority > right.priority;	//��Сֵ����
	}

};

//ͼƬ��Ϣ��
class ImgData
{
public:
	//����
	int id;
	//�������
	Camera cam;
	//�������Ϣ��ͼƬ
	cv::Mat depth_mat;
	//
	cv::Mat world_center;
	//ԭʼͼƬ
	cv::Mat origin_img;
	//�����ص��ǩ������
	cv::Mat sp_label;
	//�����ص�����������
	cv::Mat sp_contour;
	//�����ص����Ŀ
	int sp_num;

	int get_sp_rank(const SuperPixel &sp);
	//����ȶ���
	std::vector <SuperPixel> sp_no_depth;

	std::vector<SuperPixel> data;
	std::string path_output;

	std::vector<std::priority_queue<sp_Pair, std::vector<sp_Pair>, pair_cmp>> Near_sp;

	//����
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	//��ȡ�����ص�
	SuperPixel& get_superpixel(int i);
	//��ȡĳ��������Ϣ
	float& get_pixel_depth(cv::Point& point);
	//��һ��
	void depth_synthesis();

	//�ڽӾ���
	std::vector<std::vector<sp_nbr_similarity>> sim_graph;

	//�ڽӱ�
	std::vector<std::vector<adjacency_list_node>> adjacency_list;

	//�������
	void show_sky();

	void debug_depth(const cv::Point &pos);

	//ĳ�����ص��Ƿ�Ϊ���ɫ
	bool sky_color(const SuperPixel &sp);
	bool sky_color(int sp_rank)
	{
		return sky_color(get_superpixel(sp_rank));
	}
private:
	//�������ƶ�·��������
	void create_path();
	//�����������ģ�����
	void calc_world_center();
	//���������ص�
	void generate_sp();
	//���������Ϣ
	float find_depth(cv::Point& p_begin);
	//���������Ϣ��ͼƬ
	void save_depth_image();
	//���泬���ص�ͼƬ
	void save_sp_image();
	//��������
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

		return left.similarity < right.similarity;	//�������
	}

};


struct sp_nbr_path
{
	SuperPixel sp_nbr;
	double priority;
};



#endif // !IMG_DATA_H