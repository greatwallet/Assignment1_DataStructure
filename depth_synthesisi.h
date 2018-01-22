//#include<queue>
//#include<functional>
//#include <opencv2/imgproc.hpp>
//
//struct Pixel_sim
//{
//	bool operator () (SuperPixel s1, SuperPixel s2) // 重载括号
//    {
//        return chiSquareDist(pixel_now.LAB_Hist,s1.LAB_Hist) < chiSquareDist(pixel_now.LAB_Hist,s2.LAB_Hist); // 等同于less
//    }
//	
//}
//void ImgData::depth_synthesis()
//{
//	for (int i=0; i<sp_num; i++)
//	{
//		SuperPixel pixel_now = data[i];
//		if (!pixel_now.have_depth()) Pixel_no_depth.push_bakc(pixel_now);
//		
//		// 构建一个大顶堆；可以加到SuperPixel类里当成员
//		priority_queue<SuperPixel,vector<SuperPixel>,Pixel_sim> Near_pixel;
//		
//		for (int j=0; j<sp_num; j++)
//		{
//			SuperPixel pixel_neigh = data[j];
//			if (neighbor(pixel_now,pixel_neigh,j))
//			{
//				double similarity = chiSquareDist(pixel_now.LAB_Hist,pixel_neigh.LAB_Hist);
//				sim_graph[i][j] = similarity;
//			}
//			
//			if (!pixel_now.have_depth())
//			{
//				Near_pixel.push(Pixel_neigh);
//				if (Near_pixel.size() == 41) Near_pixel.pop();
//				//把它压进去；
//				//弹出大顶堆的第一个；
//			}
//		}	
//			//如果它是无深度超像素，那么把大顶堆转化成向量，即为最相似的40个超像素
//	}
//	
//	//以上构建了相邻图和无深度点的最相似超像素
//	
//	//接下来要做的是：从Pixel_no_depth队列里依次弹出超像素，求到相似超像素的最短路径的三个最小，并把这三个的深度信息存下来
//}
//
//bool Imadata::neighbor(SuperPixel& pixel_now,SuperPixel& pixel_neigh,int rank_neigh)  //neighbor or not
//{
//	if (posneigh(pixel_now.center,pixel_neigh.center))
//	{
//		int contotal = pixel_now.contour.size()
//		for (int i=0; i<contotal; i++)   //边界点是会记到两块超像素点吗？
//		{
//			Point p_con = pixel_now.contour[i];
//			int conrank = sp_label.at<int>(p_con);
//			if (conran == rank_neigh) return true;
//		}		
//	}
//	else return false;	
//}
//
//bool Imadata::posneigh(Point cen_now, Point cen_neigh)
//{
//	int dist =  abs(cen_now.x-cen_neigh.x) + abs(cen_now.y-cen_neigh.y);
//	if (dist>50) return false;
//	else return true;
//}
//
//double chiSquareDist(const cv::Mat & hist1,const cv::Mat & hist2)  
//{  
//    int rows = hist1.rows;  
//    double sum = 0.0;  
//    double d1,d2;  
//    for(int r = 0;r < rows ;r++)  
//    {  
//        d1 = hist1.at<double>(r,0);  
//        d2 = hist2.at<double>(r,0);  
//        if( d1 ==0 && d2 == 0)  
//            ;  
//        else  
//            sum +=*pow( d1 - d2,2)/(d1+d2);  
//    }  
//    return sum;  
//}
