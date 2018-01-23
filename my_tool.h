#pragma once
#ifndef MY_TOOL_H
#define MY_TOOL_H

#include <math.h>

double f_ab(double t)
{
	if ((t - 0.008856) > 1e-6)
		return pow(t, 1.0 / 3.0);
	else
		return 7.787*t + 16.0 / 116.0;
}
double f_L(double y)
{
	if ((y - 0.008856) > 1e-6)
		return 116 * pow(y, 1.0 / 3.0);
	else
		return 903.3*y;
}

bool RGB2Lab(unsigned char *rgb, double *lab/*, int iWidth, int iHeight*/)
{
	// 输入参数有效性判断
	if (rgb == nullptr || lab == nullptr)return false;

	// 每行图像数据的字节数
	/*int iLBytes = (iWidth * 24 + 31) / 32 * 4;*/

//	unsigned char *rgb;
//	double *lab;
	unsigned char R, G, B;
	double l, a, b;
	double x, y, z;
	//double fx, fy, fz;
	double BLACK = 20.0;
	double YELLOW = 70.0;

	//for (int i = 0; i < iHeight; i++)
	//{
	//	for (int j = 0; j < iWidth; j++)
	//	{
	//		rgb = src + iLBytes*i + 3 * j;
	//		lab = dst + iLBytes*i + 3 * j;

			// R\G\B像素值
	B = *rgb; G = *(rgb + 1); R = *(rgb + 2);
			// 转至X-Y-Z
			//[ X ]   [ 0.412453  0.357580  0.180423 ]   [ R/255 ]
			//[ Y ] = [ 0.212671  0.715160  0.072169 ] * [ G/255 ]
			//[ Z ]   [ 0.019334  0.119193  0.950227 ]   [ B/255 ]
	x = (0.433910*double(R) + 0.376220*double(G) + 0.1*double(B)) / 255.0;
	y = (0.212671*double(R) + 0.715160*double(G) + 0.072169*double(B)) / 255.0;
	z = (0.019334*double(R) + 0.119193*double(G) + 0.950227*double(B)) / 255.0;
			//// 除255即归一化
			//x = x / (255.0*0.950456);
			//y = y / 255.0;
			//z = z / (255.0*1.088754);

	l = f_L(y);

	a = 500 * (f_ab(x) - f_ab(y));
	b = 200 * (f_ab(y) - f_ab(z));

			//// 这里不加时出现颜色饱和的情况(见上图)
			//// 参考出处http://c.chinaitlab.com/cc/ccjq/200806/752572.html
			//if (l< BLACK)
			//{
			//	a *= exp((l - BLACK) / (BLACK / 4));
			//	b *= exp((l - BLACK) / (BLACK / 4));
			//	l = 20;
			//}
			//if (b > YELLOW)b = YELLOW;

			//// 归一化值Lab
			//*lab = l / 255.0;    // L
			//*(lab + 1) = (a + 128.0) / 255.0; // a
			//*(lab + 2) = (b + 128.0) / 255.0; // b

			//不归一化
			*lab = l ;
			*(lab + 1) = a;
			*(lab + 2) = b;

		/*}
	}*/
	return true;
}

//const float param_13 = 1.0f / 3.0f;
//const float param_16116 = 16.0f / 116.0f;
//const float Xn = 0.950456f;
//const float Yn = 1.0f;
//const float Zn = 1.088754f;
//
//void Lab2RGB(float L, float a, float b, float *R, float *G, float *B)
//{
//	float X = 0.0f, Y = 0.0f, Z = 0.0f;
//	Lab2XYZ(L, a, b, &X, &Y, &Z);
//	XYZ2RGB(X, Y, Z, R, G, B);
//}
//
//void XYZ2RGB(float X, float Y, float Z, float *R, float *G, float *B)
//{
//	float RR, GG, BB;
//	RR = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
//	GG = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
//	BB = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
//
//	*R = (float)CLAMP0255_XY(RR, 1.0f);
//	*G = (float)CLAMP0255_XY(GG, 1.0f);
//	*B = (float)CLAMP0255_XY(BB, 1.0f);
//}
//
//void Lab2XYZ(float L, float a, float b, float *X, float *Y, float *Z)
//{
//	float fX, fY, fZ;
//
//	fY = (L + 16.0f) / 116.0f;
//	if (fY > 0.206893f)
//		*Y = fY * fY * fY;
//	else
//		*Y = (fY - param_16116) / 7.787f;
//
//	fX = a / 500.0f + fY;
//	if (fX > 0.206893f)
//		*X = fX * fX * fX;
//	else
//		*X = (fX - param_16116) / 7.787f;
//
//	fZ = fY - b / 200.0f;
//	if (fZ > 0.206893f)
//		*Z = fZ * fZ * fZ;
//	else
//		*Z = (fZ - param_16116) / 7.787f;
//
//	(*X) *= Xn;
//	(*Y) *= Yn;
//	(*Z) *= Zn;
//}

float interpolate(SuperPixel &t, SuperPixel &a, SuperPixel &b, SuperPixel &c)
{
	float c1 = t.chiSquareDist(a), c2 = t.chiSquareDist(b), c3 = t.chiSquareDist(c);
	float sum = a.depth_average / c1 / a.priority + b.depth_average / c2 / b.priority + c.depth_average / c3 / c.priority;
	sum = sum / (1.0 / c1 / a.priority + 1.0 / c2 / b.priority + 1.0 / c3 / c.priority);
	return sum;
}

#endif // !MY_TOOL_H
