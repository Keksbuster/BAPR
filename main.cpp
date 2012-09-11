#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <glob.h>
#include <string.h>
#include <queue>
#include <algorithm>


using namespace std;
using namespace cv;

typedef std::vector<cv::Point> Contour;
typedef std::vector< Contour > ContourVec;

 IplImage* img_glob;
 
 CvPoint center_pt;

class Index {
	cv::PCA pca;
	cv::Mat idx_projected;

	public:
	void generatePCA(vector <vector <CvPoint> >& vec) {
		cv::Mat m(vec.size(), vec[0].size()*2, CV_32FC1);
		for(size_t contour = 0; contour < vec.size(); contour++) {
			for(size_t point = 0; point < vec[contour].size(); point++) {
				m.at<float>(contour, 2*point) = vec[contour][point].x;
				m.at<float>(contour, 2*point+1) = vec[contour][point].y;
			}
		}
		
		pca(m, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
	}

	int getBestFittingIdx(vector<cv::Point>& c) {
		cv::Mat m(1,c.size()*2, CV_32FC1);
		cv::Mat projected_c;
		for(size_t point = 0; point < c.size(); point++) {
			m.at<float>(0, 2*point) = c[point].x;
			m.at<float>(0, 2*point+1) = c[point].y;
		}
		pca.project(m, projected_c);
		

		double mindistance = std::numeric_limits<double>::max();
		int minidx = -1;
		for(int row = 0; row < m.rows; row++) {
			double distance = 0;
			for(int i = 0; i < m.cols; i++) {
				float tmp = (idx_projected.at<float>(row,i) - projected_c.at<float>(0,i));
				distance += tmp*tmp;
			}
			if(distance < mindistance) {
				mindistance = distance;
				minidx = row;
			}
	
		}
		return minidx;
	}

cv::PCA doPCA(vector <vector <CvPoint> >& vec) {
	cv::Mat m(vec.size(), vec[0].size()*2, CV_32FC1);
	for(size_t contour = 0; contour < vec.size(); contour++) {
		for(size_t point = 0; point < vec[contour].size(); point++) {
			m.at<float>(contour, 2*point) = vec[contour][point].x;
			m.at<float>(contour, 2*point+1) = vec[contour][point].y;
		}
	}
	Mat m2;
	pca.project(m, idx_projected);
	pca.backProject(idx_projected, m2);

	cv::Mat mean = pca.mean;
	cv::Mat evsqrt;
	cv::pow(abs(pca.eigenvalues.t()), 1.0/2, evsqrt);
	cv::Mat minhand = mean + 3 * evsqrt * pca.eigenvectors;
	cv::Mat maxhand = mean - 3 * evsqrt * pca.eigenvectors;
	maxhand = maxhand;// + img_glob->height/4;
	minhand = minhand ;//+ img_glob->height/4;
	mean = mean ;//+ img_glob->height/4;
	
	cv::Mat debugImg(img_glob->height, img_glob->width,CV_8UC3, cv::Scalar(0));
	
	for(int i = 0; i < maxhand.cols; i+= 2) {
		float a,b,c,d,e,f;
		float a_,b_,c_,d_,e_,f_;
		a = minhand.at<float>(0,i);
		b = minhand.at<float>(0,i+1);
		c = maxhand.at<float>(0,i);
		d = maxhand.at<float>(0,i+1);
		e = mean.at<float>(0,i);
		f = mean.at<float>(0,i+1);
		
		if(i < maxhand.cols-1){
		a_ = minhand.at<float>(0,i+2);
		b_ = minhand.at<float>(0,i+3);
		c_ = maxhand.at<float>(0,i+2);
		d_ = maxhand.at<float>(0,i+3);
		e_ = mean.at<float>(0,i+2);
		f_ = mean.at<float>(0,i+3);
	}
		
		cv::Point pca_max(c,d), pca_min(a,b), pca_mit(e,f);
		cv::Point pca_max_(c_,d_), pca_min_(a_,b_), pca_mit_(e_,f_);

		cv::circle(debugImg, pca_max, 5, CV_RGB(255,0,0),2);
		cv::circle(debugImg, pca_mit, 5, CV_RGB(0,0,255),2);
		cv::circle(debugImg, pca_min, 5, CV_RGB(0,255,0),2);
		if(i < maxhand.cols-2){
			cv::line(debugImg, pca_max, pca_max_ ,CV_RGB(255,0,0));
			cv::line(debugImg, pca_mit, pca_mit_ ,CV_RGB(0,0,255));
			cv::line(debugImg, pca_min, pca_min_ ,CV_RGB(00,255,0));
		}
	}

	cv::imshow("PCA", debugImg);
	return pca;
}
};

 char wndname[] = "Drawing Demo";


float Pi = ((float)4.0*atan(1.0));

float pt_dist(CvPoint p1, CvPoint p2)
{
	return sqrt(pow((p2.x-p1.x),2) + pow((p2.y-p1.y),2));
}

float pt_angle(Point p1, Point p2, Point p3)
{
	Point2f a = p1 - p2;
	Point2f b = p3 - p2;
	a *= 1/norm(a);
	b *= 1/norm(b);
	return (a.x*b.x+a.y*b.y);
}

int get_hand_points(char* img_name,vector<CvPoint> *all_fin, int col=0){
	
	IplImage* img = cvLoadImage(img_name);
	IplImage* img2=0;
	IplImage* img23=0;
	CvMemStorage* 	g_storage = cvCreateMemStorage(0);
	CvMemStorage* storage12 = cvCreateMemStorage(0);
	CvSeq* contours = 0;
	CvConvexityDefect* defectArray;
	int height,width, widthstep,widthstep2,count,i,anzpoint=0;
	float max1=-1.00;int max2=-1;
	CvSeq* hull2 = cvCreateSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage12 );
	CvPoint pt0,pt,pt1;
	CvSeq* hull;
	CvSeq* def;	
	vector <CvPoint> dp_h;
	vector <CvPoint> dp_l;
	vector <CvPoint> all_p;
	width = img->width;
	height= img->height;
	priority_queue< pair< float, int > > pq;
	priority_queue< pair< float, int > > pq2;

	
	img2 = cvCreateImage( cvSize(width,height), img->depth, 3 );
	img23 = cvCreateImage( cvSize(width,height), img2->depth, 1 );
	
	cvZero( img2 );
	cvZero( img23 );

	cvCvtColor(img, img2,  CV_BGR2YCrCb);
	cvInRangeS(img2, cvScalar(0,133,77), cvScalar(255,173,127), img23);//finden und glätten der Contur
	cvSmooth(img23, img23, CV_GAUSSIAN, 9, 9, 9);
	cvThreshold( img23, img23, 127, 255, CV_THRESH_BINARY);
	cvFindContours( img23, g_storage, &contours,sizeof(CvContour),CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	
	while(contours->total < 500)
		contours = (contours->h_next);

	hull = cvConvexHull2( contours, 0, CV_CLOCKWISE, 0 );
	def = cvConvexityDefects( contours, hull,NULL);


	int temp1=0;
	char tmp_str [50];
	count = def->total;
	defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*count);
	cvCvtSeqToArray(def,defectArray, CV_WHOLE_SEQ);

	int anz_points = 0;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.33, 0.33, 0, 1, CV_AA);
	for(i=0;i<count;i++){ //Packe defects und hochpunkte in def_h
		if(defectArray[i].depth>5){ 
			all_p.push_back(*defectArray[i].start);
			all_p.push_back(*defectArray[i].depth_point);
			anz_points+=2;
		}	
	}
	free(defectArray);
	
	//geringste abstände der hochpunkte untereinander finden
	
	Mat M_dist =    Mat(6, 6, CV_32FC1);
	
	for(std::vector<CvPoint>::size_type i = 0; i < all_p.size(); i+=2) {
		for(std::vector<CvPoint>::size_type j = 0; j < all_p.size(); j+=2) {	
			if(i!=j)
				M_dist.at<float>(i/2,j/2) = pt_dist(all_p[i],all_p[j]);
				
		}
	}
	
		for(std::vector<CvPoint>::size_type i = 0; i < all_p.size(); i+=2) {
		for(std::vector<CvPoint>::size_type j = 0; j < all_p.size(); j+=2) {	
			if(i!=j)
				M_dist.at<float>(i/2,j/2) = pt_dist(all_p[i],all_p[j]);
				
		}
	}
	float sum_dist = 0;
	for(int i = 0; i< 6;i++){
		float dist_left =M_dist.at<float>(i,abs(((i-1)%6)));
		float dist_right= M_dist.at<float>(i,abs(((i+1)%6)));
		
		pq.push( make_pair( -(dist_left+dist_right),i*2 ));
	}
	
	 
	int start_offset = -1;
  while (!pq.empty())
  {
	 if(start_offset<0)start_offset=pq.top().second;
    printf("%f\t%d\n",pq.top().first,pq.top().second);   //print out the highest priority element
   pq.pop();                   //remove the highest priority element
  }

	int idx=1;float temp_angl;

	CvPoint* hulli;
	CvPoint* cont;

	cvFindContours( img23, g_storage, &contours, sizeof(CvContour),CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	
	while(contours->total < 50)
		contours = (contours->h_next);

	cont = (CvPoint*)malloc(sizeof(CvPoint)*contours->total);

	cvCvtSeqToArray(contours,cont, CV_WHOLE_SEQ);

	int prev_pt = 0;
	int last_pt = 0;
	int safe_cur = 0,temp123=0,back=0;

	count = contours->total;

	if(col){
		for(std::vector<CvPoint>::size_type j = 0; j <= all_p.size(); j++) {
			sprintf (tmp_str, "%d", (int)j);
			cvPutText(img_glob, tmp_str, all_p[j], &font, cvScalar(0,255,255));
			if(col==1)cvCircle( img_glob, all_p[j], 10, CV_RGB(255,127,255)); 
			if(col==2)cvCircle( img_glob, all_p[j], 10, CV_RGB(255,0,255)); 
			if(col==3)cvCircle( img_glob, all_p[j], 10, CV_RGB(255,127,0)); 
			if(j<all_p.size()-1){
				if(col==1)cvLine(img_glob,all_p[j],all_p[j+1],CV_RGB(255,0,0));
				if(col==2)cvLine(img_glob,all_p[j],all_p[j+1],CV_RGB(0,255,0));
				if(col==3)cvLine(img_glob,all_p[j],all_p[j+1],CV_RGB(0,0,255));
			}
		}
	}
	
	int u_x=0;
	int u_y=0;




	
	for(std::vector<CvPoint>::size_type j = 0; j < all_p.size(); j++) {//Mittelpunkt
		u_x += all_p[(start_offset+j)%all_p.size()].x;
		u_y += all_p[(start_offset+j)%all_p.size()].y;
	}
	u_x /= all_p.size();
	u_y /= all_p.size();
	
	for(std::vector<CvPoint>::size_type j = 0; j < all_p.size(); j++) {//Abstand Punk <--> Mittelpunkt
		pq2.push( make_pair(pt_dist(cvPoint(u_x,u_y),all_p[j]), j ) );
	}
	
	int cur_pt=0;
	for(std::vector<CvPoint>::size_type j = 0; j < all_p.size(); j++) {
		cur_pt=(start_offset+j)%all_p.size();
		if(cur_pt==pq2.top().second)continue; //punkt mit größtem abstand überspringen
		all_fin->push_back(all_p[cur_pt]);	
	}

	
	if(col==1)cvCircle( img_glob, cvPoint(u_x,u_y), 5, CV_RGB(255,0,0),3); 
	if(col==2)cvCircle( img_glob, cvPoint(u_x,u_y), 10, CV_RGB(0,255,0),3); 
	if(col==3)cvCircle( img_glob, cvPoint(u_x,u_y), 15, CV_RGB(0,0,255),3); 
	
	return  all_fin->size();
}


void transform_points(vector<CvPoint> &all_fin,vector<CvPoint> &all_fin2){
	float delta;
	float sx   = 0;
	float sy   = 0;
	float sx_  = 0;
	float sxx  = 0;
	float sxy  = 0;
	float sxx_ = 0;
	float sxy_ = 0;
	float sy_  = 0;
	float syy  = 0;
	float syy_ = 0;
	float syx_ = 0;
	
	int anz_points,anz_points2;
	
//Mittelpunkt der hand finden

	float m_x=0.0;
	float m_xx=0.0;
	float m_xy=0.0;
	float m_xy_=0.0;
	float m_x_=0.0;	
	float m_xx_=0.0;
	float m_y=0.0;
	float m_yy=0.0;
	float m_yy_=0.0;
	float m_y_=0.0;
	float m_yx_=0.0;

	float u_x=0.0;
	float u_y=0.0;
	int tmp_counter=0;
	
	anz_points =all_fin.size();
	anz_points2=all_fin2.size();
	
	//Erst Mittelpunkt bestimmen
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size() && j < all_fin2.size(); j++) {
		m_x+=all_fin[j].x;
		m_x_+=all_fin2[j].x;
		m_y+=all_fin[j].y;
		m_y_+=all_fin2[j].y;
	}
	sx = m_x/anz_points;
	sy = m_y/anz_points;
	sx_ = m_x_/anz_points2;
	sy_ = m_y_/anz_points2;
	
	u_x = sx_;
	u_y = sy_;
	
	//alle punkte zum ursprung verschieben
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size() && j < all_fin2.size(); j++) {
		all_fin[j].x = all_fin[j].x - sx;
		all_fin[j].y = all_fin[j].y - sy;
		all_fin2[j].x = all_fin2[j].x - sx_;
		all_fin2[j].y = all_fin2[j].y - sy_;
	}
	m_x=0.0;
	m_x_=0.0;
	m_y=0.0;
	m_y_=0.0;
	

	
	//nochmal den kram rechnen
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size() && j < all_fin2.size(); j++) {
			
			printf("punkt A:\t%d\t%d.\tpunkt B:\t%d\t%d\n",all_fin[j].x,all_fin[j].y,all_fin2[j].x,all_fin2[j].y);
			
			m_x   += all_fin[j] .x;
			m_x_  += all_fin2[j].x;
			m_xx  += (all_fin[j].x * all_fin[j] .x);
			m_xy  +=  all_fin[j].x * all_fin[j] .y;
			m_xx_ += (all_fin[j].x * all_fin2[j].x);
			m_xy_ += (all_fin[j].x * all_fin2[j].y);
			m_y   += all_fin[j] .y;
			m_y_  += all_fin2[j].y;
			m_yy  += (all_fin[j].y * all_fin[j] .y);
			m_yy_ += (all_fin[j].y * all_fin2[j].y);
			m_yx_ += (all_fin[j].y * all_fin2[j].x);
		
			 
			
			tmp_counter++;
	}
	printf("---------------Punkte:--%d/%d-----\n",anz_points,tmp_counter);
	sx = m_x/anz_points;
	sxx = m_xx/anz_points;
	sxy = m_xy/anz_points;
	sxy_ = m_xy_/anz_points;
	sx_ = m_x_/anz_points;
	sxx_ = m_xx_/anz_points;
	sy = m_y/anz_points;
	syy = m_yy/anz_points;
	sy_ = m_y_/anz_points;
	syx_ = m_yx_/anz_points;
	syy_ = m_yy_/anz_points;

	printf("sx:\t%.0f\nsxx:\t%.0f\nsxy:\t%.0f\nsxy_:\t%.0f\nsx_:\t%.0f\nsxx_:\t%.0f\nsy:\t%.0f\nsyy:\t%.0f\nsy_:\t%.0f\nsyx_:\t%.0f\n\n",sx,sxx,sxy,sxy_,sx_,sxx_,sy,syy,sy_,syx_);

	delta = (sxx*syy) - (sxy*sxy);
	delta = 1/delta;
	//Transformations-Matrix
	Mat M_r =    Mat(2, 2, CV_32FC1);
	Mat M_l =    Mat(2, 2, CV_32FC1);
	Mat M_abcd = Mat(2, 2, CV_32FC1);
	
	M_l.at<float>(0,0) = sxx_;
	M_l.at<float>(0,1) = syx_;
	M_l.at<float>(1,0) = sxy_;
	M_l.at<float>(1,1) = syy_;
	
	M_r.at<float>(0,0) = syy;
	M_r.at<float>(0,1) = -sxy;
	M_r.at<float>(1,0) = -sxy;
	M_r.at<float>(1,1) = sxx;
	
	
	M_abcd = delta*(M_l*M_r);
	
	Matx22f abcd = M_abcd;
	
	printf("...............................\n");
	cout << M_abcd <<endl;
	
	//transform
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size(); j++) {
		
		Vec2f v (all_fin[j].x,all_fin[j].y);
		
		Matx21f tmp = abcd * v ; //Transformation anwenden
		Point pt = Point(tmp(0),tmp(1));
		all_fin[j] = pt;
	}	
	center_pt = cvPoint(u_x,u_y);
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size() && j < all_fin2.size(); j++) {
		all_fin[j].x = all_fin[j].x + u_x;
		all_fin[j].y = all_fin[j].y + u_y;
		all_fin2[j].x = all_fin2[j].x + u_x;
		all_fin2[j].y = all_fin2[j].y + u_y;
	}
}

void draw_hand_points(vector<CvPoint> &all_fin,IplImage *img, int col=0){
	//Start und endpunkt finden
	float tmp_dist =  0.0;
	float max_dist = -1.0;
	int st_pt=0;
	int end_pt=-1;
	for(std::vector<CvPoint>::size_type j = 0; j < all_fin.size()-1; j++) {
		if(col==0)cvCircle(img, all_fin[j],3, CV_RGB(255,255,255),3);
		if(col==1)cvCircle(img, all_fin[j],3, CV_RGB(255,0,0),3);
		if(col==2)cvCircle(img, all_fin[j],3, CV_RGB(0,255,0),3);
		if(col==3)cvCircle(img, all_fin[j],3, CV_RGB(0,0,255),3);
		if(col==4)cvCircle(img, all_fin[j],3, CV_RGB(255,255,0),3);
		if(col==5)cvCircle(img, all_fin[j],3, CV_RGB(0,255,255),3);
		if(col==6)cvCircle(img, all_fin[j],3, CV_RGB(255,0,255),3);
		if(j<all_fin.size()-1 ){
			if(col==0)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(255,255,255));
			if(col==1)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(255,0,0));
			if(col==2)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(0,255,0));
			if(col==3)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(0,0,255));
			if(col==4)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(255,255,0));
			if(col==5)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(0,255,255));
			if(col==6)cvLine(img, all_fin[j], all_fin[j+1], CV_RGB(255,0,255));	
		}
	}
	if(col==0)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(255,255,255));
	if(col==1)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(255,0,0));
	if(col==2)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(0,255,0));
	if(col==3)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(0,0,255));
	if(col==4)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(255,255,0));
	if(col==5)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(0,255,255));
	if(col==6)cvLine(img, all_fin.front(), all_fin.back(), CV_RGB(255,0,255));	
}

 int main(int argc, char *argv[])
 {
	vector <CvPoint> all_fin;
	vector <CvPoint> all_fin2;
	vector <vector <CvPoint> > hands;
	
	IplImage *img= 0;
	IplImage* img2=0;
	IplImage* img20=0;
	IplImage* img21=0;
	IplImage* img22=0;
	IplImage* img23=0;
	IplImage* img_rot=0;
	glob_t fold;
	float a,b,c,d;
    int height,width, widthstep,widthstep2,anz_points,anz_points2;
	int width1 = 256;
	int height1 = 256;
    uchar *data;
	uchar *data2;
    int i,ix,j,count=100;
	float xm=127.0;
	float f0,f1,f2,f3,k,alpha,r,new_g;
	int gray_tab[256];
	float max1=-1.00;int max2=-1;
	
	
	if(!argv[1]){
		printf("usage: ./main <0-6>\n");
		return -1;
	}

    switch( glob("*bilder/*.png", 0, NULL, &fold ) )
    {
        case 0:
            break;
        case GLOB_NOSPACE:
            printf( "Out of memory\n" );
            break;
        case GLOB_ABORTED:
            printf( "Reading error\n" );
            break;
        case GLOB_NOMATCH:
            printf( "No files found\n" );
            break;
        default:
            break;
    }
   
    
    
    
    int n_pic = atoi(argv[1]);
    img= cvLoadImage(fold.gl_pathv[n_pic]); 
    img_glob= cvLoadImage(fold.gl_pathv[n_pic]); 
    height= img->height;
    width= img->width;
    
    all_fin.clear();
    all_fin2.clear();
    
	img21 = cvCreateImage( cvSize(width,height), img->depth, 3 );
	
	anz_points2 = get_hand_points(fold.gl_pathv[n_pic ],&all_fin2);
	draw_hand_points(all_fin2,img21,1);
	
	
	for(int ii=0; ii<fold.gl_pathc; ii++)
    {
		printf("%s\n",fold.gl_pathv[ii]);
		anz_points = get_hand_points(fold.gl_pathv[ii],&all_fin);
	
		if(anz_points!=anz_points2){
			printf("Incompatible amount of points!\n");
			return -1;
		}
		transform_points(all_fin,all_fin2);
		draw_hand_points(all_fin,img21,ii);
		hands.push_back(all_fin);
		all_fin.clear();
	}
	
	Index idx;
	idx.generatePCA(hands);
	idx.doPCA(hands);

	cvShowImage("All Hands",img21);
    cvShowImage("Reference", img_glob); 
    cvWaitKey(0);
	cvReleaseImage(&img21); 
	cvReleaseImage(&img_glob); 
    return 0;
 }
