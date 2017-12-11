#include <iostream>

#include <opencv2/opencv.hpp>
const int NEIGHBOURS = 9;
const int MAX_ITER = 500;
const double alpha = 5;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void part1();
void part2();

std::string PATH_Ball = "./images/ball.png";
std::string PATH_Coffee = "./images/coffee.png";

//////////////////////////////////////
// function declarations for task 1 //
//////////////////////////////////////
void drawSnake(cv::Mat img, const std::vector<cv::Point2i>& snake);
void snakes(const cv::Mat& img, const cv::Point2i center, const int radius,
            std::vector<cv::Point2i>& snake);
void snakes_gd(const cv::Mat& img, const cv::Point2i center, const int radius,
            std::vector<cv::Point2i>& snake);
void pentadiagonal_solve(float p,  float q,  float r,  float* arr,  int N);
void initialize_diag(cv::Mat& M, float val,  int d);

//////////////////////////////////////
// function declarations for task 2 //
//////////////////////////////////////
void showGray(const cv::Mat& img, const std::string title = "Image",
              const int t = 0);
void showContour(const cv::Mat& img, const cv::Mat& contour, const int t = 0);
void levelSetContours(const cv::Mat& img, const cv::Point2f center,
                      const float radius, cv::Mat& phi);
cv::Mat computeContour(const cv::Mat& phi, const float level);

struct Operator
{
        double cutoff;
        Operator(double cutoff){
                this->cutoff = cutoff;
        }
        
        void operator ()(float &val, const int * position) const
        {
                if(val < cutoff){
                        val = 1.0;
                } else{
                        val = 1.0/((val) + 1.0);
                }
        }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() {

        // Uncomment the part of the exercise that you wish to implement.
        // For the final submission all implemented parts should be uncommented.

        part1();
//         part2();

//         cv::Mat sample_in = cv::Mat::ones(5, 5, CV_32FC1);
//         sample_in = 2.0 * sample_in;
//         
//         cv::Mat sample_out = cv::Mat::ones(5, 5, CV_32FC1);
//         sample_out = 2.0 * sample_out;
//         
//         std::cout << sample_out << std::endl;
//         std::cout << sample_in << std::endl;
//         
//         sample_out = sample_out.mul(sample_in);
//         
//         std::cout << sample_out << std::endl;
        

        std::cout << std::endl;
        std::cout
                << "////////////////////////////////////////////////////////////////////////////////////////////"
                << std::endl;
        std::cout
                << "////////////////////////////////////////////////////////////////////////////////////////////"
                << std::endl;
        std::cout
                << "////    END    /////////////////////////////////////////////////////////////////////////////"
                << std::endl;
        std::cout
                << "////////////////////////////////////////////////////////////////////////////////////////////"
                << std::endl;
        std::cout
                << "////////////////////////////////////////////////////////////////////////////////////////////"
                << std::endl;
        std::cout << std::endl;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void part1() {
    std::cout << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////    Part 1    //////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout << std::endl;

    cv::Mat ball;
    cv::imread(PATH_Ball, cv::IMREAD_COLOR).convertTo(ball, CV_32FC3,
                                                      (1. / 255.));
    cv::Mat coffee;
    cv::imread(PATH_Coffee, cv::IMREAD_COLOR).convertTo(coffee, CV_32FC3,
                                                        (1. / 255.));

    std::vector<cv::Point2i> snake;
    size_t radius;
    cv::Point2i center;

//     std::cout << "ball image" << std::endl;
//     // for snake initialization
//     center = cv::Point2i(ball.cols / 2, ball.rows / 2);
//     radius = std::min(ball.cols / 3, ball.rows / 3);
//     //////////////////////////////////////
//     snakes_gd(ball, center, radius, snake);
//     //////////////////////////////////////

    std::cout << "coffee image" << std::endl;
    // for snake initialization
    center = cv::Point2i(coffee.cols / 2, coffee.rows / 2);
    radius = std::min(coffee.cols / 2, coffee.rows / 2);
    ////////////////////////////////////////
    snakes_gd(coffee, center, radius, snake);
    ////////////////////////////////////////

    cv::destroyAllWindows();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void part2() {
    std::cout << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////    Part 2    //////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout
            << "////////////////////////////////////////////////////////////////////////////////////////////"
            << std::endl;
    std::cout << std::endl;

    cv::Mat ball;
    cv::imread(PATH_Ball, cv::IMREAD_COLOR).convertTo(ball, CV_32FC3,
                                                      (1. / 255.));
    cv::Mat coffee;
    cv::imread(PATH_Coffee, cv::IMREAD_COLOR).convertTo(coffee, CV_32FC3,
                                                        (1. / 255.));

    cv::Mat phi;
    size_t radius;
    cv::Point2i center;

    std::cout << "ball image" << std::endl;
    center = cv::Point2i(ball.cols / 2, ball.rows / 2);
    radius = std::min(ball.cols / 3, ball.rows / 3);
    /////////////////////////////////////////////////////////
    levelSetContours(ball, center, radius, phi);
    /////////////////////////////////////////////////////////

    std::cout << "coffee image" << std::endl;
    center = cv::Point2f(coffee.cols / 2.f, coffee.rows / 2.f);
    radius = std::min(coffee.cols / 3.f, coffee.rows / 3.f);
    /////////////////////////////////////////////////////////
    levelSetContours(coffee, center, radius, phi);
    /////////////////////////////////////////////////////////

    cv::destroyAllWindows();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Point2i getNewVertex(const cv::Point2i oldVertex, int state) {
    //	std::cout << "vertex: " << oldVertex << " state: " << state << std::endl;
    cv::Point2i p1;
    p1.x = (oldVertex.x - 1) + (state / 3);
    p1.y = (oldVertex.y - 1) + (state % 3);
    //	std::cout << "getNewVertex p1: " << p1 << std::endl;
    //	std::cout << " newvertex: " << p1 << std::endl;
    return p1;
}

float getUnaryCost(const cv::Mat &gradientX, cv::Mat &gradientY,
                    const cv::Point2i vertex, int k) {
    cv::Point2i tmp = getNewVertex(vertex, k);
    float val = gradientX.at<float>(tmp);
    //	std::cout << "getUnaryCost val: "<< val << std::endl;
    return -1.0
            * (std::pow(gradientX.at<float>(tmp), 2)
               + std::pow(gradientY.at<float>(tmp), 2));
}

float getPairWiseCost(const cv::Point2i currentNode,
                       const cv::Point2i prevNode, int currentState, int prevState, float d) {
    cv::Point2i p1, p2;
    p1 = getNewVertex(currentNode, currentState);
    p2 = getNewVertex(prevNode, prevState);

    cv::Point2i tmp = p2 - p1;
    return alpha * std::pow(d - std::sqrt(pow(tmp.x,2) + pow(tmp.y,2)),2);
}

///////////////////////////////////////////
// apply the snake algorithm to an image //
///////////////////////////////////////////
void snakes(const cv::Mat& img, const cv::Point2i center, const int radius,
            std::vector<cv::Point2i>& snake) {
    // initialize snake with a circle
    const int vvvTOTAL = radius * CV_PI / 7; // defines number of snake vertices // adaptive based on the circumference
    snake.resize(vvvTOTAL);
    float angle = 0;
    for (cv::Point2i& vvv : snake) {
        vvv.x = round(center.x + cos(angle) * radius);
        vvv.y = round(center.y + sin(angle) * radius);

        angle += 2 * CV_PI / vvvTOTAL;
    }

    std::cout << img.size() << std::endl;

    cv::Mat gray, gradientX, gradientY;
    cv::cvtColor(img, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 2);
    cv::Sobel(gray, gradientX, -1, 1, 0);
    cv::Sobel(gray, gradientY, -1, 0, 1);

    //show gradients
    cv::imshow("gradient X", gradientX);
    cv::waitKey(0);
    cv::imshow("gradient Y", gradientY);
    cv::waitKey(0);

    cv::Mat EnergyMat(NEIGHBOURS, snake.size(), CV_32F, cv::Scalar(0.0));
    cv::Mat BtMat(NEIGHBOURS, snake.size(), CV_8U, cv::Scalar(-1));

    std::cout<<"Energy Mat Size"<<EnergyMat.size()<<std::endl;

    for (int l = 0; l < MAX_ITER; l++) {

        std::cout<<" "<<l;
        int start = std::rand()%snake.size();
        std::cout<<start<<" ";

        float d=0;
        for(int j=1;j<snake.size();j++)
        {
            cv::Point2i avg;
            avg.x += round(std::sqrt(std::pow(snake[j-1].x,2) + std::pow(snake[j].x,2)));
            avg.y += round(std::sqrt(std::pow(snake[j-1].y,2) + std::pow(snake[j].y,2)));
            d = std::sqrt(std::pow(avg.x,2) + std::pow(avg.y,2));
        }
        d /= round(snake.size());

        for (int t = 0; t < NEIGHBOURS; t++) {    // Unary cost for first vertex
            EnergyMat.at<float>(t, start) = getUnaryCost(gradientX, gradientY,
                                                      snake[start], t);
        }

       int i= start;
       std::cout<<i<<" ";
       do {
           std::cout<<i<<" ";

           for (int j = 0; j < NEIGHBOURS; j++) { // states of current node

               EnergyMat.at<float>(j, i) = getUnaryCost(gradientX, gradientY,
                                                         snake[i], j);

               float minPwCost = std::numeric_limits<float>::max();
               int minLoc = NEIGHBOURS;

               for (int k = 0; k < NEIGHBOURS; k++) { // states of previous node
                   float pwCost = EnergyMat.at<float>(k, (i - 1)%snake.size())
                           + getPairWiseCost(snake[i], snake[(i - 1) % snake.size()], j, k, d);
                   if (pwCost < minPwCost) {
                       minPwCost = pwCost;
                       minLoc = k;
                   }
               }

               EnergyMat.at<float>(j, i) += minPwCost;
               BtMat.at<uchar>(j, i) = minLoc;
           }

           i=(i+1)%snake.size();
       }while(i!=start) ;       // Snake Vertices


       //Back Tracking

       // Find min of last vertex;
       float minVal = std::numeric_limits<float>::max();
       int minId = -1;
       bool stop = true;

       for (int j = 0; j < NEIGHBOURS; j++) {
           float val = EnergyMat.at<float>(j, (start - 1) % snake.size());
           if (val < minVal) {
               minVal = val;
               minId = j;
           }
       }
       if (minId != 4) {
           stop = false;
       }
//        std::cout << " minVal: " << minVal;
//        std::cout << " minId: " << minId;
       snake[(start - 1) % snake.size()] = getNewVertex(snake[(start - 1) % snake.size()], minId);

       // get rest of the path
       int k = (start - 2) % snake.size();
       do {
           int nextMin = BtMat.at<uchar>(minId, (k + 1)%snake.size());
           snake[k] = getNewVertex(snake[k], nextMin);
           minId = nextMin;
           if (minId != 4) {
               stop = false;
           }
           k = (k-1)%snake.size();
       }while(k != (start - 2) % snake.size());

       // visualization
       cv::Mat vis;
       img.copyTo(vis);
       drawSnake(vis, snake);
       cv::imshow("Snake", vis);
       cv::waitKey(50);

       if (stop)
           break;

    }        //MAX ITERATION

    // visualization
    cv::Mat vis;
    img.copyTo(vis);
    drawSnake(vis, snake);
    ///////////////////////////////////////////////////////////
    std::cout << "Press any key to continue...\n" << std::endl;
    ///////////////////////////////////////////////////////////
    cv::imshow("Snake", vis);
    cv::waitKey();

    // Perform optimization of the initialized snake as described in the exercise sheet and the slides.
    // You might want to apply some GaussianBlur on the edges so that the snake sidles up better.
    // Iterate until
    // - optimal solution for every point is the center of a 3x3 (or similar) box, OR
    // - until maximum number of iterations is reached

    // At each step visualize the current result
    // using **drawSnake() and cv::waitKey(10)** as in the example above and when necessary **std::cout**
    // In the end, after the last visualization, use **cv::destroyAllWindows()**

    //	cv::destroyAllWindows();
}

void snakes_gd(const cv::Mat& img, const cv::Point2i center, const int radius,
            std::vector<cv::Point2i>& snake) {

        // initialize snake with a circle
        const int vvvTOTAL = radius * CV_PI / 7;            // defines number of snake vertices ;adaptive based on the circumference
        snake.resize(vvvTOTAL);
        float angle = 0;
        for (cv::Point2i& vvv : snake) {
                vvv.x = round(center.x + cos(angle) * radius);
                vvv.y = round(center.y + sin(angle) * radius);
                angle += 2 * CV_PI / vvvTOTAL;
        }

        cv::Mat gray, gradientX, gradientY;
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 2);
        cv::Sobel(gray, gradientX, -1, 1, 0);
        cv::Sobel(gray, gradientY, -1, 0, 1);

        //show gradients
        cv::imshow("gradient X", gradientX);
        cv::waitKey(0);
        cv::imshow("gradient Y", gradientY);
        cv::waitKey(0);     
        
        const float alpha = 0.2, beta = 0.2;                // tension, stiffness
        const float ds = 1.0, ds2 = ds*ds, dt = 0.3;      // space, time
        const int N = snake.size();
        float x[N],  y[N];
        
        for (int i = 0; i < N; i++) {
                x[i] =  snake[i].x;
                y[i] =  snake[i].y;
        }

        for(int i = 0; i < MAX_ITER; i++){
                for (int i = 0; i < N; i++) {
                        x[i] +=  dt * gradientX.at<float>(y[i], x[i]);
                        y[i] +=  dt * gradientY.at<float>(y[i], x[i]);
                }
                float a = alpha*dt/ds2, b = beta*dt/ds2;
                float p = b, q = -a-(4*b), r = 1+(2*a)+(6*b);
                
                pentadiagonal_solve(p, q, r, x, N);
                pentadiagonal_solve(p, q, r, y, N);
                
                for (int i = 0; i < N; i++) {
                        snake[i].x = (int)x[i];
                        snake[i].y = (int)y[i];
                }
                
                // visualization
                cv::Mat vis;
                img.copyTo(vis);
                drawSnake(vis, snake);
                cv::imshow("Snake", vis);
                cv::waitKey(10);
        } 
        

                    
}

void pentadiagonal_solve(float p,  float q,  float r,  float* arr,  int N){
        cv::Mat u(N, 1, CV_32FC1);
        for(int i = 0; i < N; i++){
                u.at<float>(i, 0) = arr[i];
        }
        cv::Mat M = cv::Mat::zeros(N, N, CV_32FC1);
        initialize_diag(M, r, 0);
        initialize_diag(M, q, 1);
        initialize_diag(M, p, 2);
        initialize_diag(M, q, N-1);
        initialize_diag(M, p, N-2);
        initialize_diag(M, q, -1);
        initialize_diag(M, p, -2);
        initialize_diag(M, q, -N+1);
        initialize_diag(M, p, -N+2);
        
        cv::Mat u_ = M.inv() * u;
        
        for(int i = 0; i < N; i++){
                arr[i] = u_.at<float>(i, 0);
        }
}

void initialize_diag(cv::Mat& m, float val, int d){
        cv::Mat M = m.diag(d);
        int rows = M.size().height;
        for(int i = 0; i < rows; i++){
                M.at<float>(i, 0) = val;
        }
}

////////////////////////////////
// draws a snake on the image //
////////////////////////////////
void drawSnake(cv::Mat img, const std::vector<cv::Point2i>& snake) {
    const size_t siz = snake.size();

    for (size_t iii = 0; iii < siz; iii++)
        cv::line(img, snake[iii], snake[(iii + 1) % siz], cv::Scalar(0, 0, 1));

    for (const cv::Point2i& p : snake)
        cv::circle(img, p, 2, cv::Scalar(1, 0, 0), -1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////
// runs the level-set geodesic active contours algorithm //
///////////////////////////////////////////////////////////
void levelSetContours( const cv::Mat& img, const cv::Point2f center, const float radius, cv::Mat& phi )
{
        const int LS_ITERATIONS = 15000;
        const float EPSILON = 10e-4;
        phi.create( img.size(), CV_32FC1 );
        
        //////////////////////////////
        // signed distance map **phi**
        //////////////////////////////
        // initialize as a cone around the
        // center with phi(x,y)=0 at the radius
        
        for(int y=0; y<phi.rows; y++) {
                const float disty2 = pow( y-center.y, 2 );
                for (int x=0; x<phi.cols; x++) {      
                        phi.at<float>(y,x) = disty2 + pow( x-center.x, 2 );   
                }
        }

        cv::sqrt(phi, phi);

        // positive values inside
        phi = (radius - phi);
        cv::Mat temp = computeContour( phi, 0.0f);

        ///////////////////////////////////////////////////////////
        std::cout << "Press any key to continue...\n" << std::endl;
        ///////////////////////////////////////////////////////////
        showGray(phi, "phi", 0);
        showContour(img, temp,  0);
        /////////////////////////////
        
        cv::Mat gray, gradientX, gradientY; 
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 2);    // smooth image
        cv::Sobel(gray, gradientX, -1, 1, 0);               // find x-gradient
        cv::Sobel(gray, gradientY, -1, 0, 1);               // find y-gradient

        // Perform optimization of the initialized level-set function with geodesic active contours as described in the exercise sheet and the slides.
        // Iterate until
        // - the contour does not change between 2 consequitive iterations, or
        // - until a maximum number of iterations is reached

        // At each step visualize the current result
        // using **showGray() and showContour()** as in the example above and when necessary **std::cout**
        // In the end, after the last visualization, use **cv::destroyAllWindows()**
        
        int kernel_rows = 3,  kernel_cols = 3;
        cv::Mat k1_x = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k1_x.at<float>(1, 2) = 1.0;
        k1_x.at<float>(1, 0) = -1.0;
        k1_x = 0.5 * k1_x;
        
        cv::Mat k1_y = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k1_y.at<float>(2, 1) = 1.0;
        k1_y.at<float>(0, 1) = -1.0;
        k1_y = k1_y * 0.5;
        
        cv::Mat k1_xx = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k1_xx.at<float>(1, 2) = 1.0;
        k1_xx.at<float>(1, 0) = 1.0;
        k1_xx.at<float>(1, 1) = -2.0;
        
        cv::Mat k1_yy = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k1_yy.at<float>(2, 1) = 1.0;
        k1_yy.at<float>(0, 1) = 1.0;
        k1_yy.at<float>(1, 1) = -2.0;
        
        cv::Mat k1_xy = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k1_xy.at<float>(2, 2) = 1.0;
        k1_xy.at<float>(0, 2) = -1.0;
        k1_xy.at<float>(2, 0) = -1.0;
        k1_xy.at<float>(0, 0) = 1.0;
        k1_xy = 0.25 * k1_xy;
        
        cv::Mat k2_x1 = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k2_x1.at<float>(1, 2) = 1.0;
        k2_x1.at<float>(1, 1) = -1.0;
        
        cv::Mat k2_x2 = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k2_x2.at<float>(1, 1) = 1.0;
        k2_x2.at<float>(1, 0) = -1.0;
        
        cv::Mat k2_y1 = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k2_y1.at<float>(2, 1) = 1.0;
        k2_y1.at<float>(1, 1) = -1.0;
        
        cv::Mat k2_y2 = cv::Mat::zeros(kernel_rows,  kernel_cols, CV_32FC1);
        k2_y2.at<float>(1, 1) = 1.0;
        k2_y2.at<float>(0, 1) = -1.0;
        
        cv::Mat phi1_x(phi.size(),  phi.type());
        cv::Mat phi1_y(phi.size(),  phi.type());
        cv::Mat phi1_xx(phi.size(),  phi.type());
        cv::Mat phi1_yy(phi.size(),  phi.type());
        cv::Mat phi1_xy(phi.size(),  phi.type());
        
        cv::Mat phi2_x1(phi.size(),  phi.type());
        cv::Mat phi2_x2(phi.size(),  phi.type());
        cv::Mat phi2_y1(phi.size(),  phi.type());
        cv::Mat phi2_y2(phi.size(),  phi.type());
        
        // Element-wise multiplication
        cv::Mat w = gradientX.mul(gradientX) + gradientY.mul(gradientY); 
        
        // Find square root
        cv::sqrt(w, w);

        double _tmp = 0.0, max_w = 0.0;
        cv::minMaxLoc(w, &_tmp, &max_w);
        
        // Apply proposed metric
        w.forEach<float>(Operator(0.05 * max_w));
        

        cv::Mat w_x(w.size(),  w.type());
        cv::Mat w_y(w.size(),  w.type());
        
        cv::filter2D(w, w_x, -1, k1_x);
        cv::filter2D(w, w_y, -1, k1_y);
        
        
        double tau = 1.0 / (max_w);
        
        cv::Mat w_x_max = cv::max(w_x, 0.0);
        cv::Mat w_y_max = cv::max(w_y, 0.0);
        cv::Mat w_x_min = cv::min(w_x, 0.0);
        cv::Mat w_y_min = cv::min(w_y, 0.0);
        
        #pragma omp parallel for
        for(int i = 0; i < LS_ITERATIONS; i++){
                cv::filter2D(phi, phi1_x, -1 , k1_x);
                cv::filter2D(phi, phi1_y, -1 , k1_y);
                cv::filter2D(phi, phi1_xx, -1 , k1_xx);
                cv::filter2D(phi, phi1_yy, -1 , k1_yy);
                cv::filter2D(phi, phi1_xy, -1 , k1_xy);

                cv::Mat update_curvature = tau * w.mul((phi1_xx.mul(phi1_y.mul(phi1_y)) - 2.0 * phi1_x.mul(phi1_y.mul(phi1_xy)) + phi1_yy.mul(phi1_x.mul(phi1_x))) / (phi1_x.mul(phi1_x) + phi1_y.mul(phi1_y) + EPSILON));
                
                cv::filter2D(phi, phi2_x1, -1 , k2_x1);
                cv::filter2D(phi, phi2_x2, -1 , k2_x2);
                cv::filter2D(phi, phi2_y1, -1 , k2_y1);
                cv::filter2D(phi, phi2_y2, -1 , k2_y2);
                
                cv::Mat update_front = tau * (w_x_max.mul(phi2_x1) + w_x_min.mul(phi2_x2) + w_y_max.mul(phi2_y1) + w_y_min.mul(phi2_y2));
                
                phi +=  (update_curvature + update_front);

                if(i % 100 == 0){
                        std::cout << "ITERATION: " << i << std::endl;
                        temp = computeContour(phi, 0.0f);
                        showGray(phi, "phi", 10);
                        showContour(img, temp, 10);
                }
                
        }
        
        cv::destroyAllWindows();
}

////////////////////////////
// show a grayscale image //
////////////////////////////
void showGray(const cv::Mat& img, const std::string title, const int t) {
    CV_Assert(img.channels() == 1);

    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);

    cv::Mat temp;
    img.convertTo(temp, CV_32F, 1. / (maxVal - minVal),
                  -minVal / (maxVal - minVal));
    cv::imshow(title, temp);
    cv::waitKey(t);
}

//////////////////////////////////////////////
// compute the pixels where phi(x,y)==level //
//////////////////////////////////////////////
cv::Mat computeContour(const cv::Mat& phi, const float level) {
    CV_Assert(phi.type() == CV_32FC1);

    cv::Mat segmented_NORMAL(phi.size(), phi.type());
    cv::Mat segmented_ERODED(phi.size(), phi.type());

    cv::threshold(phi, segmented_NORMAL, level, 1.0, cv::THRESH_BINARY);
    cv::erode(segmented_NORMAL, segmented_ERODED,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(3, 3)));

    return (segmented_NORMAL != segmented_ERODED);
}

///////////////////////////
// draw contour on image //
///////////////////////////
void showContour(const cv::Mat& img, const cv::Mat& contour, const int t) {
    CV_Assert(
                img.cols == contour.cols && img.rows == contour.rows && img.type() == CV_32FC3 && contour.type() == CV_8UC1);

    cv::Mat temp(img.size(), img.type());

    const cv::Vec3f color(0, 0, 1); // BGR

    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            temp.at<cv::Vec3f>(y, x) =
                    contour.at<uchar>(y, x) != 255 ?
                        img.at<cv::Vec3f>(y, x) : color;

    cv::imshow("contour", temp);
    cv::waitKey(t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
