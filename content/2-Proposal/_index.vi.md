---
title: "Bản đề xuất"
date: "2025-09-09"
weight: 2
chapter: false
pre: " <b> 2. </b> "
---


# MÔ HÌNH DỰ BÁO QUỸ ĐẠO BÃO
## Tăng cường dữ liệu theo từng bước phai dần theo thời gian cho dự báo chuỗi thời gian và Học máy dựa trên vật lý

### 1. Tóm tắt điều hành  

Dự báo chuỗi thời gian là nền tảng của nhiều ứng dụng khoa học và công nghiệp, từ khí tượng học đến mô hình tài chính. Mặc dù các kiến trúc mô hình đã có nhiều tiến bộ, chất lượng và độ đa dạng của dữ liệu huấn luyện vẫn là yếu tố quyết định hiệu suất. Các kỹ thuật tăng cường dữ liệu hiện có — như nhiễu ngẫu nhiên, cắt nhỏ chuỗi, hoặc thêm nhiễu trắng — thường làm sai lệch mối quan hệ thời gian và không phản ánh được sự suy giảm tự nhiên của ảnh hưởng từ các sự kiện trong quá khứ. Khoảng trống này cho thấy cần có một phương pháp có nguyên tắc để vừa duy trì tính liên kết tuần tự vừa nắm bắt được quá trình phai dần của mức độ liên quan theo thời gian.

Bản đề xuất này giới thiệu một khung tăng cường chuỗi thời gian mới có tên Stepwise Temporal Fading Augmentation (STFA). Khác với các phương pháp truyền thống dựa trên nhiễu hoặc biến đổi ngẫu nhiên, STFA mô phỏng sự suy giảm tự nhiên trong ảnh hưởng của các quan sát cũ bằng cách áp dụng trọng số phai dần cho các giá trị trước đó trong khi vẫn giữ nguyên các giá trị gần đây. Cách tiếp cận này tạo ra các chuỗi tổng hợp thực tế và đa dạng, giúp cải thiện độ bền vững của mô hình. Phương pháp này sẽ được đánh giá trong bài toán dự đoán quỹ đạo bão, vốn phụ thuộc vào dữ liệu chuỗi theo thứ tự vĩ độ–kinh độ. Ngoài ra, nguyên tắc Học máy dựa trên vật lý (Physics-Informed Machine Learning, PIML) được kết hợp bằng cách nhúng các mối quan hệ địa lý—như khoảng cách Haversine và góc phương vị (bearing)—vào cả tập đặc trưng và hàm mất mát. Thiết kế lai này kết hợp sự linh hoạt của học sâu với tính chặt chẽ của các ràng buộc vật lý, nhằm nâng cao độ chính xác và khả năng diễn giải của mô hình dự báo.

### 2. Tuyên bố vấn đề  
### Vấn để hiện tại 

Dự báo chuỗi thời gian chính xác thường gặp hai thách thức lớn: thiếu đa dạng dữ liệu và thiếu cơ sở vật lý.  

- **Thiếu dữ liệu**: Nhiều bài toán dự báo chuỗi thời gian gặp hạn chế về lượng dữ liệu huấn luyện. Mặc dù có nhiều phương pháp tăng cường dữ liệu, rất ít trong số đó tập trung trực tiếp vào việc mô phỏng sự suy giảm tầm quan trọng của các giá trị trong quá khứ theo thời gian.

- **Bỏ qua yếu tố vật lý:**: Hầu hết các mạng nơ-ron chỉ học từ dữ liệu thô mà không xét đến các ràng buộc vật lý trong thế giới thực. Trong các bài toán dự đoán quỹ đạo (ví dụ: bão), điều này thường dẫn đến những kết quả phi thực tế.

Mục tiêu của chúng tôi là:
+ Phát triển một phương pháp tăng cường chuỗi thời gian mới (STFA) nhằm cải thiện độ bền vững và khả năng khái quát của mô hình.
+ Tích hợp các ràng buộc dựa trên vật lý vào quá trình huấn luyện mô hình, thu hẹp khoảng cách giữa học máy dựa trên dữ liệu và động lực học của thế giới thực.

### Giải Pháp 
## A - Tăng cường phai dần theo từng bước theo thời gian

STFA tạo ra các chuỗi thời gian tổng hợp bằng cách giảm dần ảnh hưởng của các giá trị trong quá khứ. Khác với phương pháp thêm nhiễu ngẫu nhiên, nó có áp dụng hệ thống các hệ số phai dần theo từng dải dữ liệu cũ.  

Giả sử một chuỗi đơn biến được cho bởi:

$$
X = [x_0, x_1, \ldots, x_{T-1}]
$$

trong đó $T$ là độ dài của chuỗi $X$.

**Các tham số**:

- $n$: khoảng giá trị gần nhất muốn giữ nguyên
- $S$: số dải được áp dụng phai dần, mỗi dải được gán một hệ số nhân cố định.
- $L = T - n$: độ dài của vùng phai dần.
- $k = \frac{L}{S}$: số giá trị trong mỗi dải.
- $I_b$: tập chỉ số của dải thứ $b$.

\\[
I_b = \{\, i \mid L - b \cdot k \;\leq\; i \;\leq\; L - (b-1)\cdot k - 1 \,\}
\\]

**Biến đổi:**:

Ta ký hiệu chuỗi sau khi được tăng cường là:

$$
X = [x_0, \ldots, x_{T-1}]
$$


với các quy tắc biến đổi sau:

$$
x_t =
\begin{cases}
x_t, & t \in \{T-n, \ldots, T-1\}, \\\
m_b \, x_t, & t \in I_b, \\\
m_{S+1} \, x_t, & t < \min(I_S),
\end{cases}
$$




trong đó các hệ số nhân $m_b \in (0,1)$ giảm dần một cách đơn điệu từ các dải dữ liệu mới đến các dải dữ liệu cũ.


Công thức này duy trì độ chính xác của các dữ liệu gần đây trong khi kiểm soát chặt chẽ hơn ảnh hưởng của các giá trị xa trong chuỗi. Quá trình tăng cường buộc mô hình tập trung vào các mẫu bền vững vượt ra ngoài dữ liệu gốc, đồng thời tăng tính đa dạng dựa trên các tham số đã chọn.

## B - Học máy dựa trên vật lý

Các mô hình mạng nơ-ron như **RNNs**, **CNNs**, và **Transformers** không cần công thức hay quy tắc đặc thù cho từng tác vụ để đạt hiệu suất tốt, miễn là chúng được huấn luyện với đủ dữ liệu.
Ví dụ, trong bài toán dịch máy như dịch từ tiếng Đức sang tiếng Anh bằng RNN, không có quy tắc ngữ pháp nào được cung cấp trong quá trình huấn luyện. Tuy nhiên, mô hình vẫn có thể tạo ra bản dịch mạch lạc, thể hiện một trong những điểm mạnh chính của học sâu: khả năng học trực tiếp các mẫu phức tạp từ dữ liệu.

Ngược lại, các phương pháp truyền thống — chẳng hạn như các hệ thống dịch dựa trên quy tắc (ví dụ Google Translate trước những năm 2000) — phụ thuộc nhiều vào quy tắc ngữ pháp và từ điển.
Mặc dù chính xác, nhưng các hệ thống này thường thiếu linh hoạt và thất bại khi gặp từ có nhiều nghĩa hoặc các cấu trúc phụ thuộc vào ngữ cảnh.

Lấy cảm hứng từ sự khác biệt đó, mục tiêu của chúng tôi là kết hợp sức mạnh của học sâu với các công thức do con người định nghĩa để đạt hiệu suất tốt hơn.
Cụ thể, trong mô hình dự đoán chuyển động bão của chúng tôi — nơi áp dụng **Stepwise Temporal Fading Augmentation (STFA)** — chúng tôi đưa vào quá trình huấn luyện hai công thức dựa trên vật lý: **khoảng cách Haversine** và **góc phương vị** (bearing).
Hai yếu tố này cung cấp cho mô hình cấu trúc bổ sung và định hướng học tập, giúp mô hình hiểu sâu hơn ngoài các mối tương quan thuần thống kê.

---

### b.1 Công thức Haversine

**Công thức Haversine** được sử dụng rộng rãi để tính khoảng cách đường tròn lớn giữa hai điểm trên bề mặt hình cầu:

$$
\\theta = \text{atan2}\!\left(
  \sin(\Delta \lambda)\cos(\varphi_2),\,
  \cos(\varphi_1)\sin(\varphi_2)
  - \sin(\varphi_1)\cos(\varphi_2)\cos(\Delta \lambda)
\right)
$$



**Trong đó:**

- $$(\varphi_1, \lambda_1)$$ và $$(\varphi_2, \lambda_2)$$ là vĩ độ và kinh độ của hai điểm. 
- $$r$$ là bán kính Trái Đất.


Vì Trái Đất có dạng gần hình cầu, nên công thức Haversine cung cấp một phép xấp xỉ rất chính xác, với sai số nhỏ hơn 1% trong hầu hết các trường hợp.

Trong khung mô hình của chúng tôi, thay vì chỉ dựa vào các hàm mất mát tiêu chuẩn như **MSE**, **RMSE**, hoặc **MAPE**, chúng tôi đề xuất sử dụng công thức Haversine làm **hàm mất mát** (main loss).
Bởi vì mô hình đầu ra là các tọa độ vĩ độ và kinh độ của vị trí bão kế tiếp, công thức Haversine cho phép đo trực tiếp khoảng cách giữa điểm dự đoán và điểm thực tế.

Khoảng cách càng gần 0 cho thấy dự đoán càng chính xác, trong khi khoảng cách lớn cho thấy sai số đáng kể.
Hàm mất mát dựa trên khoảng cách này cũng có thể được kết hợp với các cơ chế huấn luyện phổ biến như **điều chỉnh tốc độ học** (learning rate scheduler) và **dừng sớm** (early stopping) để tận dụng tối đa hiệu quả của nó.

---

### b.2 Góc phương vị

**Công thức góc phương vị** cho biết hướng đi từ một địa điểm địa lý đến điểm khác theo cung tròn lớn:

$$
\theta = \text{atan2}\!\left(
  \sin(\Delta \lambda)\cos(\varphi_2),\,
  \cos(\varphi_1)\sin(\varphi_2)
  - \sin(\varphi_1)\cos(\varphi_2)\cos(\Delta \lambda)
\right)
$$


**Trong dó:**  
- $$(\varphi_1, \lambda_1)$$ là điểm bắt đầu,  
- $$(\varphi_2, \lambda_2)$$ là điểm kết thúc,  
- $$\Delta \lambda$$ là hiệu của kinh độ giữa hai điểm.


Trong quá trình triển khai, chúng tôi sử dụng cả **công thức Haversine** và **góc phương vị** (bearing) để tính toán hai đặc trưng bổ sung — “khoảng cách” và “góc” — và thêm chúng vào tập dữ liệu.
Những đặc trưng này giúp mô hình có thêm thông tin phong phú hơn về quỹ đạo của bão, đồng thời vẫn giữ mục tiêu cốt lõi là dự đoán vị trí địa lý tiếp theo.


### Lợi ích và Hiệu quả đầu tư

- **Tăng hiệu suất**: STFA tạo ra các chuỗi tổng hợp có cấu trúc, giúp tăng độ bền vững của mô hình, giảm hiện tượng overfitting và cải thiện khả năng khái quát trên các quỹ đạo bão chưa từng thấy.

- **Nhận thức vật lý**: Việc tích hợp các nguyên lý địa lý như khoảng cách và góc phương vị giúp tăng khả năng diễn giải và đảm bảo kết quả dự đoán phù hợp với các ràng buộc vật lý.

- **Hướng nghiên cứu mới**: Thiết lập một mô hình tăng cường chuỗi thời gian mới dựa trên sự phai dần của mức độ liên quan theo thời gian, mở rộng phương pháp luận cho các bài toán học chuỗi.

- **Scalability and Reusability**: Khung kết hợp STFA + PIML có thể được mở rộng cho các lĩnh vực dự báo chuỗi khác như nhu cầu năng lượng, lưu lượng giao thông và xu hướng tài chính.

**Tác động tổng thể**: Bằng cách cải thiện độ ổn định và khả năng diễn giải của mô hình trong khi vẫn duy trì tính mở rộng, phương pháp được đề xuất mang lại cả giá trị khoa học lẫn hiệu quả thực tiễn trong đầu tư tính toán.

### 3. Kiến trúc Giải pháp
Nền tảng tích hợp một quy trình dự đoán quỹ đạo bão với việc triển khai AWS có khả năng mở rộng. Dữ liệu bão thô được tiền xử lý thành các tập dữ liệu tuần tự và xử lý qua hai giai đoạn: Giai đoạn 1 học các đặc trưng không-thời gian, trong khi Giai đoạn 2 sử dụng Transformer có trọng số STFA để dự báo quỹ đạo. Kết quả được hợp nhất trong Bộ tổng hợp Quỹ đạo và được đánh giá thông qua các số liệu định lượng và hình ảnh hóa. Hệ thống chạy trên ngăn xếp serverless của AWS sử dụng ECS, Lambda và S3 để xử lý và lưu trữ, với CloudFront và Route 53 cung cấp một bảng điều khiển dự đoán an toàn, có thể mở rộng.

![IoT Weather Station Architecture](/images/2-Proposal/ssv.png)

![Platform Architecture](/images/2-Proposal/platform_architecture.png)

### Các dịch vụ AWS được sử dụng
#### 1. Frontend & CDN
- **Amazon S3**: Lưu trữ các tệp tĩnh từ React + Vite build (2 bucket: frontend và dữ liệu thời tiết).
- 	**Amazon CloudFront** : Phân phối CDN toàn cầu cho ứng dụng React, đồng thời cache các tài nguyên tĩnh.
- 	**Route 53** : Quản lý DNS và định tuyến chứng chỉ SSL.
- 	**AWS Certificate Manager (ACM)** : Cung cấp chứng chỉ SSL/TLS miễn phí.

#### 2. Dịch vụ Backend 
- **Amazon ECS Fargate**: Chạy các container API .NET Core (3 tác vụ, tự động mở rộng).
- 	**Application Load Balancer (ALB)** : Phân phối lưu lượng đến các tác vụ ECS.
- 	**Amazon ElastiCache (Redis)** : Lớp bộ nhớ đệm cho phản hồi API và dữ liệu phiên.
- 	**Amazon RDS PostgreSQL** : Cơ sở dữ liệu chính lưu trữ dữ liệu bão và lịch sử dự đoán.

#### 3. ML & Xử lý dữ liệu
- **AWS Lambda**: 
    - Lambda Container cho suy luận ML (Python + TensorFlow, tải mô hình .h5)
    - Lambda Function cho thu thập dữ liệu (chạy theo lịch định sẵn)
- 	**Amazon EFS (Elastic File System)** : lưu trữ tệp mô hình .h5 (dùng chung cho Lambda).
- 	**Route 53** : Data lake cho dữ liệu thời tiết thô và sao lưu mô hình ML.
- 	**Amazon EventBridge** : Bộ lập lịch cho việc thu thập dữ liệu tự động (cron job theo giờ/ngày).

#### 4. Bảo mật & Giám sát
- **AWS WAF**: Tường lửa của ứng dụng web bảo vệ API khỏi các cuộc tấn công.
- 	**Amazon CloudWatch** : Ghi log, theo dõi chỉ số và giám sát tất cả dịch vụ.
- 	**AWS Secrets Manager** : Quản lý khóa API, thông tin đăng nhập cơ sở dữ liệu và token của bên thứ ba.
- 	**VPC + Security Groups** : Cô lập mạng và kiểm soát truy cập.


### Thiết kế thành phần
#### Kiến trúc luồng dữ liệu
##### 1. Lớp giao diện người dùng
- **Frontend**: Ứng dụng React + Vite được host trên S3, phân phối qua CloudFront CDN.
- **Authentication**: (Tùy chọn) Amazon Cognito cho quản lý người dùng nếu cần đăng nhập.
- **Data Storage**: Dữ liệu thô được lưu trong data lake trên S3; dữ liệu đã xử lý được lưu trong một bucket S3 khác.
- **Data Processing**: AWS Glue Crawlers lập catalog dữ liệu, các job ETL chuyển đổi dữ liệu phục vụ phân tích.
- **Real-time Updates**: WebSocket hoặc API polling để hiển thị dự đoán theo thời gian thực.
##### 2. Lớp API
- **Load Balancer**: ALB nhận các yêu cầu HTTPS từ CloudFront/người dùng.
- **Backend API**: 3 tác vụ ECS Fargate chạy .NET Core Web API.
    - **Route 1**: /api/typhoons - các thao tác CRUD.
    - **Route 2**: /api/predict - điểm cuối (endpoint) dùng cho mô hình dự đoán học máy.
    - **Route 3**: /api/weather - API dữ liệu thời tiết.
##### 3. Lớp lưu trữ dữ liệu
- **Database**:RDS PostgreSQL lưu trữ:
    - Dữ liệu lịch sử bão (quỹ đạo, thời gian, cường độ)
    - Kết quả dự đoán
    - Dữ liệu người dùng

- **Data Lake**: Các bucket S3 lưu:
    - Dữ liệu thời tiết thô (JSON/CSV từ API bên ngoài)
    - File mô hình ML (.h5, .pkl)

##### 4. Dịch vụ dự đoán ML
- **Lambda Container**:(Python + TensorFlow):
    - Đầu vào: đặc trưng bão (lat, lon, pressure, wind_speed, v.v.)
    - Xử lý: Tải mô hình model.h5 từ EFS → dự đoán hướng di chuyển.
    - Đầu ra: Hướng di chuyển, độ tin cậy (confidence score), và xác suất dự đoán.

- **Luồng xử lý** (Workflow): 
    - 1.	.NET API nhận yêu cầu dự đoán từ người dùng
    - 2.	API gọi URL function của Lambda
    - 3.	Lambda tải mô hình (model.h5) từ EFS.
    - 4.	Lambda lấy dữ liệu thời tiết từ S3.
    - 5.	Lambda thực hiện suy luận (inference).
    - 6.	Trả kết quả dự đoán về lại .NET API.
    - 7.	API lưu tạm (cache) kết quả vào Redis.

##### 5. Pipeline thu thập dữ liệu
- **EventBridge Scheduler**: Kích hoạt Lambda mỗi 1 giờ.
- **Lambda Data Collector:**
    -	Gọi các API thời tiết bên ngoài (NOAA, JMA, v.v.).
    -	Phân tích cú pháp (parse) và kiểm tra tính hợp lệ của dữ liệu.
    -	Lưu dữ liệu thô vào S3.
    -	Cập nhật dữ liệu đã xử lý vào RDS.
- **Tích hợp với đội AI:** 
    - đội AI tải model.h5 mới vào EFS
    - Lambda sẽ tự động tải lại mô hình ở lần thực thi tiếp theo

##### 6. Giám sát & Bảo mật
- **CloudWatch:**
    - Ghi log từ ECS, Lambda, và ALB.
    - Theo dõi chỉ số (metrics): CPU, bộ nhớ (Memory), số lượng yêu cầu (Request count), và tỷ lệ lỗi (Error rate).
    - Thiết lập cảnh báo (alarms): khi tỷ lệ lỗi cao, độ trễ lớn, hoặc hệ thống giảm khả dụng.

- **Secrets Manager: lưu trữ:**
    - Thông tin đăng nhập (credentials) của RDS.
    - Mật khẩu của Redis.
    - API keys bên ngoài (nguồn dữ liệu thời tiết)

- **WAF Rules:**: 
    - Giới hạn tốc độ (rate limiting) — tối đa 100 yêu cầu/phút/mỗi IP.
    - Bảo vệ khỏi SQL injection.
    - Bảo vệ khỏi tấn công XSS.

### Ước tính ngân sách
#### Chi phí hạ tầng - Hàng tháng (ap-southeast-1 Singapore)

### Chi phí hạ tầng
#### Frontend & CDN
- **S3 Standard**: $0.50/Tháng (5GB lưu trữ, 10GB truyền tải)
- **CloudFront**: $4.25/tháng (50GB truyền dữ liệu, 1 triệu yêu cầu)
- **Route 53**: $0.50/tháng (1 vùng lưu trữ, 1 triệu truy vấn)
- **ACM (SSL)**: $0.00/tháng (miễn phí)
Tổng: $5.25/tháng

#### Dịch vụ Backend
- **ECS Fargate**: $45.00/tháng (3 tác vụ × 0.5 vCPU, 1GB RAM, hoạt động liên tục)
- **ALB**: $4.25/tháng (50GB truyền dữ liệu, 1 triệu yêu cầu)
- **Route 53**: $16.00/tháng (bộ cân bằng tải cơ bản, 1 triệu LCU)
- **ElastiCache Redis**: $12.00/tháng (cache.t3.micro, 0.5GB)
- **RDS PostgreSQL**:$20.00/tháng (db.t3.micro, 20GB SSD)

Tổng: $93.00/tháng

#### ML & Xử lý dữ liệu
- **Lambda Container**: $5.00/tháng (1.000 lần gọi/ngày × 2GB RAM × 3 giây)
- **Lambda Data Collector**: $0.50/tháng (24 lần gọi/ngày × 512MB × 30 giây)
- **EFS**: $0.33/tháng (1GB lưu trữ – model.h5)
- **S3 Data Lake**: $1.50/tháng (50GB dữ liệu thời tiết, 10GB truyền)
- **EventBridge**:$0.00/tháng (730 sự kiện định kỳ/tháng)

Tổng: $7.33/tháng

#### Bảo mật & Giám sát
- **CloudWatch Logs**: $2.50/tháng (5GB dữ liệu ghi, 1GB lưu trữ)
- **CloudWatch Metrics**: $0.60/tháng (20 chỉ số tùy chỉnh)
- **Secrets Manager**: $2.00/tháng (5 bí mật)
- **AWS WAF**: $12.00/tháng (cache.t3.micro, 0.5GB)

Tổng: $15.10/tháng

#### Mạng (Networking)
- **VPC**: $0.00/tháng (miễn phí – không dùng VPN hoặc PrivateLink)
- **NAT Gateway**: $32.85/tháng (1 NAT × $0.045/giờ × 730 giờ)
- **Data Transfer (Outbound)**: $1.80/tháng (20GB ra Internet)
- **ECR**: $0.10/tháng (1GB lưu trữ)

Tổng: $34.75/tháng

#### **TỔNG CỘNG**
- **Frontend & CDN**: $5.25/tháng
- **Dịch vụ Backend**: $93.00/tháng 
- **ML & Xử lý dữ liệu**: $7.33/tháng 
- **Bảo mật & Giám sát**: $34.75/tháng
- **Mạng** (Networking): $15.10/tháng 

**Tổng cộng** $155.43/tháng

### 7. Đánh giá rủi ro

#### Ma trận rủi ro
- Quá tải ECS Fargate: **Tác động cao**, **xác suất trung bình**.  
- Độ trễ khởi động lạnh của Lambda: **Tác động trung bình**, **xác suất cao**.  
- Lỗi Redis Cache: **Tác động trung bình**, **xác suất trung bình**.  
- Sự cố RDS Single-AZ: **Tác động cao**, **xác suất trung bình**.  
- Lỗi EventBridge hoặc Data Collector: **Tác động trung bình**, **xác suất trung bình**.  
- Rò rỉ API Key hoặc thông tin xác thực: **Tác động cao**, **xác suất trung bình**.  
- Tấn công DDoS / brute-force: **Tác động cao**, **xác suất trung bình**.  
- Chi phí AWS tăng đột biến ngoài dự kiến: **Tác động trung bình**, **xác suất cao**.  
- Dung lượng S3 Data Lake tăng quá mức: **Tác động trung bình**, **xác suất trung bình**.  
- Cập nhật mô hình không được theo dõi: **Tác động cao**, **xác suất trung bình**.  

#### Chiến lược giảm thiểu
- **Tầng tính toán (ECS, Lambda):** Dùng Auto Scaling và gắn EFS để tránh quá tải và giảm khởi động lạnh.  
- **Bộ nhớ đệm & Cơ sở dữ liệu:** Thiết lập Multi-AZ cho RDS và ElastiCache, kèm logic dự phòng khi cache lỗi.  
- **Bảo mật:** Lưu khóa bí mật trong AWS Secrets Manager, áp dụng WAF để giới hạn tần suất và chặn SQLi/XSS. 
- **Kiểm soát chi phí:** Dùng AWS Budgets và cảnh báo CloudWatch; thiết lập VPC Endpoint để giảm chi phí NAT; quy tắc vòng đời S3 để lưu trữ lâu dài.
- **Quản lý dữ liệu & mô hình ML:** Lưu phiên bản mô hình trong RDS và tự động tải lại qua EventBridge.
- **Giám sát:** Dùng bảng điều khiển CloudWatch, cảnh báo và thông báo SNS cho ECS, Lambda, ALB, và RDS.  

#### Kế hoạch dự phòng
- **Sự cố hệ thống:** RDS Multi-AZ cùng sao lưu S3; triển khai lại ECS qua CloudFormation. 
- **Lỗi dịch vụ dự đoán:** Cung cấp dự đoán cuối cùng từ Redis cho đến khi khôi phục. 
- **Lỗi pipeline dữ liệu:** Lưu tạm dữ liệu đến trong S3 buffer cho đến khi Lambda hoạt động lại. 
- **Vượt chi phí:** Tự động giảm quy mô ECS và Lambda bằng trigger CloudWatch. 
- **Rò rỉ bảo mật:** Thay đổi thông tin trong Secrets Manager và cô lập các vai trò IAM bị ảnh hưởng.

### 8. Kết quả kỳ vọng

#### **Cải tiến kỹ thuật**
- Tự động hóa dự đoán quỹ đạo bão, thay thế quy trình phân tích thủ công.
- Nâng cao độ chính xác dự báo nhờ mô hình Transformer có trọng số STFA.
- Xây dựng pipeline mở rộng trên AWS, hỗ trợ thu thập dữ liệu và huấn luyện theo thời gian thực.

#### **Giá trị dài hạn**
- Hình thành bộ dữ liệu quỹ đạo bão kéo dài trong năm, phục vụ nghiên cứu AI và khí hậu nâng cao. 
- Cung cấp kiến trúc gốc trên đám mây, tái sử dụng cho các dự án dự báo không gian-thời gian khác.
- Cho phép tích hợp thêm trạm thời tiết IoT và API khí tượng bên ngoài.




