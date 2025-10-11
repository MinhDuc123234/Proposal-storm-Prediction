---
title: "Proposal"
date: 2025-09-09
weight: 2
chapter: false
pre: " <b> 2. </b> "
---

# FORECASTING MODEL FOR HURRICANE TRAJECTORY PREDICTION
## Stepwise Temporal Fading Augmentation for Time-Series Forecasting and Physics-Informed Machine Learning

### 1. Executive Summary

Time-series forecasting underpins many scientific and industrial applications, from meteorology to financial modeling. Despite advances in model architectures, the quality and diversity of training data remain decisive factors for performance. Existing data augmentation techniques—such as random perturbation, slicing, or noise injection—often distort temporal dependencies and fail to represent the natural fading influence of past events. This gap highlights the need for a principled approach that can preserve sequential coherence while capturing the gradual decay of temporal relevance.

This proposal presents a new time-series augmentation framework called Stepwise Temporal Fading Augmentation (STFA). Unlike traditional methods based on random perturbations or noise, STFA models the natural decline in the influence of past observations by applying fading weights to earlier values while preserving recent ones. It generates realistic and diverse synthetic sequences that improve model robustness. The approach will be evaluated on hurricane trajectory prediction, which relies on sequential latitude–longitude data. In addition, Physics-Informed Machine Learning (PIML) principles are incorporated by embedding geographical relationships—such as haversine distance and bearing—into both the feature set and the loss function. This hybrid design combines the flexibility of deep learning with the rigor of physical constraints to enhance prediction accuracy and interpretability.

### 2. Problem Statement
### What’s the Problem?

Accurate time-series forecasting often faces two major challenges: limited data diversity and the lack of physical grounding.

- **Data scarcity**: Many time-series forecasting tasks suffer from limited training data. While there are various augmentation methods, few approaches directly focus on the declining importance of past values over time.

- **Physics ignorance**: Most neural networks only learn from raw data, without considering real-world physical constraints. In trajectory prediction tasks (e.g., hurricanes), this often leads to unrealistic predictions.

We aim to:
+ Develop a new time-series augmentation method (STFA) to improve robustness and generalization.
+ Incorporate physics-based constraints into model training, bridging the gap between data-driven learning and real-world dynamics.

### The Solution
#### A - Stepwise temporal fading augumentation

STFA generates synthetic time-series sequences by gradually reducing the influence of earlier values. Unlike random noise injection, it systematically applies stepwise fading multipliers across bands of older data.

Let a univariate sequence be:

$$
X = [x_0, x_1, \ldots, x_{T-1}]
$$

where $T$ is the sequence length of $X$.

**Parameters**:

- $n$: number of most steps to remain unchanged.  
- $S$: number of step-bands to apply fading, each band is assigned a constant multiplier.  
- $L = T - n$: length of the fading region.  
- $k = \frac{L}{S}$: values per band.  
- $I_b$: index set of the $b$-th band.  


\\[
I_b = \{\, i \mid L - b \cdot k \;\leq\; i \;\leq\; L - (b-1)\cdot k - 1 \,\}
\\]

**Transformation**:

We denote the augmented series as:

$$
X = [x_0, \ldots, x_{T-1}]
$$


with the transformation rules:

$$
x_t =
\begin{cases}
x_t, & t \in \{T-n, \ldots, T-1\}, \\\
m_b \, x_t, & t \in I_b, \\\
m_{S+1} \, x_t, & t < \min(I_S),
\end{cases}
$$




where multipliers $m_b \in (0,1)$ decrease monotonically from recent to older bands.


This formulation preserves the fidelity of recent history while exerting stronger control on the long-range influence of the sequence. The augmentation forces the model to focus on robust patterns beyond the raw data, while increasing diversity according to the chosen parameters.

## B) Physics-Informed Machine Learning

Neural network models such as **RNNs**, **CNNs**, and **Transformers** do not require explicit formulas or task-specific rules to perform well, provided that they are trained with sufficient data.  
For example, in machine translation tasks such as *German-to-English translation* using an RNN, no explicit grammar rules are provided during training. Nevertheless, the model is capable of producing coherent translations, which demonstrates one of the major strengths of deep learning: the ability to learn complex patterns directly from data.

In contrast, traditional approaches—such as early versions of rule-based translation systems (e.g., Google Translate prior to the 2000s)—relied heavily on grammar rules and dictionaries.  
While precise, such systems often lacked flexibility and failed when encountering words with multiple meanings or when handling context-dependent structures.

Inspired by this contrast, our goal is to combine the strengths of deep learning with human-defined formulas in order to achieve better performance.  
Specifically, in our hurricane movement prediction model where we implement **Stepwise Temporal Fading Augmentation (STFA)**, we introduce two physics-based formulas into training: the **Haversine distance** and the **bearing**.  
These provide the model with additional structure and inductive bias, guiding learning beyond purely statistical correlations.

---

### b.1 Haversine Formula

The **Haversine formula** is widely used to compute the great-circle distance between two points on the surface of a sphere:

$$
\\theta = \text{atan2}\!\left(
  \sin(\Delta \lambda)\cos(\varphi_2),\,
  \cos(\varphi_1)\sin(\varphi_2)
  - \sin(\varphi_1)\cos(\varphi_2)\cos(\Delta \lambda)
\right)
$$



**Where:**

- $$(\varphi_1, \lambda_1)$$ and $$(\varphi_2, \lambda_2)$$ are the latitudes and longitudes of the two points.  
- $$r$$ is the Earth’s radius.


Since the Earth is approximately spherical, the Haversine formula provides an accurate approximation, with less than **1% error** in most cases.

In our framework, instead of relying solely on standard loss functions such as **MSE**, **RMSE**, or **MAPE**, we propose using the Haversine distance as the **primary loss function**.  
Because the model outputs latitude and longitude coordinates for the next hurricane location, the Haversine formula directly measures the distance between predicted and ground-truth points.

A distance close to zero indicates a highly accurate prediction, while a large distance signals a significant error.  
This distance-based loss can also be combined with common training mechanisms such as **learning rate schedulers** and **early stopping** to fully exploit its potential.

---

### b.2 Bearing

The **bearing formula** gives the direction from one geographic point to another along the great circle path:

$$
\theta = \text{atan2}\!\left(
  \sin(\Delta \lambda)\cos(\varphi_2),\,
  \cos(\varphi_1)\sin(\varphi_2)
  - \sin(\varphi_1)\cos(\varphi_2)\cos(\Delta \lambda)
\right)
$$


**Where:**  
- $$(\varphi_1, \lambda_1)$$ is the start point,  
- $$(\varphi_2, \lambda_2)$$ is the end point,  
- $$\Delta \lambda$$ is the difference in longitude.


In our implementation, we use both the **Haversine distance** and **bearing** to compute two additional features — “distance” and “bearing” — which are appended to the dataset.  
These features provide the model with richer information about hurricane trajectories while maintaining the core objective of predicting the next geographic location.


### Benefits and Return on Investment

- **Performance Boost**: STFA generates structured synthetic sequences that enhance model robustness, reduce overfitting, and improve generalization on unseen storm trajectories.

- **Physics Awareness**: Incorporating geographical principles such as distance and bearing increases interpretability and ensures physically consistent predictions.

- **New Research Direction**: Establishes a novel paradigm for time-series augmentation based on temporal relevance fading, expanding the methodological toolkit for sequence learning.

- **Scalability and Reusability**: The combined STFA + PIML framework can be extended to other sequential forecasting domains such as energy demand, traffic flow, and financial trends.

**Overall Impact**: By improving predictive stability and interpretability while maintaining scalability, the proposed approach delivers both scientific value and practical return on computational investment.

### 3. Solution Architecture
The platform integrates a hurricane trajectory prediction pipeline with a scalable AWS deployment. Raw storm data is preprocessed into sequential datasets and processed through two phases: Phase 1 learns spatio-temporal features, while Phase 2 uses an STFA-weighted Transformer for trajectory forecasting. Results are merged in the Trajectory Synthesizer and evaluated through quantitative metrics and visualization. The system runs on a serverless AWS stack using ECS, Lambda, and S3 for processing and storage, with CloudFront and Route 53 delivering a secure, scalable prediction dashboard.

![IoT Weather Station Architecture](/images/2-Proposal/ssv.png)

![Platform Architecture](/images/2-Proposal/platform_architecture.png)

### AWS Services Used
#### 1. Frontend & CDN
- **Amazon S3**: Storage static files React + Vite build (2 buckets: frontend + weather data).
- 	**Amazon CloudFront** : CDN global distribution for React app, caching static assets.
- 	**Route 53** : DNS management and SSL certificate routing.
- 	**AWS Certificate Manager (ACM)** : SSL/TLS certificates (free).

#### 2. Backend Services
- **Amazon ECS Fargate**: run .NET Core API containers (3 tasks by auto-scaling).
- 	**Application Load Balancer (ALB)** : Distribute traffic to ECS tasks.
- 	**Amazon ElastiCache (Redis)** : Cache layer for API responses và session data.
- 	**Amazon RDS PostgreSQL** : main database storage typhoon data, predictions history.

#### 3. ML & Data Processing
- **AWS Lambda**: 
    - Lambda Container for ML inference (Python + TensorFlow, load model .h5)
    - Lambda Function for data collection (scheduled jobs)
- 	**Amazon EFS (Elastic File System)** : storage model.h5 file (shared storage cho Lambda).
- 	**Route 53** : Data lake for raw weather data and ML model backups.
- 	**Amazon EventBridge** : Scheduler for automated data collection (hourly/daily cron jobs).

#### 4. Security & Monitoring
- **AWS WAF**: Web Application Firewall protect API from attacks.
- 	**Amazon CloudWatch** : Logs, metrics, monitoring for every services.
- 	**AWS Secrets Manager** : Management API keys, database credentials, third-party tokens.
- 	**VPC + Security Groups** : Network isolation and access control.


### Component Design
#### Data Flow Architecture
##### 1. User Interface Layer
- **Frontend**:React + Vite app hosted on S3, distribute by CloudFront CDN.
- **Authentication**: (Optional) Amazon Cognito for user management if need login.
- **Data Storage**: Raw data is stored in an S3 data lake; processed data is stored in another S3 bucket.
- **Data Processing**: AWS Glue Crawlers catalog the data, and ETL jobs transform it for analysis.
- **Real-time Updates**: WebSocket or polling API to display predictions real-time.
##### 2. API Layer
- **Load Balancer**:ALB receive HTTPS requests from CloudFront/Users.
- **Backend API**: 3x ECS Fargate tasks run .NET Core Web API. 
    - **Route 1**: /api/typhoons - CRUD operations
    - **Route 2**: /api/predict - ML prediction endpoint
    - **Route 3**: /api/weather - Weather data API.
##### 3. Data Storage Layer
- **Database**:RDS PostgreSQL storage:
    - Typhoon historical data (trajectory, time, intensity)
    - Prediction results
    - User data 

- **Data Lake**: S3 buckets lưu:
    - Raw weather data (JSON/CSV from external APIs)
    - ML model files (.h5, .pkl)

##### 4. ML Prediction Service
- **Lambda Container**:(Python + TensorFlow):
    - Input: Typhoon features (lat, lon, pressure, wind_speed, etc.)
    - Process: Load model.h5 from EFS → Predict direction
    - Output: Direction, confidence score, probabilities

- **Workflow**: 
    - 1.	.NET API receive request predict from user
    - 2.	API call Lambda function URL
    - 3.	Lambda load model from EFS 
    - 4.	Lambda fetch weather data từ S3
    - 5.	Lambda run inference
    - 6.	Return prediction to .NET API
    - 7.	API cache result into Redis

##### 5. Data Collection Pipeline
- **EventBridge Scheduler**:Trigger Lambda each 1 hour
- **Lambda Data Collector:**
    -	Call external weather APIs (NOAA, JMA, etc.)
    -	Parse and validate data
    -	Store raw data to S3
    -	Update processed data into RDS

- **AI Team Integration:** 
    - AI team upload model.h5 new into EFS
    - Lambda auto reload model at next invocation 

##### 6. Monitoring & Security
- **CloudWatch:**
    - Logs from ECS, Lambda, ALB
    - Metrics: CPU, Memory, Request count, Error rate
    - Alarms: High error rate, High latency, Low availability

- **Secrets Manager: save:**
    - RDS credentials
    - Redis password
    - External API keys (weather data sources)

- **WAF Rules:**: 
    - Rate limiting (max 100 requests/minute/IP)
    - SQL injection protection
    - XSS protection

### Budget Estimation
#### Infrastructure Costs - Monthly (ap-southeast-1 Singapore)

### Infrastructure Costs
#### Frontend & CDN
- **S3 Standard**: $0.50/Month (5GB storage, 10GB transfer)
- **CloudFront**: $4.25/Month (50GB data transfer, 1M requests)
- **Route 53**: $0.50/Month (1 hosted zone, 1M queries)
- **ACM (SSL)**:$0.00/Month (Free)

Total: $5. 25/month

#### Backend Services
- **ECS Fargate**: $45.00/Month (3 tasks × 0.5  vCPU, 1GB RAM (always-on))
- **ALB**: $4.25/Month (50GB data transfer, 1M requests)
- **Route 53**: $16.00/Month (Basic load balancer, 1M LCUs)
- **ElastiCache Redis**: $12.00/Month (cache.t3.micro (0.5GB))
- **RDS PostgreSQL**:$20.00/Month (db.t3.micro, 20GB SSD)

Total $93.00/Month

#### ML & Data Processing
- **Lambda Container**: $5.00/Month (1,000 invocations/day × 2GB RAM × 3s duration)
- **Lambda Data Collector**: $0.50/Month (24 invocations/day × 512MB × 30s)
- **EFS**: $0.33/Month (1GB storage (model.h5))
- **S3 Data Lake**: $1.50/Month (50GB weather data, 10GB transfer)
- **EventBridge**:$0.00/Month (730 scheduled events/month)

Total $7.33/Month

#### Security & Monitoring
- **CloudWatch Logs**: $2.50/Month (5GB ingestion, 1GB storage)
- **CloudWatch Metrics**: $0.60/Month (20 custom metrics)
- **Secrets Manager**: $2.00/Month (5 secrets)
- **AWS WAF**: $12.00/Month (cache.t3.micro (0.5GB))

Total $15.10/Month

#### Networking
- **VPC**: $0.00/Month (Free (no VPN, no PrivateLink))
- **NAT Gateway**: $32.85/Month (1 NAT × 0.045 usd/hour × 730h)
- **Data Transfer (Outbound)**: $1.80/Month (20GB to Internet)
- **ECR**: $0.10/Month (1GB storage)

Total $34.75/Month

#### **TOTAL**
- **Frontend & CDN**: $5.25/Month 
- **Backend Services**: $93.00/Month 
- **ML & Data Processing**: $7.33/Month 
- **Security & Monitoring**: $34.75/Month 
- **Networking**: $15.10/Month 

**Total** $155.43/month 

### 7. Risk Assessment

#### Risk Matrix
- ECS Fargate Overload: **High impact**, **medium probability**.  
- Lambda Cold Start Delays: **Medium impact**, **high probability**.  
- Redis Cache Failure: **Medium impact**, **medium probability**.  
- RDS Single-AZ Outage: **High impact**, **medium probability**.  
- EventBridge or Data Collector Failure: **Medium impact**, **medium probability**.  
- API Key or Credential Leakage: **High impact**, **medium probability**.  
- DDoS / Brute-force Attacks: **High impact**, **medium probability**.  
- Unexpected AWS Cost Spikes: **Medium impact**, **high probability**.  
- S3 Data Lake Overgrowth: **Medium impact**, **medium probability**.  
- Untracked Model Updates: **High impact**, **medium probability**.  

#### Mitigation Strategies
- **Compute Layer (ECS, Lambda):** Auto Scaling policies and EFS mount to prevent overload and cold starts.  
- **Caching & Database:** Multi-AZ setup for RDS and ElastiCache, plus fallback logic if cache fails.  
- **Security:** Store secrets in AWS Secrets Manager, enforce WAF rules for rate limiting and SQLi/XSS.  
- **Cost Control:** AWS Budgets and CloudWatch alerts; VPC Endpoints to cut NAT costs; S3 lifecycle rules for data archiving.  
- **Data & ML Model Management:** Maintain model versioning in RDS and automate reloads via EventBridge.  
- **Monitoring:** Use CloudWatch dashboards, alarms, and SNS notifications for ECS, Lambda, ALB, and RDS metrics.  

#### Contingency Plans
- **System Outages:** RDS Multi-AZ with S3 backups; redeploy ECS tasks via CloudFormation.  
- **Prediction Service Failure:** Serve last cached prediction from Redis until recovery.  
- **Data Pipeline Failure:** Temporarily store incoming data in S3 buffer until Lambda resumes.  
- **Cost Overruns:** Auto-scale down ECS and Lambda through CloudWatch triggers.  
- **Security Breach:** Rotate Secrets Manager credentials and isolate compromised IAM roles.  


### 8. Expected Outcomes

#### **Technical Improvements**
- Automated hurricane trajectory prediction replaces manual data interpretation.  
- Improved forecast accuracy through STFA-weighted Transformer modeling.  
- Scalable AWS-based pipeline supporting real-time data ingestion and retraining.  

#### **Long-term Value**
- Establishes a year-long hurricane trajectory dataset for advanced AI and climate research.  
- Provides a reusable, cloud-native architecture for future spatio-temporal forecasting projects.  
- Enables integration with additional IoT weather stations and external meteorological APIs.

