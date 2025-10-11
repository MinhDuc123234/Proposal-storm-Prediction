---
title: "Blog 1"
date: "2025-09-09"
weight: 1
chapter: false
pre: " <b> 3.1. </b> "
---

# Đưa AI agent từ thử nghiệm đến triển khai thực tế với Amazon Bedrock AgentCore

Để xây dựng một AI agent có thể hoạt động hiệu quả trong môi trường thực tế hiện nay là một thách thức lớn. Nó như một bản thử nghiệm cho thấy tiềm năng của nó, nhưng khi bước sang giai đoạn triển khai sản xuất thì hàng loạt vấn đề mới sẽ xuất hiện. Doanh nghiệp cần phải tính đến khả năng mở rộng quy mô để phục vụ nhiều người dùng đồng thời, đảm bảo bảo mật dữ liệu và quyền truy cập, giám sát được hiệu năng lẫn hành vi của agent, cũng như duy trì tính ổn định vận hành lâu dài. Đây đều là những khía cạnh thường không bộc lộ ở giai đoạn phát triển thử nghiệm.

Bài viết này giới thiệu cách **Amazon Bedrock AgentCore** hỗ trợ chuyển đổi các AI agent từ ý tưởng ban đầu thành hệ thống sẵn sàng cho sản xuất. Thông qua ví dụ thực tế về một agent hỗ trợ khách hàng, nội dung cho thấy hành trình từ nguyên mẫu chạy trên máy cục bộ đến giải pháp cấp máy chủ từ xa cho doanh nghiệp, đủ sức đáp ứng nhiều người dùng cùng lúc trong khi vẫn giữ vững chuẩn mực về hiệu năng và bảo mật.

**Amazon Bedrock AgentCore** sẽ cung cấp một bộ dịch vụ toàn diện, được thiết kế để giải quyết từng nhu cầu cụ thể trong quá trình xây dựng và vận hành agent:

- **AgentCore Runtime**: đảm bảo triển khai và mở rộng quy mô agent một cách an toàn.

- **AgentCore Gateway**: phát triển và tái sử dụng các công cụ phục vụ doanh nghiệp.

- **AgentCore Identity**: quản lý danh tính và quyền truy cập, tăng cường bảo mật cho AI agent ở quy mô lớn.

- **AgentCore Memory**: giúp agent ghi nhớ ngữ cảnh, tạo khả năng tương tác thông minh và tự nhiên hơn.

- **AgentCore Code Interpreter**: cho phép triển khai code để xử lý các yêu cầu phức tạp.

- **AgentCore Browser Tool**: hỗ trợ nhu cầu truy cập và tương tác với thông tin trên web.

- **AgentCore Observability**: cung cấp khả năng giám sát và minh bạch trong việc làm  của agent trong thực tế.

Sự kết hợp của các dịch vụ này tạo nên một nền tảng mạnh mẽ, giúp doanh nghiệp không chỉ dừng ở bước chứng minh ý tưởng mà còn xây dựng được AI agent hoàn thiện, sẵn sàng đáp ứng yêu cầu khắt khe của môi trường sản xuất. Điều này mang lại lợi ích to lớn: rút ngắn thời gian triển khai, giảm rủi ro vận hành, và quan trọng hơn là tạo ra các AI agent có thể mang lại giá trị thực tế cho người dùng cuối.

---

## Tổng quan phương pháp
Mọi hệ thống sản xuất đều bắt đầu từ một bản thử nghiệm, và agent hỗ trợ khách hàng cũng vậy. Ở giai đoạn đầu, ta xây dựng nguyên mẫu có thể hoạt động, chứng minh các khả năng cốt lõi cần thiết cho hỗ trợ khách hàng. Ví dụ này dùng Strands Agents (một framework mã nguồn mở) để tạo proof of concept và **Claude 3.7 Sonnet** trên **Amazon Bedrock** làm mô hình ngôn ngữ nền tảng (LLM). Với ứng dụng khác, có thể chọn framework hoặc mô hình khác phù hợp.

Agent cần công cụ để thực hiện hành động và tương tác với hệ thống thực. Trong trường hợp hỗ trợ khách hàng, có nhiều công cụ có thể được dùng, nhưng để đơn giản, bài viết tập trung vào ba khả năng chính, phục vụ các yêu cầu phổ biến nhất:

- **Tra cứu chính sách đổi trả** – Cung cấp thông tin cấu trúc về thời gian đổi trả, điều kiện áp dụng, hoàn tiền và chính sách vận chuyển theo từng nhóm sản phẩm.

- **Truy xuất thông tin sản phẩm** – Lấy thông số kỹ thuật, chi tiết bảo hành và tính tương thích, hữu ích cho cả câu hỏi trước khi mua và khi cần xử lý sự cố.

- **Tìm kiếm web cho xử lý sự cố** – Hỗ trợ các vấn đề kỹ thuật phức tạp bằng cách truy cập các hướng dẫn, giải pháp mới nhất từ cộng đồng hoặc nguồn mở bên ngoài.

- Phần triển khai công cụ và toàn bộ mã nguồn cho trường hợp này có sẵn trên GitHub. Bài viết chỉ tập trung vào đoạn mã chính kết nối với Amazon Bedrock AgentCore, còn hành trình end-to-end có thể theo dõi chi tiết trong [GitHub repository](https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/01-tutorials/07-AgentCore-E2E)  .

---
## Tạo agent
Với các công cụ có sẵn,ai cũng có thể tạo ra agent cho riêng minh.Bảng diagram dưới đây sẽ cho thấy điều đó hoàn toàn khả thi:
![Kiến trúc AgentCore](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/09/03/image-1-3.png)

Bạn có thể tìm thấy toàn bộ mã nguồn end-to-end của bài viết này trong kho GitHub. Để đơn giản, ở đây chỉ trình bày những phần quan trọng nhất của mã:
```yaml
from strands import Agent
from strands.models import BedrockModel

@tool
def get_return_policy(product_category: str) -> str:
    """Get return policy information for a specific product category."""
    # Returns structured policy info: windows, conditions, processes, refunds
    # check github for full code
    return {"return_window": "10 days", "conditions": ""}
    
@tool  
def get_product_info(product_type: str) -> str:
    """Get detailed technical specifications and information for electronics products."""
    # Returns warranty, specs, features, compatibility details
    # check github for full code
    return {"product": "ThinkPad X1 Carbon", "info": "ThinkPad X1 Carbon info"}
    
@tool
def web_search(keywords: str, region: str = "us-en", max_results: int = 5) -> str:
    """Search the web for updated troubleshooting information."""
    # Provides access to current technical solutions and guides
    # check github for full code
    return "results from websearch"
    
# Initialize the Bedrock model
model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    temperature=0.3
)

# Create the customer support agent
agent = Agent(
    model=model,
    tools=[
        get_product_info, 
        get_return_policy, 
        web_search
    ],
    system_prompt="""You are a helpful customer support assistant for an electronics company.
    Use the appropriate tools to provide accurate information and always offer additional help."""
) 
```
---

## Kiểm chứng thực tế của bản thử nghiệm
Bản thử nghiệm đã cho thấy được sự thành công của agent có thể xử lý nhiều tình huống hỗ trợ khách hàng khác nhau khi được kết hợp đúng công cụ và khả năng suy luận. Agent chạy mượt trên máy cục bộ và phản hồi chính xác các truy vấn. Tuy nhiên, đây cũng là lúc những khoảng trống giữa thử nghiệm và triển khai thực tế lộ rõ. Các công cụ hiện được định nghĩa như hàm cục bộ trong code của agent, agent phản hồi nhanh chóng và thoạt nhìn có vẻ đã sẵn sàng cho sản xuất. Nhưng ngay khi nghĩ đến việc phục vụ ngoài phạm vi một người dùng đơn lẻ, hàng loạt hạn chế nghiêm trọng xuất hiện:
- **Mất trí nhớ giữa các phiên** – Khi khởi động lại notebook hoặc ứng dụng, agent quên toàn bộ lịch sử hội thoại. Một khách hàng đang trao đổi về việc trả laptop hôm qua sẽ phải bắt đầu lại từ đầu hôm nay, lặp lại toàn bộ tình huống. Điều này không chỉ gây bất tiện mà còn làm gián đoạn trải nghiệm trò chuyện, vốn là giá trị cốt lõi của AI agent.

- **Giới hạn một khách hàng** – Agent hiện tại chỉ xử lý được một cuộc trò chuyện tại một thời điểm. Nếu có hai khách hàng truy cập cùng lúc, hội thoại của họ có thể chồng chéo, hoặc tệ hơn, một khách hàng thấy lịch sử trò chuyện của người khác. Không có cơ chế duy trì ngữ cảnh tách biệt cho từng người dùng.

- **Công cụ gắn chặt trong code** – Các công cụ được định nghĩa trực tiếp trong code agent, dẫn đến:

  - Không thể tái sử dụng công cụ cho các agent khác (như agent bán hàng, hỗ trợ kỹ thuật).

  - Cập nhật công cụ bắt buộc phải thay đổi code agent và triển khai lại toàn bộ.

  - Các nhóm khác nhau không thể tự duy trì công cụ một cách độc lập.

- **Thiếu hạ tầng sản xuất** – Agent chỉ chạy cục bộ, không xét đến khả năng mở rộng, bảo mật, giám sát hay độ tin cậy.

Những rào cản kiến trúc này khiến agent không thể triển khai cho khách hàng thực tế. Các nhóm phát triển agent có thể mất hàng tháng để khắc phục, làm chậm giá trị mang lại và tăng chi phí. Đây chính là lúc Amazon Bedrock AgentCore trở nên cần thiết: thay vì tự xây dựng toàn bộ năng lực sản xuất từ đầu, AgentCore cung cấp dịch vụ quản lý sẵn có, giải quyết có hệ thống từng khoảng trống.

---

## The pub/sub hub

Việc sử dụng kiến trúc **hub-and-spoke** (hay message broker) hoạt động tốt với một số lượng nhỏ các microservices liên quan chặt chẽ.  
- Mỗi microservice chỉ phụ thuộc vào *hub*  
- Kết nối giữa các microservice chỉ giới hạn ở nội dung của message được xuất  
- Giảm số lượng synchronous calls vì pub/sub là *push* không đồng bộ một chiều

Nhược điểm: cần **phối hợp và giám sát** để tránh microservice xử lý nhầm message.

---

## Core microservice

Cung cấp dữ liệu nền tảng và lớp truyền thông, gồm:  
- **Amazon S3** bucket cho dữ liệu  
- **Amazon DynamoDB** cho danh mục dữ liệu  
- **AWS Lambda** để ghi message vào data lake và danh mục  
- **Amazon SNS** topic làm *hub*  
- **Amazon S3** bucket cho artifacts như mã Lambda

> Chỉ cho phép truy cập ghi gián tiếp vào data lake qua hàm Lambda → đảm bảo nhất quán.

---

## Front door microservice

- Cung cấp API Gateway để tương tác REST bên ngoài  
- Xác thực & ủy quyền dựa trên **OIDC** thông qua **Amazon Cognito**  
- Cơ chế *deduplication* tự quản lý bằng DynamoDB thay vì SNS FIFO vì:
  1. SNS deduplication TTL chỉ 5 phút
  2. SNS FIFO yêu cầu SQS FIFO
  3. Chủ động báo cho sender biết message là bản sao

---

## Staging ER7 microservice

- Lambda “trigger” đăng ký với pub/sub hub, lọc message theo attribute  
- Step Functions Express Workflow để chuyển ER7 → JSON  
- Hai Lambda:
  1. Sửa format ER7 (newline, carriage return)
  2. Parsing logic  
- Kết quả hoặc lỗi được đẩy lại vào pub/sub hub

---

## Tính năng mới trong giải pháp

### 1. AWS CloudFormation cross-stack references
Ví dụ *outputs* trong core microservice:
```yaml
Outputs:
  Bucket:
    Value: !Ref Bucket
    Export:
      Name: !Sub ${AWS::StackName}-Bucket
  ArtifactBucket:
    Value: !Ref ArtifactBucket
    Export:
      Name: !Sub ${AWS::StackName}-ArtifactBucket
  Topic:
    Value: !Ref Topic
    Export:
      Name: !Sub ${AWS::StackName}-Topic
  Catalog:
    Value: !Ref Catalog
    Export:
      Name: !Sub ${AWS::StackName}-Catalog
  CatalogArn:
    Value: !GetAtt Catalog.Arn
    Export:
      Name: !Sub ${AWS::StackName}-CatalogArn
