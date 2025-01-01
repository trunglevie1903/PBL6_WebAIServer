# Ghi chép: Khác biệt giữa ROUGE-Score và BERT-Score trong đánh giá tóm tắt

### Tóm tắt về các phương pháp đánh giá

#### 1. ROUGE Score

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) là một bộ chỉ số thường được sử dụng để đánh giá việc tóm tắt tự động và dịch máy bằng cách so sánh tóm tắt sinh ra với các tóm tắt tham chiếu.

- Trọng tâm: Đo **mức độ trùng lặp** giữa văn bản sinh ra và văn bản tham chiếu

- Điểm mạnh: Dễ tính toán, không phụ thuộc vào ngôn ngữ, dễ hiểu

- Hạn chế: Phụ thuộc vào việc khớp từ chính xác và không bắt ý ngữ nghĩa

#### 2. BERT Score

BERT-Score là một chỉ số mới hơn, sử dụng các biểu diễn nhúng từ mô hình BERT được huấn luyện trước để đánh giá mức độ tương đồng giữa tóm tắt tham chiếu và tóm tắt sinh ra dựa trên nội dung ngữ nghĩa của chúng.

- Trọng tâm: Đo **mức độ tương đồng ngữ nghĩa** thay vì mức độ trùng lặp.

- Điểm mạnh: Bắt được các sắc thái về ý nghĩa và ngữ cảnh, xử lý tốt việc diễn giải lại câu.

- Hạn chế: Tốn nhiều tài nguyên tính toán hơn và phụ thuộc vào các mô hình ngôn ngữ cụ thể.

### Hiệu suất của các mô hình tóm tắt

| Mô hình | ROUGE-1 | BERT-Score (F1)|
|-|-|-|
|BART|45.6|88.2%|
|T5|44.3|87.9%|
|DistilBART (pretrained)|42.3|85.5%|
|DistilBART (tự train)|40.2|84.1%|
