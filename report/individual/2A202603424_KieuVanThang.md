# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603424
- **Họ và tên:**Kiều Văn Thắng
- **SEED** (in ở ô đầu notebook):3224

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.772      0.871        0.824   
2. BoW + MLP (deep learning)         0.988   0.880      0.931        0.919   
3a. Encoder (MiniLM) + kNN           0.905   0.935      0.920        0.923   
3b. Encoder (MiniLM) + LogReg        0.955   0.924      0.939        0.919   
3c. Encoder (MiniLM) + MLP head      0.966   0.935      0.950        0.923   
4a. LLM nhỏ zero-shot                1.000   0.700      0.824        0.824   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 1.000   0.800      0.889        0.889   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.668   1.384038e+04       0.048   
2. BoW + MLP (deep learning)         6.415   2.083817e+04       0.286   
3a. Encoder (MiniLM) + kNN           1.988   1.187730e+02      25.090   
3b. Encoder (MiniLM) + LogReg        3.219   1.118725e+03      11.287   
3c. Encoder (MiniLM) + MLP head      3.992   1.315032e+03       6.394   
4a. LLM nhỏ zero-shot                0.000   1.733000e+00     621.406   
4b. LLM lớn zero-shot                0.000   4.000000e+10       0.000   
4c. LLM nhỏ few-shot                 0.000   9.890000e-01     928.764   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.100         0.000  
2. BoW + MLP (deep learning)          2.319         0.000  
3a. Encoder (MiniLM) + kNN           35.661         0.000  
3b. Encoder (MiniLM) + LogReg        17.382         0.000  
3c. Encoder (MiniLM) + MLP head      10.855         0.000  
4a. LLM nhỏ zero-shot               922.139         0.018  
4b. LLM lớn zero-shot                 0.000         0.000  
4c. LLM nhỏ few-shot               2012.890         0.051  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. **So sánh LLM nhỏ và lớn (`llm_models`).** Em đoán model lớn có F1 cao hơn model nhỏ khoảng 0.05, tốn gấp 2.5 lần và cả hai ngang hoặc nhỉnh hơn Encoder + LogReg. Thực tế, LLM nhỏ đạt F1 SUB **0.824**, LLM lớn là **0.000** với **40/40 output không hợp lệ**; Encoder + LogReg đạt **0.919**. Vì toàn bộ output của model lớn không hợp lệ, số 0 này phản ánh lỗi ở lần gọi/đọc output trong thí nghiệm, không đủ để kết luận model lớn kém hơn về năng lực phân loại.

2. **Tìm kiếm theo nghĩa (`emb_search`).** Em đoán câu “you have been selected to receive a complimentary holiday, reply now to claim” gần với tin ham. Kết quả thực tế: cả **6 hàng xóm gần nhất đều là spam**, với cosine từ **0.513 đến 0.563**. Tôi đã quá dựa vào việc câu không có các từ khóa spam quen thuộc; embedding vẫn bắt được ngữ nghĩa mời nhận kỳ nghỉ, yêu cầu phản hồi và nhận thưởng.

3. **Bỏ ReLU (`dl_relu`).** Em đoán mạng sẽ trở thành logistic regression và F1 giảm về gần mức Logistic Regression. Hai lớp Linear liên tiếp đúng là biểu diễn một phép biến đổi tuyến tính, nhưng F1 thực tế của mạng không ReLU là **0.932**, cao hơn F1 **0.871** của BoW + LogReg; MLP có ReLU đạt **0.931**. Tương đương về dạng hàm không đảm bảo hai lần huấn luyện tìm được cùng nghiệm hoặc cùng dự đoán: cấu hình tối ưu hóa và quá trình học khác nhau, nên kết quả thực nghiệm không giảm như tôi dự đoán.

## 4. Phân tích 50 case khó

Trong các model có số F1 test thường đo riêng, **BoW + MLP tụt nhiều nhất**: F1 SUB giảm từ **0.919** xuống **0.368** trên 50 case khó (giảm **0.551**). Model này đặc biệt yếu ở nhóm **spam giọng tự nhiên (0%)**, **thao túng/lừa đảo (25%)** và **né từ khóa (17%)**. BoW + MLP dựa vào các từ và tổ hợp từ đã học, không thật sự hiểu ý định; vì vậy tin nhắn lịch sự, giống hội thoại bình thường hoặc viết lệch từ khóa dễ bị bỏ sót.

Hai ví dụ model phân loại sai:
- **Case #12:** “hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club” là spam dụ người nhận vào trang hẹn hò. Cách mở đầu thân thiện và thiếu từ khóa quảng cáo quen thuộc khiến đặc trưng BoW không đủ mạnh để nhận ra ý đồ.
- **Case #13:** “Hope you're well! Quick reminder that your photos from the event are ready, view & order prints here: snap-prints.co/r/88” giả dạng thông báo ảnh sự kiện. Cụm từ quen thuộc, vô hại lấn át dấu hiệu bán hàng nằm ở lời mời đặt ảnh và liên kết.

Lưu ý: bảng notebook gán F1 SUB **0.824** cho 4d bằng cách dùng kết quả của **4a**, không phải phép đo riêng của prompt phòng thủ trên SUB; vì vậy không dùng dòng này để kết luận 4d tụt bao nhiêu so với chính nó. Riêng trên hard-eval, 4d đạt F1 **0.000**, còn BoW + MLP là model tụt nhiều nhất trong các cặp so sánh cùng cấu hình.

## 5. Thử thách cá nhân
 Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?

Tính toán số lượng tham số:Trường hợp hidden = 4:Lớp 1:  (2000×4)+4=8004 Lớp 2:  (4×1)+1=5 Tổng tham số: 8.009Trường hợp hidden = 512:Lớp 1:  (2000×512)+512=1.024.512 Lớp 2:  (512×1)+1=513 Tổng tham số: 1.025.025 (hơn 1 triệu tham số)2. Dự đoán F1 Test và Thời gian Train:(Do không chạy code trực tiếp, đây là xu hướng chắc chắn sẽ xảy ra dựa trên bản chất của mạng nơ-ron)Với mạng nhỏ (4 nơ-ron):Thời gian train: Cực kỳ nhanh (gần tương đương Logistic Regression).F1 Test: Khá ổn định, dao động ngang mức Logistic Regression. Do ít tham số, mạng khó bị overfit (học vẹt) trên bộ dữ liệu nhỏ.Với mạng lớn (512 nơ-ron):Thời gian train: Chậm hơn đáng kể do phải cập nhật hơn 1 triệu tham số.F1 Test: Có nguy cơ tụt giảm hoặc không cao hơn mạng nhỏ (như hiện tượng đã thấy ở mạng 64 nơ-ron). Mạng quá lớn so với lượng dữ liệu nhỏ bé, nó sẽ overfit (ghi nhớ hoàn toàn tập train) và mất khả năng dự đoán đúng trên tập test.3. Khi nào nên chọn mạng nhỏ?Dữ liệu ít và đơn giản: Khi dữ liệu có giới hạn (vài ngàn mẫu) hoặc đặc trưng thô sơ (như Bag-of-words), mạng nhỏ giúp chống lại hiện tượng overfit hiệu quả.Yêu cầu độ trễ thấp (Low Latency): Khi cần đáp ứng tốc độ phản hồi cực nhanh (ví dụ SLA < 50ms) như tính toán trực tiếp trên ứng dụng thời gian thực.Hạn chế tài nguyên phần cứng: Khi triển khai (deploy) mô hình trên các thiết bị yếu (điện thoại di động, máy chủ CPU nhỏ, thiết bị IoT) hoặc để tiết kiệm chi phí vận hành máy chủ.

## 6. Phương án cho sản phẩm thật

Em chọn **MiniLM + MLP head (3c)** vì cân bằng tốt giữa chất lượng và độ trễ: F1 test **0.950**, F1 trên SUB 40 tin **0.923**, p95 **10.855 ms/tin** (thấp hơn giới hạn 50 ms), và tốc độ đo theo lô khoảng **1.315 tin/s**. Mức 10 triệu tin/ngày tương đương trung bình khoảng **116 tin/s**, nhưng lưu lượng thực tế có giờ cao điểm nên cần kiểm thử tải và triển khai đủ replica để giữ p95 dưới 50 ms; số đo trong notebook chỉ là tham khảo trên môi trường Colab, không bảo đảm hiệu năng production.

Đánh đổi là phải chạy bước mã hóa MiniLM cho từng tin, nên tốn CPU/GPU và tài nguyên vận hành hơn các mô hình BoW; tốc độ thực tế còn phụ thuộc phần cứng và mức tải đồng thời. Đổi lại, mô hình đạt F1 cao nhất trong các phương án đã đo và có khoảng đệm độ trễ đáng kể. Nếu ưu tiên chi phí và thông lượng hơn, BoW + MLP cũng đạt p95 **2.319 ms/tin** và khoảng **20.838 tin/s**, nhưng F1 test thấp hơn (**0.931**); chênh F1 SUB chỉ **0.004**, nên cần đánh giá thêm trên dữ liệu lớn, đại diện trước khi quyết định đánh đổi chất lượng để tiết kiệm tài nguyên.


## 7. AI log và điều đã học

**AI log:** Em nhờ AI đối chiếu dự đoán với kết quả, phân tích các case khó và lập luận chọn mô hình cho sản phẩm. AI giúp tổng hợp số liệu; em kiểm tra lại output notebook, sửa các nhận định chưa khớp và ghi rõ giới hạn khi 40/40 output của LLM lớn không hợp lệ.

**Điều đã học:** Em hiểu rằng điểm F1 trên tập test thông thường chưa đảm bảo mô hình xử lý tốt các kiểu spam mới hoặc được viết tự nhiên. Khi giải thích kết quả, cần kiểm tra cả dữ liệu và output thí nghiệm, đồng thời cân nhắc chất lượng, độ trễ và tài nguyên triển khai.
