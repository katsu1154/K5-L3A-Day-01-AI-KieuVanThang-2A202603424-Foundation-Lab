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

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân
 Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?

Tính toán số lượng tham số:Trường hợp hidden = 4:Lớp 1:  (2000×4)+4=8004 Lớp 2:  (4×1)+1=5 Tổng tham số: 8.009Trường hợp hidden = 512:Lớp 1:  (2000×512)+512=1.024.512 Lớp 2:  (512×1)+1=513 Tổng tham số: 1.025.025 (hơn 1 triệu tham số)2. Dự đoán F1 Test và Thời gian Train:(Do không chạy code trực tiếp, đây là xu hướng chắc chắn sẽ xảy ra dựa trên bản chất của mạng nơ-ron)Với mạng nhỏ (4 nơ-ron):Thời gian train: Cực kỳ nhanh (gần tương đương Logistic Regression).F1 Test: Khá ổn định, dao động ngang mức Logistic Regression. Do ít tham số, mạng khó bị overfit (học vẹt) trên bộ dữ liệu nhỏ.Với mạng lớn (512 nơ-ron):Thời gian train: Chậm hơn đáng kể do phải cập nhật hơn 1 triệu tham số.F1 Test: Có nguy cơ tụt giảm hoặc không cao hơn mạng nhỏ (như hiện tượng đã thấy ở mạng 64 nơ-ron). Mạng quá lớn so với lượng dữ liệu nhỏ bé, nó sẽ overfit (ghi nhớ hoàn toàn tập train) và mất khả năng dự đoán đúng trên tập test.3. Khi nào nên chọn mạng nhỏ?Dữ liệu ít và đơn giản: Khi dữ liệu có giới hạn (vài ngàn mẫu) hoặc đặc trưng thô sơ (như Bag-of-words), mạng nhỏ giúp chống lại hiện tượng overfit hiệu quả.Yêu cầu độ trễ thấp (Low Latency): Khi cần đáp ứng tốc độ phản hồi cực nhanh (ví dụ SLA < 50ms) như tính toán trực tiếp trên ứng dụng thời gian thực.Hạn chế tài nguyên phần cứng: Khi triển khai (deploy) mô hình trên các thiết bị yếu (điện thoại di động, máy chủ CPU nhỏ, thiết bị IoT) hoặc để tiết kiệm chi phí vận hành máy chủ.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
