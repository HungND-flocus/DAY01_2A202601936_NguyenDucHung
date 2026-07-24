# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.8: Ở 0.0 và 0.7, câu trả lời rất chuẩn xác, tập trung đúng vào các danh thắng lịch sử của Hà Nội. Đến 1.2 bắt đầu thấy từ ngữ bay bổng và văn phong đa dạng hơn. Tuy nhiên khi lên tới 1.8 thì mô hình bắt đầu bị kém mạch lạc rõ rệt, xuất hiện các từ ngữ ngẫu nhiên, ảo giác và lặp từ không đúng ngữ pháp.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn thảo hợp đồng pháp lý, em sẽ đặt temperature = 0.0 để đảm bảo câu từ chính xác tuyệt đối, nhất quán và không sinh ra ảo giác. Còn với trợ lý viết slogan quảng cáo, em sẽ chỉnh temperature khoảng 0.8 đến 1.0 để AI tự do sáng tạo, gợi ý nhiều ý tưởng mới lạ và bắt mắt.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Khối lượng token output mỗi ngày: 20.000 user × 2 lần × 500 token = 20.000.000 token (20.000K).
> - Chi phí với GPT-4o ($0.010 / 1K): 20.000 × $0.010 = $200/ngày.
> - Chi phí với GPT-4o-mini ($0.0006 / 1K): 20.000 × $0.0006 = $12/ngày (tiết kiệm hơn hẳn 16.6 lần).
> - Dùng model lớn (GPT-4o) khi: Cần phân tích các hợp đồng pháp lý phức tạp hay bài toán tư vấn y tế/tài chính đòi hỏi độ chính xác cao.
> - Dùng model nhỏ (GPT-4o-mini) khi: Làm chatbot CSKH tự động trả lời FAQ đơn giản hoặc phân loại sentiment/cảm xúc tin nhắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi của nhà thơ dùng nhiều hình ảnh ẩn dụ (so sánh ML như đứa trẻ tập đi và học từ lỗi sai), giọng văn nhẹ nhàng và né tránh thuật ngữ. Trong khi kỹ sư senior trả lời thẳng vào vấn đề, phân loại Supervised/Unsupervised Learning và đưa ra ví dụ code Python. Qua đó em thấy System Prompt giúp điều khiển: giọng văn/phong cách, đối tượng độc giả, độ sâu kỹ thuật và cấu trúc trình bày của phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Thử đếm một đoạn văn tiếng Việt 150 từ, tiktoken đếm ra tầm 235 token, trong khi công thức ước lượng thô 150 / 0.75 ra 200 token (chênh lệch khoảng 17%). Nếu dùng công thức ước lượng thô cho ứng dụng tiếng Việt thì em sẽ bị DỰ TOÁN THIẾU ngân sách. Lý do là tokenizer của OpenAI thiết kế tối ưu cho tiếng Anh, khi gặp tiếng Việt sẽ tách 1 từ thành nhiều sub-word token hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản là ứng dụng hưởng lợi NHẤT từ streaming vì người dùng thấy chữ gõ ra tức thì, không phải ngồi nhìn màn hình trống chờ vài giây. Trợ lý giọng nói đọc to chỉ cần streaming theo từng câu ngắn để đọc thành tiếng. Còn pipeline dịch tài liệu chạy ngầm ban đêm thì KHÔNG CẦN streaming vì đây là batch job chạy ngầm, không có ai ngồi chờ giao diện hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp dãn thời gian giữa các lần retry (0.1s -> 0.2s -> 0.4s...), tránh cho hàng nghìn client dồn dập gửi request làm sập luôn máy chủ đang quá tải. Kỹ thuật "jitter" thêm chút độ trễ ngẫu nhiên để các client không bị gửi retry trùng khít vào cùng một thời điểm, giúp phân tán tải mượt mà hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt của em: "Bạn là trợ giảng thân thiện của khóa AI, luôn giải thích ngắn gọn dưới 3 câu và bắt đầu câu trả lời bằng 'Chào bạn!'."
> - Chỗ 1: "luôn giải thích ngắn gọn dưới 3 câu" — Xóa đi thì AI sẽ trả lời rất dài và lan man nhiều đoạn.
> - Chỗ 2: "bắt đầu câu trả lời bằng 'Chào bạn!'" — Xóa đi thì AI sẽ đi thẳng vào nội dung mà không chào hỏi thân thiện.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: Ở lượt 1 em giới thiệu "Tôi tên là An, đang làm bài tập Python". Đến lượt 7 (sau 5 câu hội thoại khác), em hỏi "Tên tôi là gì và tôi đang làm bài gì?", AI sẽ bó tay vì thông tin ở lượt 1 đã bị cắt khỏi 4 lượt gần nhất (8 messages).
> Cách khắc phục: Dùng kỹ thuật Tóm tắt ngữ cảnh (Conversation Summarization) — duy trì một bộ nhớ chứa tóm tắt thông tin quan trọng (tên, ngữ cảnh) và đính kèm vào System Prompt ở mỗi lượt gọi.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
