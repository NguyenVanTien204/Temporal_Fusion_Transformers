---

```markdown
# Báo cáo khám phá dữ liệu (EDA) - sales_train_validation Dataset

## dữ liệu này nói về gì

Đây là dữ liệu bán lẻ theo ngày (daily sales data) của chuỗi siêu thị Walmart ở Mỹ.
Nó ghi lại số lượng sản phẩm được bán mỗi ngày trong hơn 5 năm (2011–2016).

## 1. dữ liệu

Các cột trong tập dữ liệu
id: Mã định danh của sản phẩm, kết hợp item_id, store_id, và hậu tố \_validation.
item_id: Mã sản phẩm.
dept_id: Mã phòng ban (department) chứa sản phẩm.
cat_id: Mã danh mục (category) chứa phòng ban.
store_id: Mã cửa hàng bán sản phẩm.
state_id: Bang/tiểu bang của cửa hàng.

Từ d_1 → d_1913 = doanh số (số lượng bán ra) theo từng ngày.
d_1 = 2011-01-29,
d_1913 = 2016-05-22 (mapping trong file calendar.csv).

Tổng số cột: 1919
Tổng số dòng: 30.490
Cột meta (id, item_id, dept_id, cat_id, store_id, state_id): 6
Cột ngày (d_1 → d_1913): 1913
Không có cột ngày nào sai định dạng (d_x với x là số).
không có tên cột bị trùng lặp khoảng trắng ký tự đặc biệt

## 2. kết quả phân tích và phương pháp

- dữ liệu trùng lặp => kết quả đưa ra không có giá trị trùng lặp
- dữ liệu thiếu => kết quả đưa ra không có giá trị thiếu
- giá trị ngoại lệ: áp dụng phương pháp Modified Z-score (MAD) do Khi áp dụng công thức IQR (ngưỡng 1.5 \* IQR), các điểm tăng vọt này dễ bị coi là ngoại lệ, dù thật ra chúng hợp lý trong ngữ cảnh.
  có 1908 giá trị ngoại lệ trong 30.490 dữ liệu tại cột d_1898
  => Phần lớn dữ liệu của d_1898 ổn định (≈90%), nhưng có khoảng 10% dữ liệu bất thường. Đây là tín hiệu quan trọng để xem xét trước khi đưa dữ liệu vào huấn luyện mô hình dự báo.
- giá trị duy nhất:
  các cột meta
  id → 30,490 giá trị duy nhất (mỗi dòng dữ liệu có một id riêng).
  item_id → 3,049 sản phẩm khác nhau.
  dept_id → chỉ có 7 bộ phận.
  cat_id → chỉ có 3 ngành hàng.
  store_id → 10 cửa hàng.
  state_id → 3 bang/khu vực.
  các cột ngày: Mỗi cột có từ 60 – 100 giá trị duy nhất → nghĩa là doanh số mỗi ngày có khá nhiều mức khác nhau (đa dạng), chứ không chỉ vài giá trị.
```
