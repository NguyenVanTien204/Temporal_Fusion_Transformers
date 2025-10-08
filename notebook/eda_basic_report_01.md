---

```markdown
# Báo cáo khám phá dữ liệu (EDA) - Calendar Dataset

## dữ liệu này nói về gì

đây là dữ liệu ghi lại những sự kiện được ghi lại từ ngày đầu tiên (2011-01-29) cho đến ngày cuối cùng 1931 (2016-05-22).

## 1. dữ liệu

Các cột trong tập dữ liệu:
`date`: ngày/tháng/năm thực tế
`wm_yr_wk`: Mã tuần (week number) trong toàn bộ khoảng thời gian(1 số đầu là nguồn dữ liệu 2 số sau tương ứng năm, 2 số sau là số tuần của năm đó)
`weekday`: Tên thứ trong tuần (Saturday, Sunday, … Friday)
`wday`: Mã số của thứ trong tuần (1–7)
`month`: Tháng trong năm (1–12)
`year`: Năm của ngày (2011 → 2016)
`d`: Mã định danh ngày, từ d_1 đến d_1913
`event_name_1`: tên sự kiện 1
`event_name_2`: tên sự kiện 2
`event_type_1`: loại sự kiện 1
`event_type_2`: loại sự kiện 2
`snap_CA`: Biến chỉ báo (binary: 0/1), 1 = có chính sách SNAP (Supplemental Nutrition Assistance Program) áp dụng tại California ngày đó
`snap_TX`: Tương tự cho bang Texas
`snap_WI`: Tương tự cho bang Wisconsin

## 2. phương pháp kết quả phân tích từng cột

**d**
kiểm tra sự liên tục bằng phép lặp chạy liên tục xem d_1 có tăng thêm 1 đơn vị sau mỗi ngày không

- chạy liên tục từ d_1 đến d_1969 -> dữ liệu hợp lệ
  **date**
  kiểm tra xem có đủ các ngày trong khoảng thời gian từ 29-1-2011 đến 19-6-2016 chạy liên tục không
- Ngày nhỏ nhất: 2011-01-29 00:00:00
- Ngày lớn nhất: 2016-06-19 00:00:00
- Tổng số ngày trong khoảng: 1969
- Số ngày có trong dữ liệu: 1969
- Số ngày bị thiếu: 0
  **wm_yr_wk**
  kiểm tra xem 1 năm có đủ 52 tuần (trừ năm 2013 là 53) không
- Boxplot trải đều, không có outlier rõ rệt
- Phân bố tuần trong năm đồng đều
  **wday**
  kiểm tra xem có chạy theo chu kỳ từ 1-7 lặp lại liên tục không
- Giá trị nằm trong khoảng 1–7, đúng với ngày trong tuần
- Không có outlier → dữ liệu hợp lệ
  **mont**
  kiểm tra xem dữ liệu có trùng khớp với dữ liệu tháng của cột date ko
- Giá trị từ 1–12, đúng quy luật tháng
- Không có outlier.
  **year**
  kiểm tra dữ liệu có trùng khớp với dữ liệu năm của cột date không
- Trải từ 2011–2016, đúng khoảng thời gian dataset Walmart
- Không có năm ngoài phạm vi này
  **snap_CA, snap_TX, snap_WI (binary: 0/1)**
  kiểm tra có giá trị nào khác 0 hoặc 1 không
- Boxplot cho thấy phân bố nhị phân (đa số giá trị = 0, một số = 1)
- Không có outlier (vì chỉ nhận 0 hoặc 1)
- Có thể phân tích thêm tỷ lệ ngày có SNAP event theo từng bang

## 3. Các phát hiện chính

Danh sách cột: ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']

- Không có cột trùng lặp.
- Không có cột chứa ký tự đặc biệt hoặc khoảng trắng.
- không có mismatch của 2 cặp event_name với event type 1 và 2
- cột wm_yr_wk được tính theo lịch walmart retail calendar
- các dữ liệu trong bộ dữ liệu đa số đã đưa về dạng chuẩn, cột date nên ở dạng datetime
- không phát hiện dữ liệu trùng lặp, ngoại lai, bất thường
- các giá trị thiếu
  `event_name_1` thiếu 1807 giá trị (91.77%)
  `event_name_2` thiếu 1964 giá trị (99.74%)
  `event_type_1` thiếu 1807 giá trị (91.77%)
  `event_type_2` thiếu 1964 giá trị (99.74%)
  => hầu hết là những ngày không có sự kiện được tổ chức
  => các cột khác ngoài 4 cột trên không có giá trị thiếu
- không có giá trị ngoại lệ
```
