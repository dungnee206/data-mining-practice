scaler.pkl (Bộ chuẩn hóa dữ liệu):
  Công dụng: Lưu lại các giá trị như thoi_gian_gan_bo, cuoc_phi_hang_thang, tong_cuoc_phi đã học được trong quá trình huấn luyện
  Tại sao cần: Khi người dùng nhập một khách hàng mới trên Web (ví dụ: cước 75$), dữ liệu thô này bắt buộc phải đi qua các công thức đã lưu trong scaler.pkl 
    để đưa về đúng thang đo chuẩn trước khi chuyển cho mô hình.
feature_names.pkl (Danh sách cột đặc trưng):
  Công dụng: Lưu chính xác thứ tự, số lượng và tên của toàn bộ các cột sau khi One-Hot Encoding (khoảng 13 đặc trưng).
  Tại sao cần: Tránh lỗi lệch kích thước hoặc lệch vị trí đặc trưng (Feature Mismatch). Ứng dụng Web sẽ dựa vào tệp này để tạo ra 
    một vector đầu vào đúng định dạng trước khi đưa vào bộ chuẩn hóa.
best_model.pkl (Mô hình gom cụm tối ưu):
  Công dụng: Lưu lại tọa độ các tâm cụm (Centroids) và trọng số của thuật toán được chọn (K-Means).
  Tại sao cần: Khi nhận được vector đặc trưng đã chuẩn hóa của khách hàng mới, tệp này thực thi hàm .predict() 
    để xác định ngay khách hàng thuộc cụm nào (0, 1, 2 hay 3) trong vài mili-giây.

telco_cleaned.csv (Dữ liệu đã làm sạch cơ sở): đã làm sạch dữ liệu, người dùng như giáo viên đọc trực tiếp file data đã được làm sạch
telco_scaled.csv (Ma trận dữ liệu đã mã hóa và chuẩn hóa):Dành riêng cho file Notebook (clustering_comparison.ipynb) nạp thẳng vào 4 thuật toán gom cụm 
(K-Means, GMM, Hierarchical, DBSCAN)  để tính toán khoảng cách hình học.
