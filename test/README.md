# So sánh Models: Có Delta vs Không Delta

Thư mục này chứa các script để so sánh hiệu suất của 2 models Change3D:
- **Model không có delta**: `runs\change3d\20251031_224419_nodenta\best_change3d_dir.pth`
- **Model có delta**: `runs\change3d\20251031_230655\best_change3d_dir.pth`

## Cấu trúc

- `compare_models_with_without_delta.py`: Script chính để so sánh 2 models
- `run_comparison.py`: Script wrapper để chạy nhanh
- `README.md`: File này

## Cách sử dụng

### Chạy với tham số mặc định

```bash
python test/compare_models_with_without_delta.py
```

### Chạy với tham số tùy chỉnh

**PowerShell:**
```powershell
python test/compare_models_with_without_delta.py --ckpt_no_delta runs\change3d\20251031_224419_nodenta\best_change3d_dir.pth --ckpt_with_delta runs\change3d\20251031_230655\best_change3d_dir.pth --test_dir outputs\sum\dataset_new\val --batch_size 128 --workers 0 --output_dir test
```

**Bash/Linux:**
```bash
python test/compare_models_with_without_delta.py \
    --ckpt_no_delta runs/change3d/20251031_224419_nodenta/best_change3d_dir.pth \
    --ckpt_with_delta runs/change3d/20251031_230655/best_change3d_dir.pth \
    --test_dir outputs/sum/dataset_new/val \
    --batch_size 128 \
    --workers 0 \
    --output_dir test
```

## Kết quả

Sau khi chạy, script sẽ tạo các file kết quả trong thư mục `test/`:

1. **`comparison.csv`**: Bảng so sánh dạng CSV
2. **`comparison.md`**: Bảng so sánh dạng Markdown (dễ đọc, phù hợp cho báo cáo)
3. **`comparison.json`**: Kết quả chi tiết dạng JSON (để xử lý tự động)

## Các metrics được tính

- **Accuracy**: Độ chính xác tổng thể
- **Macro F1**: F1-score trung bình macro (cho tất cả classes)
- **Per-class F1**: F1-score cho từng class
- **Per-class Precision**: Precision cho từng class
- **Per-class Recall**: Recall cho từng class
- **ROC-AUC**: Area Under ROC Curve (nếu có thể tính)
- **False Alarm Rate (FAR)**: Tỷ lệ báo động sai
- **Confusion Matrix**: Ma trận nhầm lẫn

## Lưu ý

- Script sẽ tự động phát hiện xem model có sử dụng delta channel hay không từ metadata trong checkpoint
- Test set mặc định là `outputs\sum\dataset_new\val`
- Nếu không có GPU, script sẽ tự động chuyển sang CPU

