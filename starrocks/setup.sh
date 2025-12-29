#!/bin/bash
# file starrocks/setup.sh
# =================================================================
# RAGHOUSE MIGRATION & SETUP SCRIPT
# Tác dụng:
# 1. Tự động tắt Docker
# 2. Copy dữ liệu từ Docker Volume cũ sang thư mục mới (nếu có)
# 3. Tạo file .env cấu hình động
# 4. Khởi động lại hệ thống
# =================================================================

# --- CẤU HÌNH MẶC ĐỊNH ---
# Đường dẫn đích (Nơi bạn muốn lưu data). Có thể sửa tại đây.
DEFAULT_DATA_ROOT="/drive1/ducanhne/rag_data"
DOCKER_VOL_PATH="/var/lib/docker/volumes"

# Màu mè cho đẹp
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   RAGHOUSE SETUP & MIGRATION WIZARD   ${NC}"
echo -e "${GREEN}=============================================${NC}"

# 1. Hỏi người dùng đường dẫn data (hoặc dùng mặc định)
read -p "Nhập đường dẫn lưu Data (Enter để dùng '$DEFAULT_DATA_ROOT'): " USER_DATA_ROOT
DATA_ROOT=${USER_DATA_ROOT:-$DEFAULT_DATA_ROOT}

# 2. Hỏi API Key (để tạo file .env)
read -p "Nhập Google Gemini API Key (Enter để bỏ qua): " GEMINI_KEY

echo -e "\n${YELLOW}>>> Bước 1: Dừng hệ thống Docker cũ...${NC}"
docker compose down

echo -e "\n${YELLOW}>>> Bước 2: Chuẩn bị thư mục tại: $DATA_ROOT ${NC}"
# Tạo các thư mục rỗng trước
sudo mkdir -p "$DATA_ROOT"/{postgres,etcd,minio,milvus,ollama,uploads,hf_cache,starrocks_fe,starrocks_be}

# 3. Hàm copy thông minh (Chỉ copy nếu tìm thấy dữ liệu cũ)
copy_volume() {
    VOL_NAME=$1
    DEST_DIR=$2
    
    SRC="$DOCKER_VOL_PATH/$VOL_NAME/_data/."
    DEST="$DATA_ROOT/$DEST_DIR/"

    if [ -d "$DOCKER_VOL_PATH/$VOL_NAME/_data" ]; then
        echo -e "   • Tìm thấy dữ liệu cũ của [${GREEN}$VOL_NAME${NC}]. Đang copy sang [$DEST_DIR]..."
        # Dùng rsync hoặc cp -a để giữ nguyên meta data
        sudo cp -a "$SRC" "$DEST"
    else
        echo -e "   • Không thấy dữ liệu cũ của [$VOL_NAME]. Tạo mới thư mục rỗng."
    fi
}

echo -e "\n${YELLOW}>>> Bước 3: Di chuyển dữ liệu (Migration)...${NC}"

# --- LIST COPY ---
# Cú pháp: copy_volume "TÊN_VOLUME_DOCKER_CŨ" "TÊN_THƯ_MỤC_MỚI"

copy_volume "rag_postgres_data" "postgres"
copy_volume "rag_etcd_data" "etcd"
copy_volume "rag_minio_data" "minio"
copy_volume "rag_milvus_data" "milvus"
copy_volume "rag_ollama_data" "ollama"
copy_volume "rag_api_uploads" "uploads"
copy_volume "rag_hf_cache" "hf_cache"

# LƯU Ý: StarRocks không copy cái cũ vì đang lỗi, ta để nó tự tạo mới
echo -e "   • [StarRocks] Bỏ qua dữ liệu cũ để Reset lỗi Disk Limit. Tạo mới sạch sẽ."

echo -e "\n${YELLOW}>>> Bước 4: Phân quyền (Chmod 777)...${NC}"
sudo chmod -R 777 "$DATA_ROOT"
echo "   • Đã cấp quyền ghi cho $DATA_ROOT"

echo -e "\n${YELLOW}>>> Bước 5: Tạo file cấu hình (.env)...${NC}"
# Ghi đè file .env mới
cat <<EOF > .env
# Tự động tạo bởi setup.sh
DATA_ROOT=$DATA_ROOT
GEMINI_API_KEY=$GEMINI_KEY
EOF
echo "   • Đã tạo file .env với DATA_ROOT=$DATA_ROOT"

echo -e "\n${YELLOW}>>> Bước 6: Khởi động hệ thống...${NC}"
# Build lại để đảm bảo config mới được áp dụng
docker compose up -d --build

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   CÀI ĐẶT HOÀN TẤT!   ${NC}"
echo -e "   • Data nằm tại: $DATA_ROOT"
echo -e "   • Kiểm tra Init: docker logs -f rag_starrocks_init"
echo -e "${GREEN}=============================================${NC}"