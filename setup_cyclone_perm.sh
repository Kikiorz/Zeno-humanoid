#!/bin/bash
# ============================================================
# CycloneDDS 永久配置 — 写入 ~/.bashrc，每次开终端自动生效
# 用法: bash setup_cyclone_perm.sh  或  source setup_cyclone_perm.sh
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

CYCLONEDDS_XML="$HOME/.ros/cyclonedds.xml"
BASHRC="$HOME/.bashrc"

# ---- 检测现有配置 ----
has_config() {
    grep -q '# >>> cyclonedds >>>' "$BASHRC" 2>/dev/null
}

# ---- 删除永久配置 ----
delete_config() {
    echo ""
    log_warn "删除 CycloneDDS 永久配置..."

    # 从 .bashrc 移除
    if has_config; then
        sed -i '/# >>> cyclonedds >>>/,/# <<< cyclonedds <<</d' "$BASHRC" 2>/dev/null
        log_info "已从 $BASHRC 移除配置块"
    else
        log_info "$BASHRC 中无配置块"
    fi

    # 删除 xml 文件
    if [ -f "$CYCLONEDDS_XML" ]; then
        rm -f "$CYCLONEDDS_XML"
        log_info "已删除 $CYCLONEDDS_XML"
    else
        log_info "$CYCLONEDDS_XML 不存在"
    fi

    # 清除当前终端的环境变量
    unset RMW_IMPLEMENTATION CYCLONEDDS_URI ROS_DOMAIN_ID

    echo ""
    log_info "✅ 永久配置已删除 (当前终端变量已清除，新终端不再加载)"
}

# ---- 设置永久配置 ----
setup_config() {
    # 1. 确认 ROS_DOMAIN_ID
    echo ""
    read -p "ROS_DOMAIN_ID [默认=10]: " DOMAIN_ID
    DOMAIN_ID="${DOMAIN_ID:-10}"

    # 2. 列出有线网口，用数字选择
    echo ""
    echo "可用有线网口:"
    WIRED_IFACES=()
    declare -i idx=1
    for iface in $(ls /sys/class/net/); do
        # 跳过 lo，跳过无线网口（有 wireless 目录的是无线）
        if [ "$iface" != "lo" ] && [ ! -d "/sys/class/net/$iface/wireless" ]; then
            WIRED_IFACES+=("$iface")
            echo "  ${idx}) $iface"
            idx+=1
        fi
    done

    if [ ${#WIRED_IFACES[@]} -eq 0 ]; then
        log_error "未找到任何有线网口！"
        return 1
    fi

    read -p "选择网口 [默认=1]: " IFACE_NUM
    IFACE_NUM="${IFACE_NUM:-1}"
    if ! [[ "$IFACE_NUM" =~ ^[0-9]+$ ]] || [ "$IFACE_NUM" -lt 1 ] || [ "$IFACE_NUM" -gt ${#WIRED_IFACES[@]} ]; then
        echo "无效选择，使用默认=1"
        IFACE_NUM=1
    fi
    IFACE="${WIRED_IFACES[$((IFACE_NUM - 1))]}"

    # 3. 写入 cyclonedds.xml
    log_info "写入 $CYCLONEDDS_XML ..."
    mkdir -p "$HOME/.ros"

    cat > "$CYCLONEDDS_XML" << EOF
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config
            https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain Id="${DOMAIN_ID}">
    <General>
      <Interfaces>
        <NetworkInterface name="${IFACE}" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
    <Discovery>
      <SPDPInterval>3s</SPDPInterval>
      <LeaseDuration>15s</LeaseDuration>
    </Discovery>
    <Internal>
      <SocketReceiveBufferSize min="10MB"/>
      <SocketSendBufferSize min="10MB"/>
      <Watermarks>
        <WhcHigh>5MB</WhcHigh>
      </Watermarks>
      <HeartbeatInterval>100ms</HeartbeatInterval>
    </Internal>
  </Domain>
</CycloneDDS>
EOF

    # 4. 写入 ~/.bashrc
    log_info "写入环境变量到 $BASHRC ..."
    sed -i '/# >>> cyclonedds >>>/,/# <<< cyclonedds <<</d' "$BASHRC" 2>/dev/null || true

    cat >> "$BASHRC" << EOF
# >>> cyclonedds >>>
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$CYCLONEDDS_XML
export ROS_DOMAIN_ID=$DOMAIN_ID
# <<< cyclonedds <<<
EOF

    # 5. 设置网络缓冲区大小（提高 CycloneDDS 吞吐量）
    log_info "设置网络缓冲区..."
    sudo sysctl -w net.core.rmem_max=20971520
    sudo sysctl -w net.core.wmem_max=20971520

    # 6. 立即使当前终端生效
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI="file://${CYCLONEDDS_XML}"
    export ROS_DOMAIN_ID="${DOMAIN_ID}"

    echo ""
    log_info "✅ 永久配置完成"
    echo "   RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
    echo "   CYCLONEDDS_URI=$CYCLONEDDS_URI"
    echo "   ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    echo "   网口: $IFACE"
    echo ""
    log_info "新终端自动生效，或手动: source ~/.bashrc"
}

# ========== 主入口 ==========

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    SCRIPT_PATH="$(readlink -f "$0")"
    echo "启动新 shell 并加载配置工具..."
    RC_FILE=$(mktemp /tmp/cyclone_rc_XXXXXX)
    cat ~/.bashrc 2>/dev/null > "$RC_FILE"
    echo "source '$SCRIPT_PATH'" >> "$RC_FILE"
    exec bash --rcfile "$RC_FILE" -i
fi

echo ""
echo "===== CycloneDDS 永久配置 ====="

# 显示现有配置状态
if has_config; then
    echo ""
    log_warn "检测到已有配置:"
    grep -A3 '# >>> cyclonedds >>>' "$BASHRC" 2>/dev/null
fi

# 选择操作
echo ""
echo "  s) 设置/更新配置"
echo "  d) 删除永久配置"
read -p "选择操作 [默认=s]: " ACTION
ACTION="${ACTION:-s}"

case "$ACTION" in
    d|D) delete_config ;;
    s|S) setup_config ;;
    *)  log_error "无效选择，请输入 s(设置) 或 d(删除)"
        return 1
        ;;
esac
