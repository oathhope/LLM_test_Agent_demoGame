"""
MsPacman 状态提取器
从 gymnasium ALE/MsPacman-v5 的 RGB 帧中提取结构化状态
"""
import numpy as np
from typing import Optional


# 颜色定义（RGB）
PACMAN_COLOR   = (210, 164, 74)
GHOST_COLORS   = {
    "Blinky": (200, 72,  72),   # 红鬼
    "Pinky":  (198, 89,  179),  # 粉鬼
    "Inky":   (84,  184, 153),  # 蓝鬼
    "Clyde":  (198, 108, 58),   # 橙鬼
}
FRIGHTENED_COLOR = (66, 72, 200)  # 鬼魂受惊变蓝
DOT_COLOR      = (228, 111, 111)
WALL_COLOR     = (0, 0, 255)      # 墙壁蓝色 (近似值)
BG_COLOR       = (0, 0, 0)        # 背景黑色

# 游戏区域（像素）
GAME_X1, GAME_Y1 = 0,   0
GAME_X2, GAME_Y2 = 160, 210

COLOR_TOL = 20  # 颜色容差


def _find_color_centroid(frame: np.ndarray, color: tuple, tol: int = COLOR_TOL):
    """找指定颜色的像素质心，返回 (x, y) 或 (None, None)"""
    r, g, b = color
    mask = (
        (np.abs(frame[:, :, 0].astype(int) - r) < tol) &
        (np.abs(frame[:, :, 1].astype(int) - g) < tol) &
        (np.abs(frame[:, :, 2].astype(int) - b) < tol)
    )
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    return int(xs.mean()), int(ys.mean())


def _count_color_pixels(frame: np.ndarray, color: tuple, tol: int = COLOR_TOL) -> int:
    """统计指定颜色像素数量"""
    r, g, b = color
    mask = (
        (np.abs(frame[:, :, 0].astype(int) - r) < tol) &
        (np.abs(frame[:, :, 1].astype(int) - g) < tol) &
        (np.abs(frame[:, :, 2].astype(int) - b) < tol)
    )
    return int(mask.sum())


def extract_state(frame: np.ndarray, prev_score: int = 0, lives: int = 3,
                  ale=None) -> dict:
    """
    从 RGB 帧提取结构化游戏状态

    Args:
        frame: RGB帧 (H, W, 3)
        prev_score: 当前累计分数
        lives: 当前生命数
        ale: 可选，ALE对象（用于精确的方向通行性检测）

    Returns:
        dict with keys:
          pacman_x, pacman_y: 玩家像素坐标
          ghosts: [{name, x, y, frightened, distance}]
          dots_remaining: 估算剩余豆子像素数
          score: 分数（从ALE info获取）
          lives: 生命数
          nearest_ghost: 最近鬼魂信息
          safe_directions: 建议安全方向（基于鬼魂位置）
    """
    # 玩家坐标
    px, py = _find_color_centroid(frame, PACMAN_COLOR)

    # 鬼魂坐标
    ghosts = []
    for name, color in GHOST_COLORS.items():
        gx, gy = _find_color_centroid(frame, color)
        frightened_px = _count_color_pixels(frame, FRIGHTENED_COLOR)
        frightened = frightened_px > 50  # 有大量蓝色=受惊状态

        if gx is not None and px is not None:
            dist = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            ghosts.append({
                "name": name,
                "x": gx,
                "y": gy,
                "frightened": frightened,
                "distance": round(dist, 1),
                "direction_from_pacman": _direction_hint(gx - px, gy - py),
            })
        elif gx is not None:
            ghosts.append({
                "name": name,
                "x": gx,
                "y": gy,
                "frightened": frightened,
                "distance": 999,
                "direction_from_pacman": "unknown",
            })

    ghosts.sort(key=lambda g: g["distance"])

    # 豆子数量（像素面积估算）
    dots_pixels = _count_color_pixels(frame, DOT_COLOR)

    # 最近鬼魂
    nearest_ghost = ghosts[0] if ghosts else None
    dangerous_ghost = next(
        (g for g in ghosts if not g["frightened"] and g["distance"] < 50), None
    )

    # 可通行方向（排除墙壁）
    passable_directions = _get_passable_directions(frame, px, py, ale)

    # 每方向深度扫描：通道长度 + 豆子数
    direction_info = _scan_direction_info(frame, px, py, passable_directions)

    # 建议安全方向（在可通行方向中远离鬼魂）
    safe_directions = _calc_safe_directions(px, py, ghosts, passable_directions)

    return {
        "pacman_x": px,
        "pacman_y": py,
        "ghosts": ghosts,
        "dots_pixels": dots_pixels,
        "score": prev_score,
        "lives": lives,
        "nearest_ghost": nearest_ghost,
        "dangerous_ghost": dangerous_ghost,
        "passable_directions": passable_directions,
        "direction_info": direction_info,
        "safe_directions": safe_directions,
    }


def _is_wall_pixel(frame: np.ndarray, x: int, y: int) -> bool:
    """判断某像素是否是墙壁（蓝色系）"""
    if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
        return True  # 越界视为墙
    r, g, b = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])
    # 墙壁是蓝色：b 明显大于 r 和 g
    return b > 100 and b > r * 2 and b > g * 2


def _is_path_pixel(frame: np.ndarray, x: int, y: int) -> bool:
    """判断某像素是否是可通行区域（黑色背景、豆子、Pacman 等非墙像素）"""
    if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
        return False
    return not _is_wall_pixel(frame, x, y)


def _get_passable_directions(frame: np.ndarray, px: int, py: int,
                             ale=None) -> list[str]:
    """
    检测 Pacman 四个方向是否有通路。
    策略：沿方向前进，在每一步检查正交方向 ±4px 的截面。
    只要截面上存在非墙像素就算该步"可通行"。
    近距离（4~14px）至少 50% 的步可通行 → 判为可通行。
    """
    if px is None or py is None:
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    passable = []
    # 正交方向偏移量
    cross_offsets = list(range(-4, 5))  # -4 到 +4

    for name, dx, dy in [("UP", 0, -1), ("DOWN", 0, 1), ("LEFT", -1, 0), ("RIGHT", 1, 0)]:
        path_steps = 0
        total_steps = 0
        # 只看近距离（4~14px），这是 Pacman 即将进入的区域
        for dist in range(4, 16):
            cx, cy = px + dx * dist, py + dy * dist
            # 沿正交方向扫描截面
            has_path = False
            for off in cross_offsets:
                # 正交方向：如果前进方向是垂直(dx=0)，正交为水平(off在x)
                sx = cx + dy * off   # dy!=0 时 off 在 x 方向
                sy = cy + dx * off   # dx!=0 时 off 在 y 方向
                if _is_path_pixel(frame, sx, sy):
                    has_path = True
                    break
            if has_path:
                path_steps += 1
            total_steps += 1

        if total_steps > 0 and path_steps / total_steps >= 0.5:
            passable.append(name)

    return passable if passable else ["NOOP"]


def _scan_direction_info(frame: np.ndarray, px: int, py: int,
                        passable: list[str]) -> dict:
    """
    对每个方向做深度扫描：通道距离(到第一面墙的像素距离)和沿途豆子像素数。
    返回 {"UP": {"corridor": 25, "dots": 8, "passable": True}, ...}
    """
    if px is None or py is None:
        return {}

    dir_vectors = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    result = {}
    dot_r, dot_g, dot_b = DOT_COLOR

    for name, (ddx, ddy) in dir_vectors.items():
        corridor_len = 0
        dot_count = 0
        wall_streak = 0  # 连续墙壁像素计数，容忍短墙（sprite 边缘）
        # 从 pacman 中心往外扫描，步进 1px，用宽截面判断是否撞墙
        for dist in range(4, 60):
            cx, cy = px + ddx * dist, py + ddy * dist
            if cx < 0 or cx >= frame.shape[1] or cy < 0 or cy >= frame.shape[0]:
                break
            # 宽截面检测：正交方向 ±4px，只要有一个非墙像素就算通路
            has_path = False
            for off in range(-4, 5):
                sx = cx + ddy * off
                sy = cy + ddx * off
                if _is_path_pixel(frame, sx, sy):
                    has_path = True
                    break
            if not has_path:
                wall_streak += 1
                if wall_streak >= 3:  # 连续 3px 墙壁才认为通道结束
                    break
                continue
            wall_streak = 0
            corridor_len = dist
            # 检查该像素及其正交方向 ±2px 有无豆子颜色
            for off in range(-2, 3):
                sx = cx + (ddy * off)  # 正交偏移
                sy = cy + (ddx * off)
                if 0 <= sx < frame.shape[1] and 0 <= sy < frame.shape[0]:
                    r, g, b = int(frame[sy, sx, 0]), int(frame[sy, sx, 1]), int(frame[sy, sx, 2])
                    if (abs(r - dot_r) < COLOR_TOL and abs(g - dot_g) < COLOR_TOL
                            and abs(b - dot_b) < COLOR_TOL):
                        dot_count += 1
                        break  # 一步中只计一次

        result[name] = {
            "corridor": corridor_len,
            "dots": dot_count,
            "passable": name in passable,
        }
    return result


def _find_centroid_from_frame(frame: np.ndarray, color: tuple) -> tuple:
    """从帧中找颜色质心"""
    r, g, b = color
    tol = COLOR_TOL
    mask = (
        (np.abs(frame[:, :, 0].astype(int) - r) < tol) &
        (np.abs(frame[:, :, 1].astype(int) - g) < tol) &
        (np.abs(frame[:, :, 2].astype(int) - b) < tol)
    )
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    return int(xs.mean()), int(ys.mean())


def _direction_hint(dx: int, dy: int) -> str:
    parts = []
    if abs(dy) > abs(dx) * 0.5:
        parts.append("下方" if dy > 0 else "上方")
    if abs(dx) > abs(dy) * 0.5:
        parts.append("右方" if dx > 0 else "左方")
    return "".join(parts) or "同位置"


def _calc_safe_directions(px, py, ghosts, passable: list[str]) -> list[str]:
    """在可通行方向中，推荐远离鬼魂的方向"""
    if px is None or not passable:
        return ["NOOP"]
    dangerous = [g for g in ghosts if not g["frightened"] and g["distance"] < 60]
    if not dangerous:
        return passable  # 无危险，所有通路都安全

    # 计算鬼魂平均位置
    avg_gx = sum(g["x"] for g in dangerous) / len(dangerous)
    avg_gy = sum(g["y"] for g in dangerous) / len(dangerous)

    # 每个可通行方向打分：远离鬼魂得分高
    dir_vectors = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    scored = []
    for d in passable:
        if d not in dir_vectors:
            continue
        ddx, ddy = dir_vectors[d]
        # 该方向移动后与鬼魂的距离变化（正=远离）
        score = (px + ddx * 8 - avg_gx) ** 2 + (py + ddy * 8 - avg_gy) ** 2
        scored.append((score, d))
    scored.sort(reverse=True)
    return [d for _, d in scored] if scored else passable
