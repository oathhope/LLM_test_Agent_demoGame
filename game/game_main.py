"""
MsPacman 游戏主进程
- 运行 Atari MsPacman（真实开源游戏，ale_py + gymnasium）
- pygame 实时渲染画面（可视窗口）
- 内嵌 Flask HTTP RPC 服务器，暴露状态和动作接口
- 支持 Agent 通过 RPC 控制游戏
"""
import sys
import os
import threading
import time
import numpy as np

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import pygame
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.state_extractor import extract_state, _find_color_centroid

# Pacman 颜色常量
PACMAN_COLOR = (210, 164, 74)

# ------------------------------------------------------------------ #
#  全局游戏状态
# ------------------------------------------------------------------ #
ACTIONS = {
    "NOOP":      {"id": 0, "desc": "不动"},
    "UP":        {"id": 1, "desc": "向上移动"},
    "RIGHT":     {"id": 2, "desc": "向右移动"},
    "LEFT":      {"id": 3, "desc": "向左移动"},
    "DOWN":      {"id": 4, "desc": "向下移动"},
    "UPRIGHT":   {"id": 5, "desc": "右上斜移"},
    "UPLEFT":    {"id": 6, "desc": "左上斜移"},
    "DOWNRIGHT": {"id": 7, "desc": "右下斜移"},
    "DOWNLEFT":  {"id": 8, "desc": "左下斜移"},
}

game_state = {
    "frame": None,          # 当前RGB帧 (numpy)
    "structured": {},       # 结构化状态
    "score": 0,
    "lives": 3,
    "episode_done": False,
    "total_reward": 0.0,
    "step_count": 0,
    "pending_action": None, # Agent下发的动作名
    "pending_steps": 1,     # 连续执行步数（1-10）
    "action_result": None,  # 上次动作结果
    "last_action_name": "NOOP",
    "prev_pacman_x": None,  # 上次动作前Pacman X坐标（用于判断移动是否成功）
    "prev_pacman_y": None,
    "last_move_success": True,   # 上次动作是否真正移动了
    "blocked_directions": [],    # 上次检测到可能被堵的方向
}
state_lock = threading.Lock()

# ------------------------------------------------------------------ #
#  Flask RPC 服务器
# ------------------------------------------------------------------ #
app = Flask(__name__)


@app.route("/state", methods=["GET"])
def get_state():
    with state_lock:
        s = game_state["structured"].copy()
        s["score"] = game_state["score"]
        s["lives"] = game_state["lives"]
        s["episode_done"] = game_state["episode_done"]
        s["total_reward"] = game_state["total_reward"]
        s["step_count"] = game_state["step_count"]
        s["last_action"] = game_state["last_action_name"]
        s["available_actions"] = {k: v["desc"] for k, v in ACTIONS.items()}
    return jsonify({"status": "ok", "data": s})


@app.route("/action", methods=["POST"])
def post_action():
    body = request.get_json(force=True, silent=True) or {}
    action_name = body.get("action", "NOOP").upper().strip()
    steps = max(1, min(10, int(body.get("steps", 1))))
    if action_name not in ACTIONS:
        return jsonify({"status": "error", "message": f"未知动作: {action_name}"}), 400

    with state_lock:
        game_state["pending_action"] = action_name
        game_state["pending_steps"] = steps

    # 等待游戏主循环执行完毕（steps越多等越久，最多 steps*0.5+2 秒）
    deadline = time.time() + steps * 0.5 + 2.0
    while time.time() < deadline:
        with state_lock:
            if game_state["pending_action"] is None:
                result = game_state["action_result"]
                break
        time.sleep(0.02)
    else:
        result = {"success": False, "message": "执行超时"}

    with state_lock:
        s = game_state["structured"].copy()
        s["score"] = game_state["score"]
        s["lives"] = game_state["lives"]
        s["episode_done"] = game_state["episode_done"]

    return jsonify({"status": "ok", "action_result": result, "state": s})


@app.route("/reset", methods=["POST"])
def reset_game():
    with state_lock:
        game_state["pending_action"] = "__RESET__"
    time.sleep(0.5)
    return jsonify({"status": "ok", "message": "游戏已重置"})


@app.route("/actions", methods=["GET"])
def list_actions():
    return jsonify({"status": "ok", "actions": {k: v["desc"] for k, v in ACTIONS.items()}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "MsPacman game server running"})


# ------------------------------------------------------------------ #
#  游戏主循环
# ------------------------------------------------------------------ #
SCALE = 3  # 窗口放大倍数（原始 160×210 → 480×630）

def run_game(rpc_host: str = "127.0.0.1", rpc_port: int = 5001):
    # 启动 Flask 线程
    flask_thread = threading.Thread(
        target=lambda: app.run(host=rpc_host, port=rpc_port, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"[GameServer] RPC 服务器启动在 http://{rpc_host}:{rpc_port}")

    # 初始化游戏
    env = gym.make("ALE/MsPacman-v5", obs_type="rgb", render_mode=None, frameskip=4)
    obs, info = env.reset()
    ale = env.unwrapped.ale  # ALE 对象，用于精确方向检测（clone/restore state）

    # 跳过开场动画（约70步NOOP后Pacman才真正可以移动）
    SKIP_INTRO_STEPS = 72
    def skip_intro(env):
        """执行NOOP跳过开场动画，返回最新obs和info"""
        obs, info = None, {}
        for _ in range(SKIP_INTRO_STEPS):
            obs, reward, term, trunc, info = env.step(0)  # NOOP
            if term or trunc:
                obs, info = env.reset()
                break
        return obs, info

    print("[Game] 跳过开场动画...")
    obs, info = skip_intro(env)
    print("[Game] 开场动画跳过完毕，游戏正式开始")

    # 初始化 pygame 窗口
    pygame.init()
    W, H = 160 * SCALE, 210 * SCALE
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MsPacman — LLM Agent Playing")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    with state_lock:
        game_state["frame"] = obs
        game_state["structured"] = extract_state(obs, 0, 3, ale)
        game_state["lives"] = info.get("lives", 3)

    current_action_id = 0  # NOOP
    step_reward = 0.0

    print("[Game] 游戏窗口已启动，等待 Agent 连接...")

    running = True
    while running:
        # pygame 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 检查 Agent 下发的动作
        with state_lock:
            pending = game_state["pending_action"]

        if pending == "__RESET__":
            obs, info = env.reset()
            with state_lock:
                game_state["pending_action"] = None
                game_state["episode_done"] = False
                game_state["score"] = 0
                game_state["total_reward"] = 0.0
                game_state["step_count"] = 0
                game_state["lives"] = info.get("lives", 3)
                game_state["frame"] = obs
                game_state["structured"] = extract_state(obs, 0, 3, ale)
                game_state["action_result"] = {"success": True, "message": "游戏已重置"}
            current_action_id = 0
        elif pending is not None:
            current_action_id = ACTIONS[pending]["id"]
            with state_lock:
                n_steps = game_state["pending_steps"]
                # 记录动作前 Pacman 位置
                prev_s = game_state["structured"]
                pre_px = prev_s.get("pacman_x")
                pre_py = prev_s.get("pacman_y")

            total_reward = 0
            done = False
            for _ in range(n_steps):
                obs, reward, terminated, truncated, info = env.step(current_action_id)
                total_reward += int(reward)
                # 每步都更新帧供 pygame 渲染（画面流畅）
                with state_lock:
                    game_state["frame"] = obs
                    game_state["lives"] = info.get("lives", game_state["lives"])
                # 渲染一次中间帧
                if frame is not None:
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    surf_scaled = pygame.transform.scale(surf, (W, H))
                    screen.blit(surf_scaled, (0, 0))
                    pygame.display.flip()
                    clock.tick(30)
                if terminated or truncated:
                    done = True
                    break

            # 计算实际位移，判断动作是否真正移动了 Pacman
            post_px, post_py = _find_color_centroid(obs, PACMAN_COLOR)
            move_dx = (post_px or pre_px or 0) - (pre_px or 0)
            move_dy = (post_py or pre_py or 0) - (pre_py or 0)
            actually_moved = abs(move_dx) > 2 or abs(move_dy) > 2

            # 判断移动方向是否与请求方向一致
            dir_vectors = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            if pending in dir_vectors and pre_px is not None and post_px is not None:
                exp_dx, exp_dy = dir_vectors[pending]
                # 检查实际移动方向是否与期望方向大致一致
                dir_match = (exp_dx * move_dx > 0 or exp_dy * move_dy > 0) and actually_moved
                move_success = dir_match
                # 如果没按期望方向移动（可能撞墙了），记录被堵方向
                blocked = [pending] if not dir_match else []
            else:
                move_success = actually_moved
                blocked = []

            with state_lock:
                game_state["score"] += total_reward
                game_state["total_reward"] += total_reward
                game_state["step_count"] += n_steps
                game_state["episode_done"] = done
                game_state["last_action_name"] = pending
                game_state["last_move_success"] = move_success
                game_state["blocked_directions"] = blocked
                new_state = extract_state(
                    obs,
                    game_state["score"],
                    game_state["lives"],
                    ale,
                )
                # 把移动结果注入 state 供 agent 使用
                new_state["last_move_success"] = move_success
                new_state["last_blocked_direction"] = pending if blocked else None
                game_state["structured"] = new_state
                game_state["action_result"] = {
                    "success": True,
                    "message": f"执行 {pending}×{n_steps}步，获得奖励 {total_reward}",
                    "reward": total_reward,
                    "done": done,
                    "moved": actually_moved,
                    "move_dx": move_dx,
                    "move_dy": move_dy,
                }
                game_state["pending_action"] = None

            if done:
                obs, info = env.reset()
                obs, info = skip_intro(env)  # 每次episode结束也跳过开场动画
                with state_lock:
                    game_state["episode_done"] = False
                    game_state["score"] = 0
                    game_state["frame"] = obs
                    game_state["structured"] = extract_state(obs, 0, info.get("lives", 3), ale)
        else:
            # 无 Agent 动作时：游戏暂停，只渲染当前帧，不推进时间步
            # 这样 LLM 思考期间怪物不会移动，彻底解决"AI还没决策就被追到"的问题
            pass

        # 渲染画面
        with state_lock:
            frame = game_state["frame"]
            score = game_state["score"]
            lives = game_state["lives"]
            last_action = game_state["last_action_name"]
            structured = game_state["structured"]
            waiting = game_state["pending_action"] is None  # True=正在等Agent决策

        if frame is not None:
            # RGB numpy -> pygame surface
            surf = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2))  # (H,W,3) -> (W,H,3)
            )
            surf_scaled = pygame.transform.scale(surf, (W, H))
            screen.blit(surf_scaled, (0, 0))

            # HUD 信息叠加
            lines = [
                f"Score: {score}  Lives: {lives}  Steps: {game_state['step_count']}",
                f"Last Action: {last_action}  {'[ 等待AI决策... ]' if waiting else '[ 执行中 ]'}",
                f"Pacman: ({structured.get('pacman_x','?')}, {structured.get('pacman_y','?')})",
            ]
            nearest = structured.get("nearest_ghost")
            if nearest:
                lines.append(f"Nearest Ghost: {nearest['name']} dist={nearest['distance']:.0f} {'⚠' if not nearest['frightened'] else '✓scared'}")

            for i, line in enumerate(lines):
                text_surf = font.render(line, True, (255, 255, 0), (0, 0, 0))
                screen.blit(text_surf, (5, 5 + i * 18))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS 渲染

    pygame.quit()
    env.close()
    print("[Game] 游戏结束")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    run_game(args.host, args.port)
