"""
MsPacman LLM Agent
轮询游戏状态 → 构建Prompt → 调用LLM → 发送动作
"""
import os
import re
import json
import time
import sys
import requests
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pacman.logger import AgentLogger


ACTIONS_DESC = {
    "NOOP":      "不动，保持原地",
    "UP":        "向上移动",
    "RIGHT":     "向右移动",
    "LEFT":      "向左移动",
    "DOWN":      "向下移动",
    "UPRIGHT":   "右上斜移",
    "UPLEFT":    "左上斜移",
    "DOWNRIGHT": "右下斜移",
    "DOWNLEFT":  "左下斜移",
}


class PacmanAgent:
    def __init__(
        self,
        game_url: str = "http://127.0.0.1:5001",
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_steps: int = 1000,
        interval: float = 0.5,
        log_dir: Optional[str] = None,
    ):
        self.game_url = game_url.rstrip("/")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_steps = max_steps
        self.interval = interval
        self.session = requests.Session()
        self.history = []

        # 初始化日志
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs"
            )
        self.logger = AgentLogger(log_dir)

    # ------------------------------------------------------------------ #
    #  游戏通信
    # ------------------------------------------------------------------ #
    def _get_state(self) -> dict:
        r = self.session.get(f"{self.game_url}/state", timeout=5)
        r.raise_for_status()
        return r.json()["data"]

    def _send_action(self, action: str, steps: int = 5) -> dict:
        r = self.session.post(
            f"{self.game_url}/action",
            json={"action": action, "steps": steps},
            timeout=steps * 0.5 + 5,
        )
        r.raise_for_status()
        return r.json()

    def _is_alive(self) -> bool:
        try:
            return self.session.get(f"{self.game_url}/health", timeout=2).status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Prompt 构建
    # ------------------------------------------------------------------ #
    def _build_prompt(self, state: dict) -> str:
        px = state.get("pacman_x", "?")
        py = state.get("pacman_y", "?")
        score = state.get("score", 0)
        lives = state.get("lives", 3)
        dots = state.get("dots_pixels", 0)
        safe_dirs = state.get("safe_directions", [])
        passable = state.get("passable_directions", ["UP", "DOWN", "LEFT", "RIGHT"])
        dir_info = state.get("direction_info", {})
        ghosts = state.get("ghosts", [])
        dangerous = state.get("dangerous_ghost")
        last_action = state.get("last_action", "NOOP")
        last_move_success = state.get("last_move_success", True)
        last_blocked = state.get("last_blocked_direction")

        lines = [
            "你是控制 Ms.Pac-Man 的AI。目标：吃豆子得高分，躲鬼魂。",
            "",
            "=== 当前状态 ===",
            f"位置: ({px},{py})  分数:{score}  生命:{lives}  剩余豆子像素:{dots}",
            f"上一步: {last_action} {'✓成功' if last_move_success else '✗撞墙！请换方向！'}",
        ]

        if last_blocked:
            lines.append(f"  ⚠ {last_blocked} 方向不通，禁止再选！")

        # === 历史轨迹摘要（最近8步） ===
        if self.history:
            recent = self.history[-8:]
            trail = []
            for h in recent:
                pos = h.get("pos", "")
                act = h.get("action", "?")
                trail.append(f"{act}→{pos}")
            lines += [
                "",
                "=== 最近轨迹（旧→新） ===",
                "  " + " | ".join(trail),
            ]
            # 计算来回振荡警告
            recent_actions = [h["action"] for h in recent]
            if len(recent_actions) >= 4:
                osc = sum(1 for i in range(len(recent_actions) - 1)
                          if recent_actions[i] != recent_actions[i + 1]) / (len(recent_actions) - 1)
                if osc > 0.7:
                    lines.append("  ⚠ 检测到来回走！请选一个方向持续走下去，不要反复切换！")

            # 统计最近去过的方向——帮助AI探索新区域
            from collections import Counter
            act_counts = Counter(recent_actions)
            most_used = act_counts.most_common(1)[0]
            least_dirs = [d for d in ["UP", "DOWN", "LEFT", "RIGHT"]
                          if act_counts.get(d, 0) == 0]
            if least_dirs:
                lines.append(f"  💡 最近没有走过: {', '.join(least_dirs)}，建议探索这些方向")

        # === 四方向详细信息 ===
        lines += ["", "=== 四方向详情 ==="]
        for d_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
            info = dir_info.get(d_name, {})
            is_pass = info.get("passable", d_name in passable)
            corridor = info.get("corridor", 0)
            dot_count = info.get("dots", 0)
            blocked_mark = " ⛔禁止(上次撞墙)" if d_name == last_blocked else ""

            if not is_pass:
                lines.append(f"  {d_name}: ✗墙壁{blocked_mark}")
            else:
                dot_str = f"🟡{dot_count}颗豆子" if dot_count > 0 else "无豆子"
                lines.append(f"  {d_name}: ✓通行 通道{corridor}px {dot_str}{blocked_mark}")

        if ghosts:
            lines.append("")
            lines.append("=== 鬼魂 ===")
            for g in ghosts[:4]:
                scared_str = "【可吃！】" if g["frightened"] else "【危险】"
                lines.append(
                    f"  {g['name']}: {g['direction_from_pacman']} {g['distance']:.0f}px {scared_str}"
                )

        if dangerous:
            lines.append("")
            lines.append(
                f"⚠ 紧急: {dangerous['name']} 仅{dangerous['distance']:.0f}px！"
            )
            safe_passable = [d for d in safe_dirs if d in passable]
            lines.append(f"  逃跑方向: {', '.join(safe_passable) if safe_passable else ', '.join(passable)}")

        lines += [
            "",
            "=== 决策规则(按优先级) ===",
            "1. 撞墙→必须换方向",
            "2. 鬼魂<50px且非受惊→逃跑(选远离鬼的可通行方向)",
            "3. 受惊鬼魂→追着吃",
            "4. 优先选有豆子的方向，且选最近没走过的路",
            "5. ❗不要来回走！选定方向就持续走到底（至少3-5步）",
            "",
            '输出JSON: {"action":"方向","steps":连续步数(1-10),"reason":"理由"}',
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  LLM 调用
    # ------------------------------------------------------------------ #
    def _call_llm(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "mock":
            return self._call_mock(prompt)
        elif self.provider in ("openai", "ollama"):
            return self._call_openai_compat(prompt)
        raise ValueError(f"不支持的provider: {self.provider}")

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        api_key = self.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**kwargs)
        msg = client.messages.create(
            model=self.model,
            max_tokens=128,
            system="你是游戏AI，只输出JSON格式决策，不要其他文字。",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def _call_openai_compat(self, prompt: str) -> str:
        from openai import OpenAI
        kwargs = {"api_key": self.api_key or os.environ.get("OPENAI_API_KEY", "openai")}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        elif self.provider == "ollama":
            kwargs["base_url"] = "http://localhost:11434/v1"
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是游戏AI，只输出JSON格式决策。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def _call_mock(self, prompt: str) -> str:
        """规则引擎mock"""
        import random
        # 提取上次动作是否成功
        last_move_failed = "✗撞墙" in prompt
        # 提取上次被堵的方向
        blocked_match = re.search(r"(\w+) 方向不通", prompt)
        blocked_dir = blocked_match.group(1) if blocked_match else None

        # 从四方向详情提取可通行且有豆子的方向
        passable = []
        dot_dirs = []  # 有豆子的方向
        for m in re.finditer(r"(\w+): ✓通行 通道(\d+)px\s*(🟡(\d+)颗豆子|无豆子)", prompt):
            d_name = m.group(1)
            if d_name in ("UP", "DOWN", "LEFT", "RIGHT"):
                passable.append(d_name)
                dots = int(m.group(4)) if m.group(4) else 0
                if dots > 0:
                    dot_dirs.append((dots, d_name))
        dot_dirs.sort(reverse=True)  # 豆子多的优先

        if not passable:
            passable = ["DOWN"]
        # 排除被堵方向
        if blocked_dir:
            passable = [d for d in passable if d != blocked_dir] or passable
            dot_dirs = [(n, d) for n, d in dot_dirs if d != blocked_dir]

        # 紧急逃跑
        if "⚠ 紧急" in prompt:
            safe_m = re.search(r"逃跑方向:\s*([^\n]+)", prompt)
            if safe_m:
                dirs = [d.strip() for d in safe_m.group(1).split(",")
                        if d.strip() in ("UP", "DOWN", "LEFT", "RIGHT")]
                dirs = [d for d in dirs if d != blocked_dir] or dirs
                if dirs:
                    return f'{{"action": "{dirs[0]}", "steps": 3, "reason": "紧急逃跑"}}'
            return f'{{"action": "{random.choice(passable)}", "steps": 3, "reason": "紧急逃跑"}}'
        # 有受惊鬼魂就追
        if "可吃" in prompt:
            ghost_m = re.search(r"(\S+)\s+\d+px\s+【可吃", prompt)
            if ghost_m:
                dir_map = {"右方": "RIGHT", "左方": "LEFT", "上方": "UP", "下方": "DOWN"}
                d = ghost_m.group(1)
                for k, v in dir_map.items():
                    if k in d and v in passable:
                        return f'{{"action": "{v}", "steps": 4, "reason": "追击受惊鬼魂"}}'
        # 优先选有豆子的方向
        if dot_dirs:
            return f'{{"action": "{dot_dirs[0][1]}", "steps": 6, "reason": "朝豆子方向探索"}}'
        return f'{{"action": "{random.choice(passable)}", "steps": 6, "reason": "探索"}}'

    def _parse_action(self, response: str) -> tuple[str, str, int]:
        steps = 5
        try:
            data = json.loads(response)
            action = data.get("action", "NOOP").upper()
            reason = data.get("reason", "")
            steps = max(1, min(10, int(data.get("steps", 5))))
            return action, reason, steps
        except (json.JSONDecodeError, ValueError):
            pass
        m = re.search(r'"action"\s*:\s*"([^"]+)"', response)
        if m:
            return m.group(1).upper(), "", steps
        return "NOOP", f"解析失败: {response[:80]}", 1

    # ------------------------------------------------------------------ #
    #  主循环
    # ------------------------------------------------------------------ #
    def run(self):
        log = self.logger
        log.log_connect(self.game_url, self.provider, self.model)

        if not self._is_alive():
            log.error("游戏服务器未启动，请先运行 run_pacman_game.py")
            return

        consecutive_llm_errors = 0
        episode = 0
        episode_score = 0
        episode_steps = 0
        # 卡住检测：记录最近几步的位置
        recent_positions = []

        for step in range(self.max_steps):
            # 1. 获取状态
            try:
                state = self._get_state()
            except Exception as e:
                log.error(f"获取状态失败: {e}")
                time.sleep(1)
                continue

            # 检测本局结束（三条命全部用完 或 ALE episode 结束）
            lives = state.get("lives", 3)
            if state.get("episode_done") or lives <= 0:
                log.log_episode_end(episode, episode_score, episode_steps,
                                    f"游戏结束 lives={lives}")
                break  # 一局结束就退出

            # 卡住检测：如果最近3步位置都在±3像素内，强制换方向
            cur_pos = (state.get("pacman_x"), state.get("pacman_y"))
            recent_positions.append(cur_pos)
            if len(recent_positions) > 4:
                recent_positions.pop(0)

            stuck = False
            if len(recent_positions) >= 3 and cur_pos[0] is not None:
                px0, py0 = cur_pos
                stuck = all(
                    abs((p[0] or px0) - px0) <= 3 and abs((p[1] or py0) - py0) <= 3
                    for p in recent_positions[-3:]
                )

            if stuck:
                # 从所有四方向中选一个最近没用过的方向（不限于 passable，因为 passable 可能不准）
                all_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
                recent_actions = [h["action"] for h in self.history[-5:]]
                # 优先选没用过的方向
                fallback = next(
                    (d for d in all_dirs if d not in recent_actions),
                    None,
                )
                if not fallback:
                    # 所有方向都试过了，选用得最少的
                    from collections import Counter
                    counts = Counter(recent_actions)
                    fallback = min(all_dirs, key=lambda d: counts.get(d, 0))
                log.warn(f"[卡住检测] 位置 {cur_pos} 连续3步未移动，强制执行 {fallback}")
                try:
                    result = self._send_action(fallback, 3)
                    reward = result.get("action_result", {}).get("reward", 0)
                    action_done = result.get("action_result", {}).get("done", False)
                except Exception as e:
                    log.error(f"发送动作失败: {e}")
                    time.sleep(0.5)
                    continue
                episode_score += reward
                episode_steps += 1
                log.log_step(
                    step=step, action=fallback, reason="[卡住强制换向]",
                    state=state, reward=reward, llm_response=None, llm_error=None,
                )
                self.history.append({
                    "step": step, "action": fallback, "reward": reward,
                    "score": state.get("score", 0), "reason": "[卡住强制换向]",
                    "pos": f"({state.get('pacman_x','?')},{state.get('pacman_y','?')})",
                })
                recent_positions.clear()  # 清空，重新检测
                if action_done:
                    log.log_episode_end(episode, episode_score, episode_steps,
                                        "游戏结束(卡住检测中)")
                    break
                time.sleep(self.interval)
                continue

            # 2. 构建 Prompt & 调用 LLM
            prompt = self._build_prompt(state)
            llm_response = None
            llm_error = None
            t0 = time.time()
            try:
                llm_response = self._call_llm(prompt)
                elapsed_ms = int((time.time() - t0) * 1000)
                log.log_llm_call(len(prompt), llm_response, elapsed_ms)
                action, reason, steps = self._parse_action(llm_response)
                consecutive_llm_errors = 0
            except Exception as e:
                llm_error = str(e)
                log.warn(f"LLM 调用失败: {e}")
                action, reason, steps = "NOOP", "LLM异常", 1
                consecutive_llm_errors += 1
                if consecutive_llm_errors >= 3:
                    log.warn("连续LLM失败，暂停3秒...")
                    time.sleep(3)
                    consecutive_llm_errors = 0

            # 3. 执行动作
            try:
                result = self._send_action(action, steps)
                reward = result.get("action_result", {}).get("reward", 0)
                action_done = result.get("action_result", {}).get("done", False)
            except Exception as e:
                log.error(f"发送动作失败: {e}")
                time.sleep(0.5)
                continue

            episode_score += reward
            episode_steps += 1

            # 4. 结构化日志
            log.log_step(
                step=step,
                action=action,
                reason=reason,
                state=state,
                reward=reward,
                llm_response=llm_response,
                llm_error=llm_error,
            )

            self.history.append({
                "step": step, "action": action, "reward": reward,
                "score": state.get("score", 0), "reason": reason,
                "pos": f"({state.get('pacman_x','?')},{state.get('pacman_y','?')})",
            })

            # 5. 检测动作执行后游戏是否结束（三条命用完）
            if action_done:
                log.log_episode_end(episode, episode_score, episode_steps,
                                    "游戏结束(action_done)")
                break

            time.sleep(self.interval)

        total_score = sum(h["reward"] for h in self.history)
        log.log_game_done(total_score)
        log.close()
