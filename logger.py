"""
日志模块 - MsPacman LLM Agent 运行日志
支持同时输出到控制台和日志文件，带时间戳和颜色高亮
"""
import os
import sys
import json
import logging
import datetime
from typing import Optional

# Windows 终端 UTF-8 输出支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ANSI 颜色（Windows 终端支持）
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"


def _supports_color() -> bool:
    """检测终端是否支持ANSI颜色"""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 启用 ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def _c(color: str, text: str) -> str:
    if USE_COLOR:
        return f"{color}{text}{C.RESET}"
    return text


class AgentLogger:
    """
    结构化日志记录器
    - 控制台：带颜色的格式化输出
    - 文件：JSONL 格式（每行一条记录），方便后续分析
    - 文件：同步写入人类可读的 .log 文本文件
    """

    def __init__(self, log_dir: str, run_id: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.jsonl_path = os.path.join(log_dir, f"run_{run_id}.jsonl")
        self.text_path  = os.path.join(log_dir, f"run_{run_id}.log")
        self.stats_path = os.path.join(log_dir, f"run_{run_id}_stats.json")

        # 初始化文本日志（只让 WARNING+ 输出到控制台，减少噪音）
        logging.basicConfig(level=logging.WARNING)
        self._logger = logging.getLogger(f"pacman.{run_id}")
        self._logger.handlers.clear()
        self._logger.propagate = False

        fh = logging.FileHandler(self.text_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(fh)

        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        self._step_count = 0
        self._total_score = 0
        self._actions_taken = []
        self._llm_calls = 0
        self._llm_errors = 0

        self.info(f"=== 新运行开始 run_id={run_id} ===")
        print(_c(C.CYAN, f"\n{'='*60}"))
        print(_c(C.CYAN, f"  MsPacman LLM Agent 日志系统启动"))
        print(_c(C.CYAN, f"  Run ID : {run_id}"))
        print(_c(C.CYAN, f"  日志目录: {log_dir}"))
        print(_c(C.CYAN, f"  JSONL  : {self.jsonl_path}"))
        print(_c(C.CYAN, f"  Text   : {self.text_path}"))
        print(_c(C.CYAN, f"{'='*60}\n"))

    # ------------------------------------------------------------------ #
    #  基础日志方法
    # ------------------------------------------------------------------ #
    def info(self, msg: str):
        self._logger.info(msg)

    def warn(self, msg: str):
        self._logger.warning(msg)
        print(_c(C.YELLOW, f"[WARN] {msg}"))

    def error(self, msg: str):
        self._logger.error(msg)
        print(_c(C.RED, f"[ERROR] {msg}"))

    def _write_jsonl(self, record: dict):
        record["timestamp"] = datetime.datetime.now().isoformat()
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    # ------------------------------------------------------------------ #
    #  游戏专用日志
    # ------------------------------------------------------------------ #
    def log_connect(self, server_url: str, provider: str, model: str):
        msg = f"连接游戏服务器 {server_url} | provider={provider} model={model}"
        self.info(msg)
        print(_c(C.GREEN, f"[Agent] {msg}"))
        self._write_jsonl({"type": "connect", "server_url": server_url,
                           "provider": provider, "model": model})

    def log_step(
        self,
        step: int,
        action: str,
        reason: str,
        state: dict,
        reward: int,
        llm_response: Optional[str] = None,
        llm_error: Optional[str] = None,
    ):
        self._step_count += 1
        self._total_score += reward
        self._actions_taken.append(action)
        if llm_error:
            self._llm_errors += 1
        else:
            self._llm_calls += 1

        px    = state.get("pacman_x", "?")
        py    = state.get("pacman_y", "?")
        score = state.get("score", 0)
        lives = state.get("lives", 3)
        dots  = state.get("dots_pixels", 0)
        danger = state.get("dangerous_ghost")

        # 颜色选择
        if llm_error:
            action_color = C.GREY
        elif danger:
            action_color = C.RED
        else:
            action_color = C.GREEN

        danger_str = ""
        if danger:
            danger_str = _c(C.RED, f" ⚠{danger['name']}({danger['distance']:.0f}px)")

        # 控制台输出
        step_str   = _c(C.GREY,   f"[Step {step:04d}]")
        action_str = _c(action_color, f"{action:10s}")
        score_str  = _c(C.WHITE,  f"Score:{score:5d}")
        lives_str  = _c(C.YELLOW if lives <= 1 else C.WHITE, f"Lives:{lives}")
        pos_str    = _c(C.CYAN,   f"Pos:({px},{py})")
        dots_str   = _c(C.GREY,   f"Dots:{dots}")
        reason_str = _c(C.GREY,   f"| {reason[:45]}")

        line = f"{step_str} {action_str} | {score_str} {lives_str} {pos_str} {dots_str}{danger_str} {reason_str}"
        print(line)

        # 文件日志
        self.info(
            f"Step {step:04d} action={action} score={score} lives={lives} "
            f"pos=({px},{py}) dots={dots} reason={reason[:60]}"
        )

        # JSONL 结构化记录
        record = {
            "type": "step",
            "step": step,
            "action": action,
            "reason": reason,
            "reward": reward,
            "score": score,
            "lives": lives,
            "pacman_x": px,
            "pacman_y": py,
            "dots_pixels": dots,
            "dangerous_ghost": danger["name"] if danger else None,
            "ghost_distance": danger["distance"] if danger else None,
        }
        if llm_response:
            record["llm_response"] = llm_response
        if llm_error:
            record["llm_error"] = llm_error
        self._write_jsonl(record)

    def log_llm_call(self, prompt_len: int, response: str, elapsed_ms: int):
        self.info(f"LLM调用 prompt_len={prompt_len} elapsed={elapsed_ms}ms response={response[:80]}")
        self._write_jsonl({
            "type": "llm_call",
            "prompt_len": prompt_len,
            "response_preview": response[:120],
            "elapsed_ms": elapsed_ms,
        })

    def log_episode_end(self, episode: int, score: int, steps: int, reason: str):
        msg = f"Episode {episode} 结束 | score={score} steps={steps} reason={reason}"
        self.info(msg)
        print(_c(C.YELLOW, f"\n[Episode] {msg}"))
        self._write_jsonl({"type": "episode_end", "episode": episode,
                           "score": score, "steps": steps, "reason": reason})

    def log_game_done(self, total_score: int):
        msg = f"游戏结束 | 总步数={self._step_count} 总分={total_score} LLM调用={self._llm_calls} LLM错误={self._llm_errors}"
        self.info(msg)
        print(_c(C.CYAN, f"\n{'='*60}"))
        print(_c(C.CYAN, f"[完成] {msg}"))

        # 动作分布
        from collections import Counter
        action_dist = dict(Counter(self._actions_taken))
        print(_c(C.GREY, f"  动作分布: {action_dist}"))
        print(_c(C.CYAN, f"{'='*60}"))

        self._write_jsonl({
            "type": "run_end",
            "total_steps": self._step_count,
            "total_score": total_score,
            "llm_calls": self._llm_calls,
            "llm_errors": self._llm_errors,
            "action_distribution": action_dist,
        })

        # 保存统计摘要
        stats = {
            "run_id": self.run_id,
            "total_steps": self._step_count,
            "total_score": total_score,
            "llm_calls": self._llm_calls,
            "llm_errors": self._llm_errors,
            "action_distribution": action_dist,
        }
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(_c(C.GREY, f"  统计已保存: {self.stats_path}"))

    def close(self):
        self._jsonl_file.close()
        self.info("日志关闭")
