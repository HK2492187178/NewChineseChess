"""MiniMax API调用模块 - 用于蒸馏学习"""

import json
import time
import hashlib
import numpy as np
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG
from game import move_id2move_action, move_action2move_id


class MiniMaxAPI:
    """MiniMax API调用封装，支持蒸馏学习"""

    def __init__(self):
        self.api_key = CONFIG['minimax_api_key']
        self.api_base = CONFIG['minimax_api_base']  # API base: https://api.minimaxi.com
        self.api_url = self.api_base + '/v1/chat/completions'  # 完整端点地址
        self.model = CONFIG['minimax_model']
        self.temperature = CONFIG['distill_temperature']
        self.cache_size = CONFIG['distill_cache_size']
        self.cache = {}  # 局面hash -> (policy, value)
        self.cache_order = deque(maxlen=self.cache_size)
        self.request_interval = 0.5  # 请求间隔(秒)，避免限流
        self.last_request_time = 0
        self.timeout = 30  # 请求超时时间

        if not self.api_key:
            print("警告: MiniMax API密钥未配置，蒸馏功能将不可用")

    def _generate_cache_key(self, state_array):
        """生成局面的唯一缓存键"""
        state_bytes = state_array.tobytes()
        return hashlib.md5(state_bytes).hexdigest()

    def _encode_board_to_text(self, state_list):
        """
        将棋盘状态转换为文本描述，供LLM理解

        Args:
            state_list: 10x9的二维列表，表示棋盘状态

        Returns:
            str: 棋盘文本描述
        """
        piece_names = {
            '红车': 'R车', '红马': 'R马', '红象': 'R象', '红士': 'R士', '红帅': 'R帅', '红炮': 'R炮', '红兵': 'R兵',
            '黑车': 'B车', '黑马': 'B马', '黑象': 'B象', '黑士': 'B士', '黑帅': 'B帅', '黑炮': 'B炮', '黑兵': 'B兵',
            '一一': '　'
        }

        lines = ["中国象棋局面描述：", "红方(上半部分)："]
        for i in range(5):
            row = ' '.join([piece_names.get(state_list[i][j], '？') for j in range(9)])
            lines.append(f"第{i+1}行: {row}")

        lines.append("黑方(下半部分)：")
        for i in range(5, 10):
            row = ' '.join([piece_names.get(state_list[i][j], '？') for j in range(9)])
            lines.append(f"第{i+1}行: {row}")

        return '\n'.join(lines)

    def _build_prompt(self, state_list, current_player_color):
        """
        构建请求MiniMax的prompt

        Args:
            state_list: 当前棋盘状态
            current_player_color: 当前玩家颜色('红'或'黑')

        Returns:
            str: 格式化的问题prompt
        """
        board_text = self._encode_board_to_text(state_list)
        prompt = f"""{board_text}

当前行动方: {'红方' if current_player_color == '红' else '黑方'}

请作为中国象棋专家，分析当前局面并给出建议：

1. 给出当前局面的评估(从红方视角，范围-1到1，1表示红方必胜，-1表示红方必败)
2. 给出下一步的最佳走法建议(从以下合法走子中选择):
   - 列出你认为最好的3-5个走法，并说明理由

请用JSON格式回复，包含以下字段：
- "value": 局面评估分数(-1到1之间)
- "best_moves": 最佳走法列表，每个元素包含"move"(如"0010"格式)和"reason"(理由)
- "analysis": 简要分析当前局面

注意：走子格式为"起点y起点x终点y终点x"，例如"0010"表示从(0,0)走到(1,0)
"""
        return prompt

    def _parse_response(self, response_text):
        """
        解析MiniMax API响应，提取策略和价值

        Args:
            response_text: API返回的文本

        Returns:
            tuple: (policy_probs, value) 或 None
        """
        try:
            data = json.loads(response_text)

            # 提取价值评估
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            if not content:
                return None

            # 尝试解析JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            analysis = json.loads(content.strip())

            value = float(analysis.get('value', 0))
            value = np.clip(value, -1, 1)

            # 策略通过best_moves生成
            # 注意：MiniMax返回的是文本描述的走子，需要映射到动作ID
            policy_probs = np.zeros(2086)
            best_moves = analysis.get('best_moves', [])

            if best_moves:
                total_weight = 0
                move_weights = {}

                # 给前几个走子分配较高权重
                for i, move_info in enumerate(best_moves[:5]):
                    move_str = move_info.get('move', '')
                    if move_str in move_action2move_id:
                        weight = 1.0 / (i + 1)  # 越靠前权重越高
                        move_weights[move_action2move_id[move_str]] = weight
                        total_weight += weight

                # 归一化为概率
                if total_weight > 0:
                    for move_id, weight in move_weights.items():
                        policy_probs[move_id] = weight / total_weight

            return policy_probs, value

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"解析API响应失败: {e}")
            return None

    def get_teacher_guidance(self, board, use_cache=True):
        """
        获取教师模型（MiniMax API）对局面的指导

        Args:
            board: Board对象
            use_cache: 是否使用缓存

        Returns:
            tuple: (policy_probs, value) 或 None
            policy_probs: 2086维动作概率向量
            value: 局面评估分数 [-1, 1]
        """
        if not self.api_key:
            return None

        # 获取当前状态
        state_list = board.state_deque[-1]
        current_player_color = board.current_player_color

        # 生成缓存键
        cache_key = self._generate_cache_key(board.current_state())

        # 检查缓存
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # 请求间隔控制
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

        # 构建请求
        prompt = self._build_prompt(state_list, current_player_color)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_tokens': 2048
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            self.last_request_time = time.time()

            if response.status_code == 200:
                result = self._parse_response(response.text)
                if result and use_cache:
                    self.cache[cache_key] = result
                    self.cache_order.append(cache_key)
                return result
            else:
                print(f"API请求失败: {response.status_code} - {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            print("API请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"API请求异常: {e}")
            return None

    def get_batch_guidance(self, boards, use_cache=True, max_workers=4):
        """
        批量获取教师指导（并发）

        Args:
            boards: Board对象列表
            use_cache: 是否使用缓存
            max_workers: 最大并发数

        Returns:
            list: [(policy_probs, value), ...]，失败的位置为None
        """
        results = [None] * len(boards)

        def fetch_single(idx, board):
            result = self.get_teacher_guidance(board, use_cache)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_single, i, board)
                for i, board in enumerate(boards)
            ]

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def get_guidance_from_state(self, state_list, current_player_color):
        """
        从棋盘列表状态获取指导（不依赖Board对象）

        Args:
            state_list: 10x9的二维列表
            current_player_color: 当前玩家颜色

        Returns:
            tuple: (policy_probs, value) 或 None
        """
        if not self.api_key:
            return None

        # 生成缓存键
        cache_key = hashlib.md5(str(state_list).encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 请求间隔控制
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

        prompt = self._build_prompt(state_list, current_player_color)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_tokens': 2048
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            self.last_request_time = time.time()

            if response.status_code == 200:
                result = self._parse_response(response.text)
                if result:
                    self.cache[cache_key] = result
                    self.cache_order.append(cache_key)
                return result
            else:
                print(f"API请求失败: {response.status_code}")
                return None

        except Exception as e:
            print(f"请求异常: {e}")
            return None

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_order.clear()
        print("蒸馏缓存已清空")

    def get_cache_size(self):
        """获取当前缓存大小"""
        return len(self.cache)


def encode_state_for_api(state_array):
    """
    将状态数组转换为API可处理的格式

    Args:
        state_array: [9, 10, 9]的numpy数组

    Returns:
        dict: 编码后的状态字典
    """
    # 7通道棋子编码 -> 14通道(红7+黑7)
    piece_planes = state_array[:7]  # [7, 10, 9]

    red_pieces = np.zeros((7, 10, 9))
    black_pieces = np.zeros((7, 10, 9))

    for c in range(7):
        red_pieces[c] = np.maximum(piece_planes[c], 0)
        black_pieces[c] = np.maximum(-piece_planes[c], 0)

    encoded = np.concatenate([red_pieces, black_pieces], axis=0)  # [14, 10, 9]

    return {
        'planes': encoded.tolist(),
        'shape': encoded.shape
    }


# 全局API实例
_minimax_api = None


def get_minimax_api():
    """获取MiniMax API单例"""
    global _minimax_api
    if _minimax_api is None:
        _minimax_api = MiniMaxAPI()
    return _minimax_api


if __name__ == '__main__':
    # 测试代码
    api = MiniMaxAPI()
    print("MiniMax API模块初始化完成")
    print(f"API密钥状态: {'已配置' if api.api_key else '未配置'}")
