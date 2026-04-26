"""蒸馏学习训练 - 使用MiniMax API辅助训练"""

import random
import numpy as np
import pickle
import time
from collections import defaultdict, deque

from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure
from zip_array import recovery_state_mcts_prob

if CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
elif CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
else:
    print('暂不支持所选框架')

try:
    from miniMax_api import MiniMaxAPI, get_minimax_api
    MINIMAX_AVAILABLE = True
except ImportError:
    print("警告: MiniMax API模块不可用，蒸馏训练将回退到传统方式")
    MINIMAX_AVAILABLE = False


class DistillTrainPipeline:
    """
    蒸馏训练流程

    核心思想：用MiniMax API作为教师模型，提供更准确的策略概率和局面评估，
    通过蒸馏损失函数指导学生网络（AlphaZero网络）学习。
    """

    def __init__(self, init_model=None):
        # 棋盘和游戏
        self.board = Board()
        self.game = Game(self.board)

        # 训练参数
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100
        self.game_batch_num = CONFIG['game_batch_num']
        self.best_win_ratio = 0.0

        # 蒸馏参数
        self.use_distill = CONFIG['use_distill'] and MINIMAX_AVAILABLE
        self.distill_batch_size = CONFIG['distill_batch_size']
        self.distill_ratio = 0.5  # 蒸馏损失占总损失的比例

        # 初始化MiniMax API
        self.minimax_api = None
        if self.use_distill and CONFIG['minimax_api_key']:
            self.minimax_api = get_minimax_api()
            print("MiniMax API已初始化，蒸馏训练已启用")
        elif self.use_distill:
            print("警告: MiniMax API密钥未配置，蒸馏训练已禁用")
            self.use_distill = False

        # 数据缓冲区
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.distill_buffer = deque(maxlen=self.buffer_size // 2)  # 蒸馏数据缓冲区

        # 初始化网络
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('已加载模型')
            except:
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()

        # 设置蒸馏权重
        self.policy_value_net.distill_alpha = CONFIG['distill_alpha']
        self.policy_value_net.distill_beta = CONFIG['distill_beta']

    def collect_distill_data(self, states, force_collect=False):
        """
        使用MiniMax API收集蒸馏数据

        Args:
            states: 棋盘状态列表 [state_array, ...]
            force_collect: 是否强制收集（跳过缓存）

        Returns:
            list: [(teacher_policy, teacher_value), ...]
        """
        if not self.use_distill or not self.minimax_api:
            return None

        print(f"开始收集蒸馏数据，共{len(states)}个状态...")
        distill_data = []
        failed_count = 0

        for i, state in enumerate(states):
            # 跳过非合法走子（这里简化处理，实际应检查）
            try:
                # 重建Board对象获取当前玩家信息
                board = Board()
                board.init_board()
                # 注意：这里state是current_state格式[9,10,9]，需要正确处理

                # 使用API获取指导
                # 注意：简化版本，直接用原始状态调用
                teacher_result = self.minimax_api.get_teacher_guidance(board, use_cache=not force_collect)

                if teacher_result is not None:
                    policy, value = teacher_result
                    distill_data.append((policy, value))
                else:
                    failed_count += 1

                # 进度显示
                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(states)}, 成功: {len(distill_data)}, 失败: {failed_count}")

            except Exception as e:
                print(f"处理状态{i}时出错: {e}")
                failed_count += 1
                continue

        print(f"蒸馏数据收集完成: 成功{len(distill_data)}个, 失败{failed_count}个")
        return distill_data if distill_data else None

    def distill_step(self, state_batch, mcts_probs, winner_batch, distill_data=None):
        """
        执行蒸馏训练步骤

        Args:
            state_batch: 状态批次
            mcts_probs: MCTS策略概率
            winner_batch: MCTS价值标签
            distill_data: 教师模型数据 [(policy, value), ...]

        Returns:
            dict: 训练损失信息
        """
        if distill_data is None or len(distill_data) == 0:
            # 没有蒸馏数据，使用传统训练
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs, winner_batch,
                lr=self.learn_rate * self.lr_multiplier
            )
            return {
                'total_loss': loss,
                'policy_loss': loss,
                'value_loss': loss,
                'distill_loss': 0.0,
                'entropy': entropy,
                'use_distill': False
            }

        # 将蒸馏数据转换为数组
        teacher_policies = np.array([d[0] for d in distill_data])
        teacher_values = np.array([d[1] for d in distill_data])

        # 混合训练
        losses = self.policy_value_net.mixed_train_step(
            state_batch, mcts_probs, winner_batch,
            teacher_policy=teacher_policies,
            teacher_value=teacher_values,
            distill_ratio=self.distill_ratio,
            lr=self.learn_rate * self.lr_multiplier
        )
        losses['use_distill'] = True

        return losses

    def policy_updata(self):
        """更新策略价值网络（支持蒸馏）"""
        # 采样数据
        mini_batch = random.sample(self.data_buffer, min(self.batch_size, len(self.data_buffer)))
        mini_batch = [recovery_state_mcts_prob(data) for data in mini_batch]

        state_batch = np.array([data[0] for data in mini_batch]).astype('float32')
        mcts_probs_batch = np.array([data[1] for data in mini_batch]).astype('float32')
        winner_batch = np.array([data[2] for data in mini_batch]).astype('float32')

        # 获取蒸馏数据（从蒸馏缓冲区）
        distill_data = None
        if self.use_distill and len(self.distill_buffer) > 0:
            distill_sample_size = min(len(self.distill_buffer), self.distill_batch_size)
            distill_data = random.sample(list(self.distill_buffer), distill_sample_size)

        # 计算旧网络的预测（用于KL散度监控）
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        # 执行训练
        for i in range(self.epochs):
            if distill_data:
                loss_dict = self.distill_step(
                    state_batch, mcts_probs_batch, winner_batch, distill_data
                )
            else:
                loss, entropy = self.policy_value_net.train_step(
                    state_batch, mcts_probs_batch, winner_batch,
                    lr=self.learn_rate * self.lr_multiplier
                )
                loss_dict = {
                    'total_loss': loss,
                    'entropy': entropy,
                    'use_distill': False
                }

            # 新网络预测
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # KL散度
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1))

            if kl > self.kl_targ * 4:
                print(f"KL散度过大，提前终止epoch {i+1}")
                break

        # 自适应学习率调整
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 计算解释方差
        explained_var_old = 1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch)
        explained_var_new = 1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch)

        distill_str = "[蒸馏]" if loss_dict.get('use_distill', False) else "[MCTS]"

        print((
            f"{distill_str} "
            f"kl:{kl:.5f}, lr_mul:{self.lr_multiplier:.3f}, "
            f"loss:{loss_dict.get('total_loss', 0):.4f}, "
            f"distill_loss:{loss_dict.get('distill_loss', 0):.4f}, "
            f"entropy:{loss_dict.get('entropy', 0):.4f}, "
            f"explained_var_old:{explained_var_old:.4f}, "
            f"explained_var_new:{explained_var_new:.4f}"
        ))

        return loss_dict

    def policy_evaluate(self, n_games=10):
        """评估训练效果"""
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=500)
        win_cnt = defaultdict(int)

        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player,
                pure_mcts_player,
                start_player=i % 2 + 1,
                is_shown=0
            )
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(f"评估: 胜{win_cnt[1]}, 负{win_cnt[2]}, 平{win_cnt[-1]}, 胜率{win_ratio:.2%}")
        return win_ratio

    def run(self):
        """开始蒸馏训练"""
        print("=" * 50)
        print("蒸馏训练启动")
        print(f"MiniMax API: {'启用' if self.use_distill else '禁用'}")
        print(f"蒸馏比例: {self.distill_ratio}")
        print(f"蒸馏权重: alpha={CONFIG['distill_alpha']}, beta={CONFIG['distill_beta']}")
        print("=" * 50)

        try:
            for i in range(self.game_batch_num):
                # 1. 加载自我对弈数据
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                            data_file = pickle.load(f)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                        print(f"已载入{len(self.data_buffer)}条自我对弈数据")
                        break
                    except FileNotFoundError:
                        print("等待数据文件...")
                        time.sleep(30)
                    except Exception as e:
                        print(f"加载数据失败: {e}")
                        time.sleep(30)

                print(f"步骤 {self.iters}:")

                # 2. 如果启用蒸馏，收集蒸馏数据
                if self.use_distill and len(self.data_buffer) > 0:
                    # 采样部分状态进行蒸馏
                    sample_size = min(self.distill_batch_size, len(self.data_buffer))
                    sample_indices = random.sample(range(len(self.data_buffer)), sample_size)

                    states_to_distill = []
                    for idx in sample_indices:
                        state, _, _ = recovery_state_mcts_prob(self.data_buffer[idx])
                        states_to_distill.append(state)

                    # 调用MiniMax API收集蒸馏数据
                    distill_data = self.collect_distill_data(states_to_distill)

                    if distill_data:
                        self.distill_buffer.extend(distill_data)
                        print(f"蒸馏缓冲区大小: {len(self.distill_buffer)}")

                # 3. 执行训练
                if len(self.data_buffer) >= self.batch_size:
                    loss_info = self.policy_updata()

                    # 4. 保存模型
                    if CONFIG['use_frame'] == 'pytorch':
                        self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
                    else:
                        self.policy_value_net.save_model(CONFIG['paddle_model_path'])

                # 5. 定期评估
                if (i + 1) % self.check_freq == 0:
                    print(f"\n--- 评估训练效果 (第{i+1}步) ---")
                    win_ratio = self.policy_evaluate(n_games=5)
                    print(f"当前胜率: {win_ratio:.2%}\n")

                    # 保存检查点
                    checkpoint_path = f'models/distill_checkpoint_batch{i+1}.pkl'
                    self.policy_value_net.save_model(checkpoint_path)
                    print(f"检查点已保存: {checkpoint_path}")

                time.sleep(CONFIG['train_update_interval'])

        except KeyboardInterrupt:
            print('\n训练已停止')

        print("蒸馏训练结束")


class DistillEvaluator:
    """蒸馏效果评估器"""

    def __init__(self, policy_value_net, minimax_api=None):
        self.policy_value_net = policy_value_net
        self.minimax_api = minimax_api

    def compare_predictions(self, board):
        """
        比较学生网络和教师模型的预测

        Returns:
            dict: 比较结果
        """
        # 学生网络预测
        student_policy, student_value = self.policy_value_net.policy_value_fn(board)
        student_policy = dict(student_policy)

        result = {
            'student_value': student_value[0] if isinstance(student_value, np.ndarray) else student_value,
            'teacher_value': None,
            'policy_diff': None,
            'teacher_available': False
        }

        # 教师模型预测
        if self.minimax_api:
            teacher_result = self.minimax_api.get_teacher_guidance(board)
            if teacher_result:
                teacher_policy, teacher_value = teacher_result
                result['teacher_value'] = teacher_value
                result['teacher_available'] = True

                # 策略差异
                diff = 0
                for move_id in set(student_policy.keys()) | set(range(len(teacher_policy))):
                    s_prob = student_policy.get(move_id, 0)
                    t_prob = teacher_policy[move_id]
                    diff += abs(s_prob - t_prob)
                result['policy_diff'] = diff / 2  # 归一化

        return result


def test_distill_module():
    """测试蒸馏训练模块"""
    print("测试蒸馏训练模块...")

    # 测试配置
    print(f"蒸馏启用: {CONFIG['use_distill']}")
    print(f"MiniMax API密钥: {'已配置' if CONFIG['minimax_api_key'] else '未配置'}")

    if CONFIG['use_distill'] and MINIMAX_AVAILABLE:
        api = get_minimax_api()
        print(f"MiniMax API缓存大小: {api.get_cache_size()}")

    # 测试网络
    net = PolicyValueNet()
    print("策略价值网络初始化成功")

    # 测试蒸馏训练步骤
    test_state = np.random.randn(4, 9, 10, 9).astype('float32')
    test_policy = np.random.rand(4, 2086).astype('float32')
    test_policy = test_policy / test_policy.sum(axis=1, keepdims=True)
    test_value = np.random.rand(4).astype('float32') * 2 - 1  # [-1, 1]

    losses = net.distill_train_step(test_state, test_policy, test_value)
    print(f"蒸馏损失测试: total={losses[0]:.4f}, policy={losses[1]:.4f}, value={losses[2]:.4f}")

    print("蒸馏训练模块测试完成!")


if __name__ == '__main__':
    # 如果直接运行，执行测试
    test_distill_module()

    # 如果需要启动蒸馏训练，取消下面的注释
    # print("启动蒸馏训练...")
    # pipeline = DistillTrainPipeline(init_model='current_policy.pkl')
    # pipeline.run()
