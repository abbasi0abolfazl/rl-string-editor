import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Dict, Any
import Levenshtein

class QLearningAgent:
    """عامل Q-Learning برای ویرایش متن"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict = None):
        self.state_size = state_size
        self.action_size = action_size
        
        # پارامترهای Q-Learning
        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.epsilon = self.config.get('epsilon', 0.3)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        
        # Q-Table
        self.q_table = {}
        
        # تاریخچه برای تحلیل
        self.training_history = []
    
    def get_state_key(self, state: np.ndarray) -> str:
        """تبدیل بردار حالت به کلید برای Q-table"""
        # گرد کردن برای کاهش فضای حالت
        rounded_state = np.round(state, 2)
        return str(rounded_state.tolist())
    
    def get_action(self, state: np.ndarray, possible_actions: List[int]) -> int:
        """انتخاب عمل با استراتژی epsilon-greedy"""
        state_key = self.get_state_key(state)
        
        # اکتشاف با احتمال epsilon
        if np.random.rand() <= self.epsilon:
            return random.choice(possible_actions)
        
        # بهره‌برداری: انتخاب بهترین عمل
        if state_key in self.q_table:
            # فقط بین اعمال ممکن جستجو کن
            q_values = self.q_table[state_key]
            valid_q_values = {a: q_values[a] for a in possible_actions if a in q_values}
            
            if valid_q_values:
                # انتخاب عمل با بیشترین Q-value
                return max(valid_q_values, key=valid_q_values.get)
        
        # اگر حالت جدید است، عمل تصادفی
        return random.choice(possible_actions)
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool, next_possible_actions: List[int]):
        """به‌روزرسانی Q-table با Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # مقداردهی اولیه اگر حالت جدید است
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        
        # محاسبه target Q-value
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            # حداکثر Q-value برای حالت بعدی (فقط بین اعمال ممکن)
            if next_state_key in self.q_table:
                next_q_values = self.q_table[next_state_key]
                valid_next_q = {a: next_q_values[a] for a in next_possible_actions 
                               if a in next_q_values}
                max_next_q = max(valid_next_q.values()) if valid_next_q else 0
            else:
                max_next_q = 0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # به‌روزرسانی Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # ذخیره تاریخچه
        self.training_history.append({
            'state': state_key,
            'action': action,
            'reward': reward,
            'next_state': next_state_key,
            'q_value': self.q_table[state_key][action]
        })
        
        # کاهش epsilon
        if len(self.training_history) % 100 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path: str):
        """ذخیره مدل"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'config': self.config,
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, path: str):
        """بارگذاری مدل"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.config = data.get('config', {})
            self.epsilon = data.get('epsilon', self.epsilon)


class DQNAgent:
    """عامل Deep Q-Network برای ویرایش متن"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict = None):
        self.state_size = state_size
        self.action_size = action_size
        
        # پارامترها
        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.epsilon = self.config.get('epsilon', 0.3)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_size = self.config.get('memory_size', 1000)
        
        # شبکه‌های عصبی
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self._update_target_model()
        
        # بهینه‌ساز و تابع خطا
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # حافظه replay
        self.memory = deque(maxlen=self.memory_size)
        
        # تاریخچه
        self.training_history = []
        self.loss_history = []
    
    def _build_model(self) -> nn.Module:
        """ساخت شبکه عصبی"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.action_size)
        )
    
    def _update_target_model(self):
        """به‌روزرسانی شبکه هدف"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """تبدیل حالت به تنسور"""
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_action(self, state: np.ndarray, possible_actions: List[int]) -> int:
        """انتخاب عمل با epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(possible_actions)
        
        # پیش‌بینی Q-values
        state_tensor = self.get_state_tensor(state)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # فقط بین اعمال ممکن انتخاب کن
        valid_q_values = {a: q_values[a] for a in possible_actions}
        
        if valid_q_values:
            return max(valid_q_values, key=valid_q_values.get)
        
        return random.choice(possible_actions)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool, next_possible_actions: List[int]):
        """ذخیره تجربه در حافظه"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'next_possible_actions': next_possible_actions
        })
    
    def replay(self):
        """آموزش روی batch از تجربیات"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # نمونه‌گیری از حافظه
        batch = random.sample(self.memory, self.batch_size)
        
        # آماده‌سازی داده‌ها
        states = []
        targets = []
        
        for experience in batch:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']
            done = experience['done']
            next_possible_actions = experience['next_possible_actions']
            
            # تبدیل به تنسور
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # محاسبه target Q-value
            with torch.no_grad():
                current_q = self.model(state_tensor)
                next_q = self.target_model(next_state_tensor)
                
                # حداکثر Q-value برای اعمال ممکن بعدی
                if next_possible_actions:
                    next_q_valid = next_q[0][next_possible_actions]
                    max_next_q = torch.max(next_q_valid).item()
                else:
                    max_next_q = 0
            
            target_q = reward if done else reward + self.discount_factor * max_next_q
            
            # آماده‌سازی target vector
            target = current_q.clone()
            target[0][action] = target_q
            
            states.append(state_tensor)
            targets.append(target)
        
        # آموزش
        states_batch = torch.stack(states)
        targets_batch = torch.stack(targets)
        
        self.optimizer.zero_grad()
        outputs = self.model(states_batch)
        loss = self.criterion(outputs, targets_batch)
        loss.backward()
        self.optimizer.step()
        
        # ذخیره loss
        self.loss_history.append(loss.item())
        
        # کاهش epsilon
        if len(self.training_history) % 10 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, path: str):
        """ذخیره مدل"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'memory': list(self.memory)[-1000:],  # آخرین 1000 تجربه
            'loss_history': self.loss_history
        }, path)
    
    def load_model(self, path: str):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint.get('config', {})
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.loss_history = checkpoint.get('loss_history', [])
        
        # بارگذاری حافظه
        memory_data = checkpoint.get('memory', [])
        self.memory = deque(memory_data, maxlen=self.memory_size)


class StringMetricRLAgent:
    """عامل RL با هدایت متریک‌های رشته‌ای"""
    
    def __init__(self, agent_type: str = 'dqn', config: Dict = None):
        self.agent_type = agent_type
        self.config = config or {}
        
        # متریک‌های رشته‌ای برای پاداش
        self.metric_weights = {
            'levenshtein': 1.0,
            'jaro_winkler': 0.8,
            'sequence_ratio': 0.5,
            'hamming': 0.3
        }
        
        if agent_type == 'qlearning':
            self.agent = QLearningAgent(
                state_size=7,  # اندازه بردار حالت از محیط
                action_size=50,  # حداکثر تعداد اعمال
                config=self.config
            )
        else:  # dqn
            self.agent = DQNAgent(
                state_size=7,
                action_size=50,
                config=self.config
            )
    
    def calculate_string_metric_reward(self, old_text: str, new_text: str, 
                                      target_text: str) -> float:
        """محاسبه پاداش بر اساس متریک‌های رشته‌ای"""
        rewards = []
        
        # Levenshtein ratio improvement
        old_lev = Levenshtein.ratio(old_text, target_text)
        new_lev = Levenshtein.ratio(new_text, target_text)
        lev_improvement = new_lev - old_lev
        rewards.append(lev_improvement * self.metric_weights['levenshtein'] * 10)
        
        # Jaro-Winkler improvement
        old_jw = Levenshtein.jaro_winkler(old_text, target_text)
        new_jw = Levenshtein.jaro_winkler(new_text, target_text)
        jw_improvement = new_jw - old_jw
        rewards.append(jw_improvement * self.metric_weights['jaro_winkler'] * 8)
        
        # Sequence ratio improvement
        old_seq = Levenshtein.seqratio(old_text, target_text)
        new_seq = Levenshtein.seqratio(new_text, target_text)
        seq_improvement = new_seq - old_seq
        rewards.append(seq_improvement * self.metric_weights['sequence_ratio'] * 6)
        
        # Bonus for exact match
        if new_text == target_text:
            rewards.append(20.0)
        
        # Penalty for invalid edit
        if new_text == old_text:
            rewards.append(-0.5)
        
        # Penalty for making text worse
        if new_lev < old_lev - 0.1:
            rewards.append(-5.0)
        
        return sum(rewards)
    
    def get_action(self, state: np.ndarray, possible_actions: List[int]) -> int:
        """دریافت عمل از عامل"""
        return self.agent.get_action(state, possible_actions)
    
    def update(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool, next_possible_actions: List[int]):
        """به‌روزرسانی عامل"""
        if self.agent_type == 'qlearning':
            self.agent.update_q_table(state, action, reward, next_state, 
                                     done, next_possible_actions)
        else:
            self.agent.remember(state, action, reward, next_state, 
                               done, next_possible_actions)
            loss = self.agent.replay()
            
            # به‌روزرسانی دوره‌ای شبکه هدف
            if len(self.agent.training_history) % 100 == 0:
                self.agent._update_target_model()
            
            return loss
    
    def save(self, path: str):
        """ذخیره عامل"""
        self.agent.save_model(path)
    
    def load(self, path: str):
        """بارگذاری عامل"""
        self.agent.load_model(path)