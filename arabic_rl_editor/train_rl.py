import numpy as np
import torch
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from rl_agent import StringMetricRLAgent
from rl import AdvancedArabicTextEnv, EditAction  # از کد اصلی شما

class RLTextEditorTrainer:
    """آموزش‌دهنده RL برای ویرایشگر متن"""
    
    def __init__(self, agent_type: str = 'dqn', config: Dict = None):
        self.agent_type = agent_type
        self.config = config or {
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'epsilon': 0.3,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 32,
            'episodes': 1000
        }
        
        self.agent = StringMetricRLAgent(agent_type=agent_type, config=self.config)
        self.training_history = []
    
    def create_training_data(self) -> List[Dict]:
        """ایجاد داده‌های آموزشی"""
        # مثال‌های آموزشی (میتواند از فایل بارگذاری شود)
        training_pairs = [
            # (متن اولیه, متن هدف, توضیح)
            ("الحمد لل رب العالمين", "الحمد لله رب العالمين", "حذف ه"),
            ("بسم الله الرحمن", "بسم الله الرحمن الرحيم", "تکمیل آیه"),
            ("كتابت", "كتابة", "اصلاح همزه"),
            ("اللهه اكبر", "الله أكبر", "حذف ه اضافی"),
            ("صلاه", "صلاة", "همزه روی الف"),
            ("علي", "على", "اصلاح ياء"),
            ("الرحمن الرحيم", "الرحمن الرحيم", "فاصله اضافی"),
            ("محمد", "محمد", "بدون تغییر"),
            ("قميص", "قميص", "بدون تغییر"),
            ("مكتبه", "مكتبة", "همزه روی تاء")
        ]
        
        # تبدیل به فرمت استاندارد
        data = []
        for wrong, correct, desc in training_pairs:
            data.append({
                'initial': wrong,
                'target': correct,
                'description': desc,
                'difficulty': len(Levenshtein.editops(wrong, correct))
            })
        
        # مرتب بر اساس سختی
        data.sort(key=lambda x: x['difficulty'])
        return data
    
    def train_episode(self, env: AdvancedArabicTextEnv, max_steps: int = 20) -> Dict:
        """آموزش یک اپیزود"""
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_history = []
        
        while not env.done and steps < max_steps:
            # دریافت اعمال ممکن
            possible_actions_objs = env.get_advanced_actions(30)
            possible_actions = list(range(len(possible_actions_objs)))
            
            # دریافت بردار حالت
            state_vector = env.get_state_vector()
            
            # انتخاب عمل
            action_idx = self.agent.get_action(state_vector, possible_actions)
            
            # اعمال عمل
            action = possible_actions_objs[action_idx]
            old_text = env.current_text
            new_text, reward, done, info = env.step(action)
            
            # محاسبه پاداش بر اساس متریک‌های رشته‌ای
            metric_reward = self.agent.calculate_string_metric_reward(
                old_text, new_text, env.target_text
            )
            
            # ترکیب پاداش‌ها
            combined_reward = reward * 0.7 + metric_reward * 0.3
            
            # بردار حالت بعدی
            next_state_vector = env.get_state_vector()
            next_possible_actions = list(range(len(env.get_advanced_actions(30))))
            
            # به‌روزرسانی عامل
            loss = self.agent.update(
                state_vector, action_idx, combined_reward,
                next_state_vector, done, next_possible_actions
            )
            
            # ذخیره اطلاعات
            total_reward += combined_reward
            episode_history.append({
                'step': steps,
                'action': str(action),
                'old_text': old_text,
                'new_text': new_text,
                'reward': combined_reward,
                'similarity': info['similarity'],
                'loss': loss if loss else 0
            })
            
            steps += 1
            state = new_text
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'final_similarity': env.current_similarity,
            'improvement': env.current_similarity - env.initial_similarity,
            'history': episode_history,
            'success': env.current_similarity >= 0.95
        }
    
    def train(self, num_episodes: int = None, save_path: str = None):
        """آموزش اصلی"""
        if num_episodes is None:
            num_episodes = self.config.get('episodes', 1000)
        
        print(f"🚀 شروع آموزش {self.agent_type.upper()} با {num_episodes} اپیزود")
        print("=" * 60)
        
        # داده‌های آموزشی
        training_data = self.create_training_data()
        
        # آمار آموزش
        episode_rewards = []
        episode_successes = []
        episode_steps = []
        
        for episode in range(num_episodes):
            # انتخاب تصادفی یک مثال آموزشی
            example = random.choice(training_data)
            
            # ایجاد محیط
            env = AdvancedArabicTextEnv(
                initial_text=example['initial'],
                target_text=example['target'],
                max_edits=15
            )
            
            # آموزش اپیزود
            result = self.train_episode(env)
            
            # ذخیره آمار
            episode_rewards.append(result['total_reward'])
            episode_successes.append(1 if result['success'] else 0)
            episode_steps.append(result['steps'])
            
            # نمایش پیشرفت
            if (episode + 1) % 50 == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                success_rate = np.mean(episode_successes[-50:]) if len(episode_successes) >= 50 else np.mean(episode_successes)
                
                print(f"📊 اپیزود {episode + 1:4d}/{num_episodes} | "
                      f"میانگین پاداش: {avg_reward:6.2f} | "
                      f"نرخ موفقیت: {success_rate:5.1%} | "
                      f"اپسیلون: {self.agent.agent.epsilon:.3f}")
            
            # ذخیره تاریخچه
            self.training_history.append({
                'episode': episode,
                'example': example,
                **result
            })
        
        # ذخیره مدل
        if save_path:
            self.agent.save(save_path)
            print(f"💾 مدل در '{save_path}' ذخیره شد")
        
        return self.training_history
    
    def plot_training_progress(self):
        """نمودار پیشرفت آموزش"""
        if not self.training_history:
            print("⚠️ تاریخچه آموزش موجود نیست")
            return
        
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['total_reward'] for h in self.training_history]
        similarities = [h['final_similarity'] for h in self.training_history]
        successes = [1 if h['success'] else 0 for h in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # نمودار پاداش
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.6)
        axes[0, 0].set_title('پاداش در هر اپیزود')
        axes[0, 0].set_xlabel('اپیزود')
        axes[0, 0].set_ylabel('پاداش')
        axes[0, 0].grid(True, alpha=0.3)
        
        # میانگین متحرک پاداش
        window = 50
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'میانگین {window}تایی')
            axes[0, 0].legend()
        
        # نمودار شباهت
        axes[0, 1].scatter(episodes, similarities, alpha=0.5, s=10)
        axes[0, 1].set_title('شباهت نهایی')
        axes[0, 1].set_xlabel('اپیزود')
        axes[0, 1].set_ylabel('شباهت')
        axes[0, 1].grid(True, alpha=0.3)
        
        # نرخ موفقیت متحرک
        axes[1, 0].plot(episodes, successes, 'g-', alpha=0.3)
        if len(successes) >= window:
            success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(episodes[window-1:], success_rate, 'g-', linewidth=2)
        axes[1, 0].set_title('نرخ موفقیت')
        axes[1, 0].set_xlabel('اپیزود')
        axes[1, 0].set_ylabel('موفقیت')
        axes[1, 0].grid(True, alpha=0.3)
        
        # هیستوگرام مراحل
        axes[1, 1].hist([h['steps'] for h in self.training_history], bins=20, alpha=0.7)
        axes[1, 1].set_title('توزیع تعداد مراحل')
        axes[1, 1].set_xlabel('تعداد مراحل')
        axes[1, 1].set_ylabel('فراوانی')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """تابع اصلی آموزش"""
    import argparse
    
    parser = argparse.ArgumentParser(description='آموزش RL برای ویرایش متن عربی')
    parser.add_argument('--agent', type=str, default='dqn', 
                       choices=['qlearning', 'dqn'], help='نوع عامل RL')
    parser.add_argument('--episodes', type=int, default=1000, help='تعداد اپیزودها')
    parser.add_argument('--save', type=str, default='arabic_editor_rl.pth', 
                       help='مسیر ذخیره مدل')
    parser.add_argument('--config', type=str, help='فایل پیکربندی JSON')
    
    args = parser.parse_args()
    
    # بارگذاری پیکربندی
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # ایجاد آموزش‌دهنده
    trainer = RLTextEditorTrainer(agent_type=args.agent, config=config)
    
    # آموزش
    history = trainer.train(num_episodes=args.episodes, save_path=args.save)
    
    # نمایش نمودار
    trainer.plot_training_progress()
    
    # خلاصه نتایج
    print("\n" + "=" * 60)
    print("📊 خلاصه آموزش")
    print("=" * 60)
    
    success_count = sum(1 for h in history if h['success'])
    avg_reward = np.mean([h['total_reward'] for h in history])
    avg_steps = np.mean([h['steps'] for h in history])
    avg_similarity = np.mean([h['final_similarity'] for h in history])
    
    print(f"تعداد اپیزودها: {len(history)}")
    print(f"نرخ موفقیت: {success_count}/{len(history)} ({success_count/len(history)*100:.1f}%)")
    print(f"میانگین پاداش: {avg_reward:.2f}")
    print(f"میانگین مراحل: {avg_steps:.1f}")
    print(f"میانگین شباهت نهایی: {avg_similarity:.4f}")
    
    return trainer


if __name__ == "__main__":
    main()