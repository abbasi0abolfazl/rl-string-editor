import numpy as np
import json
import pickle
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class RLExperimentLogger:
    """لاگر برای آزمایش‌های RL"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_data = {
            'experiment_name': experiment_name,
            'config': {},
            'training_history': [],
            'evaluation_results': [],
            'timestamps': []
        }
    
    def log_config(self, config: Dict):
        """ذخیره پیکربندی"""
        self.log_data['config'] = config
    
    def log_training_step(self, episode: int, data: Dict):
        """ذخیره مرحله آموزش"""
        self.log_data['training_history'].append({
            'episode': episode,
            **data
        })
    
    def log_evaluation(self, eval_data: Dict):
        """ذخیره نتایج ارزیابی"""
        self.log_data['evaluation_results'].append(eval_data)
    
    def save_logs(self, path: str = None):
        """ذخیره لاگ‌ها"""
        if path is None:
            path = f"{self.experiment_name}_logs.json"
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 لاگ‌ها در '{path}' ذخیره شد")
    
    def load_logs(self, path: str):
        """بارگذاری لاگ‌ها"""
        with open(path, 'r', encoding='utf-8') as f:
            self.log_data = json.load(f)
    
    def generate_report(self):
        """تولید گزارش از آزمایش"""
        if not self.log_data['training_history']:
            return "⚠️ داده‌ای برای گزارش وجود ندارد"
        
        # محاسبه آمار
        training_history = self.log_data['training_history']
        
        if training_history:
            final_episode = training_history[-1]
            episodes = len(training_history)
            
            # محاسبه آمار نهایی
            final_rewards = [h.get('total_reward', 0) for h in training_history[-100:]]
            final_successes = [1 if h.get('success', False) else 0 for h in training_history[-100:]]
            
            avg_final_reward = np.mean(final_rewards) if final_rewards else 0
            avg_success_rate = np.mean(final_successes) * 100 if final_successes else 0
            
            report = f"""
            📊 گزارش آزمایش: {self.experiment_name}
            {'=' * 50}
            
            ⚙️ پیکربندی:
            {json.dumps(self.log_data['config'], indent=2, ensure_ascii=False)}
            
            📈 آمار آموزش:
            - تعداد اپیزودها: {episodes}
            - میانگین پاداف نهایی (آخرین ۱۰۰): {avg_final_reward:.2f}
            - نرخ موفقیت نهایی (آخرین ۱۰۰): {avg_success_rate:.1f}%
            
            🎯 نتایج ارزیابی:
            """
            
            for i, eval_result in enumerate(self.log_data['evaluation_results'][-3:], 1):
                report += f"\n  {i}. {eval_result.get('test_name', f'تست {i}')}:\n"
                if 'statistics' in eval_result:
                    stats = eval_result['statistics']
                    report += f"     - نرخ موفقیت RL: {stats.get('rl_success_rate', 0)*100:.1f}%\n"
                    report += f"     - نرخ موفقیت Rule: {stats.get('rule_success_rate', 0)*100:.1f}%\n"
            
            return report
        
        return "⚠️ داده آموزشی وجود ندارد"


def analyze_edit_patterns(history: List[Dict]) -> Dict:
    """تحلیل الگوهای ویرایش از تاریخچه"""
    action_types = []
    positions = []
    improvements = []
    
    for step in history:
        if 'action' in step:
            action_str = str(step['action'])
            action_types.append(action_str.split('@')[0] if '@' in action_str else action_str)
        
        if 'improvement' in step:
            improvements.append(step['improvement'])
    
    # تحلیل فراوانی
    action_counter = {}
    for action in action_types:
        action_counter[action] = action_counter.get(action, 0) + 1
    
    return {
        'total_steps': len(history),
        'action_distribution': action_counter,
        'avg_improvement': np.mean(improvements) if improvements else 0,
        'successful_actions': sum(1 for imp in improvements if imp > 0),
        'unsuccessful_actions': sum(1 for imp in improvements if imp <= 0)
    }


def visualize_learning_curves(histories: Dict[str, List]):
    """نمایش نمودارهای یادگیری برای مقایسه"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, history) in enumerate(histories.items()):
        if history:
            episodes = [h['episode'] for h in history]
            rewards = [h.get('total_reward', 0) for h in history]
            
            # میانگین متحرک
            window = 50
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(episodes[window-1:], moving_avg, 
                        label=f'{name} (میانگین {window}تایی)',
                        color=colors[i % len(colors)], linewidth=2)
            else:
                plt.plot(episodes, rewards, label=name, 
                        color=colors[i % len(colors)], alpha=0.6)
    
    plt.xlabel('اپیزود')
    plt.ylabel('پاداش')
    plt.title('مقایسه منحنی یادگیری')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=150)
    plt.show()


def create_test_dataset(size: int = 100) -> List[Dict]:
    """ایجاد مجموعه داده تست"""
    # الگوهای خطای رایج در عربی
    error_patterns = [
        # (الگو اشتباه, الگو صحیح, توضیح)
        ("ت$", "ة$", "ت به جای ة"),
        ("ي$", "ى$", "ي به جای ى"),
        ("ء$", "ؤ$", "همزه نادرست"),
        ("  ", " ", "فاصله اضافی"),
        ("اللهه", "الله", "ه اضافی"),
        ("الرحمن", "الرحمن", "بدون تغییر"),
        ("صلاه", "صلاة", "همزه روی الف"),
        ("كتابت", "كتابة", "همزه روی تاء"),
        ("علي", "على", "يای مقصوره"),
        ("شئ", "شيء", "همزه ناقص"),
    ]
    
    # کلمات پایه
    base_words = [
        "الحمد", "الله", "رب", "العالمين", "الرحمن", "الرحيم",
        "بسم", "محمد", "علي", "فاطمة", "الحسن", "الحسين",
        "الكتاب", "القلم", "البيت", "المسجد", "المدرسة",
        "الصلاة", "الزكاة", "الصوم", "الحج", "الجنة", "النار"
    ]
    
    dataset = []
    
    for _ in range(size):
        # انتخاب تصادفی یک کلمه پایه
        base_word = np.random.choice(base_words)
        
        # اعمال یا عدم اعمال خطا
        if np.random.random() < 0.7:  # 70% مواقع خطا دارد
            pattern = np.random.choice(error_patterns)
            wrong_pattern, correct_pattern, description = pattern
            
            # تولید کلمه اشتباه
            if wrong_pattern.endswith('$'):
                char = wrong_pattern[0]
                wrong_word = base_word + char
                correct_word = base_word + correct_pattern[0]
            else:
                wrong_word = base_word.replace(correct_pattern, wrong_pattern)
                correct_word = base_word
        else:
            wrong_word = base_word
            correct_word = base_word
            description = "بدون خطا"
        
        dataset.append({
            'initial': wrong_word,
            'target': correct_word,
            'description': description,
            'difficulty': Levenshtein.distance(wrong_word, correct_word)
        })
    
    return dataset


# تابع اصلی برای اجرای آزمایش‌ها
def run_complete_experiment():
    """اجرای آزمایش کامل برای مقاله"""
    print("🔬 آزمایش کامل برای مقاله")
    print("=" * 60)
    
    # 1. ایجاد مجموعه داده
    print("📊 1. ایجاد مجموعه داده...")
    train_data = create_test_dataset(200)
    test_data = create_test_dataset(50)
    
    print(f"   داده آموزشی: {len(train_data)} نمونه")
    print(f"   داده تست: {len(test_data)} نمونه")
    
    # 2. آموزش مدل‌های مختلف
    print("\n🤖 2. آموزش مدل‌های RL...")
    
    models = {}
    
    # Q-Learning
    print("   آموزش Q-Learning...")
    trainer_q = RLTextEditorTrainer(agent_type='qlearning', 
                                   config={'episodes': 500})
    history_q = trainer_q.train(num_episodes=500, 
                               save_path='qlearning_model.pkl')
    models['qlearning'] = trainer_q.agent
    
    # DQN
    print("   آموزش DQN...")
    trainer_dqn = RLTextEditorTrainer(agent_type='dqn', 
                                     config={'episodes': 500})
    history_dqn = trainer_dqn.train(num_episodes=500, 
                                   save_path='dqn_model.pth')
    models['dqn'] = trainer_dqn.agent
    
    # 3. ارزیابی
    print("\n📈 3. ارزیابی مدل‌ها...")
    
    evaluator = RLTextEditorEvaluator()
    results = {}
    
    for name, agent in models.items():
        print(f"   ارزیابی {name}...")
        evaluator.agent = agent
        test_results = []
        
        for test in test_data[:20]:  # ۲۰ نمونه اول
            result = evaluator.evaluate_single(
                test['initial'], test['target'], use_agent=True
            )
            test_results.append(result)
        
        # محاسبه آمار
        success_rate = sum(1 for r in test_results if r['success']) / len(test_results)
        avg_steps = np.mean([r['steps'] for r in test_results])
        avg_similarity = np.mean([r['final_similarity'] for r in test_results])
        
        results[name] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_similarity': avg_similarity,
            'test_results': test_results
        }
    
    # 4. مقایسه با baseline
    print("\n⚖️ 4. مقایسه با rule-based baseline...")
    
    baseline_results = []
    for test in test_data[:20]:
        result = evaluator.evaluate_single(
            test['initial'], test['target'], use_agent=False
        )
        baseline_results.append(result)
    
    baseline_success = sum(1 for r in baseline_results if r['success']) / len(baseline_results)
    baseline_steps = np.mean([r['steps'] for r in baseline_results])
    baseline_similarity = np.mean([r['final_similarity'] for r in baseline_results])
    
    # 5. تولید گزارش
    print("\n📋 5. تولید گزارش نهایی...")
    
    report = f"""
    📊 گزارش آزمایش برای مقاله
    {'=' * 60}
    
    📁 داده‌ها:
    - داده آموزشی: {len(train_data)} نمونه
    - داده تست: {len(test_data)} نمونه
    
    🤖 نتایج مدل‌های RL:
    """
    
    for name, res in results.items():
        report += f"\n  {name.upper()}:"
        report += f"\n    - نرخ موفقیت: {res['success_rate']:.1%}"
        report += f"\n    - میانگین مراحل: {res['avg_steps']:.1f}"
        report += f"\n    - میانگین شباهت: {res['avg_similarity']:.4f}"
    
    report += f"""
    
    ⚙️ Baseline (Rule-based):
    - نرخ موفقیت: {baseline_success:.1%}
    - میانگین مراحل: {baseline_steps:.1f}
    - میانگین شباهت: {baseline_similarity:.4f}
    
    📈 بهترین عملکرد:
    """
    
    # یافتن بهترین مدل
    best_model = max(results.items(), key=lambda x: x[1]['success_rate'])
    improvement_over_baseline = best_model[1]['success_rate'] - baseline_success
    
    report += f"""
    - بهترین مدل: {best_model[0].upper()}
    - نرخ موفقیت: {best_model[1]['success_rate']:.1%}
    - بهبود نسبت به baseline: {improvement_over_baseline*100:+.1f}%
    - کارایی (مراحل کمتر بهتر است): {baseline_steps - best_model[1]['avg_steps']:+.1f}
    """
    
    # ذخیره گزارش
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n💾 گزارش در 'experiment_report.txt' ذخیره شد")
    
    # 6. ذخیره نتایج برای مقاله
    final_results = {
        'experiment_date': '2024',
        'models': results,
        'baseline': {
            'success_rate': baseline_success,
            'avg_steps': baseline_steps,
            'avg_similarity': baseline_similarity
        },
        'best_model': best_model[0],
        'improvement_over_baseline': improvement_over_baseline
    }
    
    with open('paper_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    return final_results


if __name__ == "__main__":
    run_complete_experiment()