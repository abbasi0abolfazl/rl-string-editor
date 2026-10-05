import numpy as np
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from rl_agent import StringMetricRLAgent
from rl import AdvancedArabicTextEnv, ExpertArabicEditor
import Levenshtein

class RLTextEditorEvaluator:
    """ارزیابی‌کننده ویرایشگر RL"""
    
    def __init__(self, model_path: str = None, agent_type: str = 'dqn'):
        self.agent_type = agent_type
        
        # بارگذاری عامل
        self.agent = StringMetricRLAgent(agent_type=agent_type)
        if model_path:
            self.agent.load(model_path)
    
    def evaluate_single(self, initial_text: str, target_text: str, 
                       max_edits: int = 15, use_agent: bool = True) -> Dict:
        """ارزیابی روی یک مثال"""
        env = AdvancedArabicTextEnv(initial_text, target_text, max_edits)
        
        if use_agent:
            # استفاده از عامل RL
            steps = 0
            episode_history = []
            
            while not env.done and steps < max_edits:
                # دریافت اعمال ممکن
                possible_actions_objs = env.get_advanced_actions(30)
                possible_actions = list(range(len(possible_actions_objs)))
                
                # دریافت بردار حالت
                state_vector = env.get_state_vector()
                
                # انتخاب عمل (فقط exploitation)
                epsilon_backup = self.agent.agent.epsilon
                self.agent.agent.epsilon = 0.0  # غیرفعال کردن اکتشاف
                
                action_idx = self.agent.get_action(state_vector, possible_actions)
                action = possible_actions_objs[action_idx]
                
                # بازگرداندن epsilon
                self.agent.agent.epsilon = epsilon_backup
                
                # اعمال عمل
                old_text = env.current_text
                new_text, reward, done, info = env.step(action)
                
                episode_history.append({
                    'step': steps,
                    'action': str(action),
                    'old_text': old_text,
                    'new_text': new_text,
                    'reward': reward,
                    'similarity': info['similarity']
                })
                
                steps += 1
        else:
            # استفاده از ویرایشگر اصلی (rule-based)
            editor = ExpertArabicEditor(use_ai=True)
            result = editor.expert_edit(initial_text, target_text, max_edits)
            
            # تبدیل به فرمت مشابه
            episode_history = []
            for i, step in enumerate(result.get('edit_history', [])):
                episode_history.append({
                    'step': i,
                    'action': str(step['action']),
                    'old_text': step.get('old_text', ''),
                    'new_text': step.get('new_text', ''),
                    'similarity': step.get('similarity', 0)
                })
        
        return {
            'initial_text': initial_text,
            'target_text': target_text,
            'final_text': env.current_text,
            'initial_similarity': env.initial_similarity,
            'final_similarity': env.current_similarity,
            'improvement': env.current_similarity - env.initial_similarity,
            'steps': len(episode_history),
            'success': env.current_similarity >= 0.95,
            'exact_match': env.current_text == target_text,
            'history': episode_history
        }
    
    def comparative_evaluation(self, test_cases: List[Dict]) -> Dict:
        """ارزیابی مقایسه‌ای بین RL و rule-based"""
        print("🧪 ارزیابی مقایسه‌ای")
        print("=" * 60)
        
        results_rl = []
        results_rule = []
        
        for i, test in enumerate(test_cases):
            print(f"\n📋 تست {i+1}/{len(test_cases)}: {test['name']}")
            print(f"   '{test['initial']}' → '{test['target']}'")
            
            # ارزیابی با RL
            print("   🤖 در حال ارزیابی با RL...")
            rl_result = self.evaluate_single(
                test['initial'], test['target'], use_agent=True
            )
            results_rl.append(rl_result)
            
            # ارزیابی با rule-based
            print("   ⚙️  در حال ارزیابی با rule-based...")
            rule_result = self.evaluate_single(
                test['initial'], test['target'], use_agent=False
            )
            results_rule.append(rule_result)
            
            # نمایش نتایج
            print(f"   📊 RL: شباهت={rl_result['final_similarity']:.4f}, "
                  f"مراحل={rl_result['steps']}, موفق={rl_result['success']}")
            print(f"   📊 Rule: شباهت={rule_result['final_similarity']:.4f}, "
                  f"مراحل={rule_result['steps']}, موفق={rule_result['success']}")
        
        # محاسبه آمار
        stats = self._calculate_statistics(results_rl, results_rule)
        
        # نمایش نمودار
        self._plot_comparison(results_rl, results_rule)
        
        return {
            'rl_results': results_rl,
            'rule_results': results_rule,
            'statistics': stats
        }
    
    def _calculate_statistics(self, results_rl: List, results_rule: List) -> Dict:
        """محاسبه آمار مقایسه‌ای"""
        stats = {}
        
        # نرخ موفقیت
        stats['rl_success_rate'] = sum(1 for r in results_rl if r['success']) / len(results_rl)
        stats['rule_success_rate'] = sum(1 for r in results_rule if r['success']) / len(results_rule)
        
        # تطابق دقیق
        stats['rl_exact_match_rate'] = sum(1 for r in results_rl if r['exact_match']) / len(results_rl)
        stats['rule_exact_match_rate'] = sum(1 for r in results_rule if r['exact_match']) / len(results_rule)
        
        # میانگین بهبود
        stats['rl_avg_improvement'] = np.mean([r['improvement'] for r in results_rl])
        stats['rule_avg_improvement'] = np.mean([r['improvement'] for r in results_rule])
        
        # میانگین مراحل
        stats['rl_avg_steps'] = np.mean([r['steps'] for r in results_rl])
        stats['rule_avg_steps'] = np.mean([r['steps'] for r in results_rule])
        
        # میانگین شباهت نهایی
        stats['rl_avg_final_similarity'] = np.mean([r['final_similarity'] for r in results_rl])
        stats['rule_avg_final_similarity'] = np.mean([r['final_similarity'] for r in results_rule])
        
        return stats
    
    def _plot_comparison(self, results_rl: List, results_rule: List):
        """نمودار مقایسه"""
        n_tests = len(results_rl)
        
        # داده‌ها برای نمودار
        indices = np.arange(n_tests)
        width = 0.35
        
        rl_similarities = [r['final_similarity'] for r in results_rl]
        rule_similarities = [r['final_similarity'] for r in results_rule]
        
        rl_steps = [r['steps'] for r in results_rl]
        rule_steps = [r['steps'] for r in results_rule]
        
        # نمودار 1: شباهت نهایی
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # نمودار میله‌ای شباهت
        axes[0].bar(indices - width/2, rl_similarities, width, 
                   label='RL Agent', color='blue', alpha=0.7)
        axes[0].bar(indices + width/2, rule_similarities, width, 
                   label='Rule-based', color='orange', alpha=0.7)
        axes[0].set_xlabel('شماره تست')
        axes[0].set_ylabel('شباهت نهایی')
        axes[0].set_title('مقایسه شباهت نهایی')
        axes[0].set_xticks(indices)
        axes[0].set_xticklabels([f'Test {i+1}' for i in range(n_tests)])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # نمودار 2: تعداد مراحل
        axes[1].bar(indices - width/2, rl_steps, width, 
                   label='RL Agent', color='green', alpha=0.7)
        axes[1].bar(indices + width/2, rule_steps, width, 
                   label='Rule-based', color='red', alpha=0.7)
        axes[1].set_xlabel('شماره تست')
        axes[1].set_ylabel('تعداد مراحل')
        axes[1].set_title('مقایسه کارایی (تعداد مراحل)')
        axes[1].set_xticks(indices)
        axes[1].set_xticklabels([f'Test {i+1}' for i in range(n_tests)])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rl_vs_rule_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # نمودار 3: Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(rl_similarities, rl_steps, c='blue', 
                  alpha=0.7, s=100, label='RL Agent', marker='o')
        ax.scatter(rule_similarities, rule_steps, c='orange', 
                  alpha=0.7, s=100, label='Rule-based', marker='s')
        
        ax.set_xlabel('شباهت نهایی')
        ax.set_ylabel('تعداد مراحل')
        ax.set_title('کارایی در مقابل دقت')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # اضافه کردن خطوط میانگین
        ax.axvline(x=np.mean(rl_similarities), color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(rule_similarities), color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=np.mean(rl_steps), color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=np.mean(rule_steps), color='orange', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('efficiency_vs_accuracy.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_benchmark(self):
        """اجرای بنچمارک کامل"""
        # مجموعه تست استاندارد
        test_cases = [
            {
                'name': 'حذف حرف "ه"',
                'initial': 'الحمد لل رب العالمين',
                'target': 'الحمد لله رب العالمين'
            },
            {
                'name': 'تکمیل آیه',
                'initial': 'بسم الله الرحمن',
                'target': 'بسم الله الرحمن الرحيم'
            },
            {
                'name': 'اصلاح همزه',
                'initial': 'كتابت',
                'target': 'كتابة'
            },
            {
                'name': 'حذف کاراکتر اضافی',
                'initial': 'اللهه اكبر',
                'target': 'الله أكبر'
            },
            {
                'name': 'اصلاح فاصله',
                'initial': 'بسم الله الرحم ن',
                'target': 'بسم الله الرحمن'
            },
            {
                'name': 'ترکیب چند خطا',
                'initial': 'صلاه الجمعه',
                'target': 'صلاة الجمعة'
            },
            {
                'name': 'اصلاح ياء',
                'initial': 'علي',
                'target': 'على'
            },
            {
                'name': 'بدون تغییر',
                'initial': 'محمد',
                'target': 'محمد'
            }
        ]
        
        # اجرای ارزیابی
        results = self.comparative_evaluation(test_cases)
        
        # نمایش آمار نهایی
        print("\n" + "=" * 60)
        print("📊 نتایج بنچمارک")
        print("=" * 60)
        
        stats = results['statistics']
        
        print(f"\n🤖 RL Agent:")
        print(f"   نرخ موفقیت: {stats['rl_success_rate']:.1%}")
        print(f"   نرخ تطابق دقیق: {stats['rl_exact_match_rate']:.1%}")
        print(f"   میانگین بهبود: {stats['rl_avg_improvement']:.4f}")
        print(f"   میانگین مراحل: {stats['rl_avg_steps']:.1f}")
        print(f"   میانگین شباهت نهایی: {stats['rl_avg_final_similarity']:.4f}")
        
        print(f"\n⚙️  Rule-based Editor:")
        print(f"   نرخ موفقیت: {stats['rule_success_rate']:.1%}")
        print(f"   نرخ تطابق دقیق: {stats['rule_exact_match_rate']:.1%}")
        print(f"   میانگین بهبود: {stats['rule_avg_improvement']:.4f}")
        print(f"   میانگین مراحل: {stats['rule_avg_steps']:.1f}")
        print(f"   میانگین شباهت نهایی: {stats['rule_avg_final_similarity']:.4f}")
        
        print(f"\n📈 مقایسه:")
        print(f"   تفاوت در نرخ موفقیت: {(stats['rl_success_rate'] - stats['rule_success_rate'])*100:+.1f}%")
        print(f"   تفاوت در میانگین مراحل: {stats['rl_avg_steps'] - stats['rule_avg_steps']:+.1f}")
        print(f"   تفاوت در میانگین شباهت: {stats['rl_avg_final_similarity'] - stats['rule_avg_final_similarity']:+.4f}")
        
        return results


def main():
    """تابع اصلی ارزیابی"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ارزیابی ویرایشگر RL')
    parser.add_argument('--model', type=str, required=True, help='مسیر مدل آموزش دیده')
    parser.add_argument('--agent', type=str, default='dqn', 
                       choices=['qlearning', 'dqn'], help='نوع عامل')
    parser.add_argument('--benchmark', action='store_true', help='اجرای بنچمارک کامل')
    
    args = parser.parse_args()
    
    # ایجاد ارزیابی‌کننده
    evaluator = RLTextEditorEvaluator(model_path=args.model, agent_type=args.agent)
    
    if args.benchmark:
        # اجرای بنچمارک
        results = evaluator.run_benchmark()
        
        # ذخیره نتایج
        with open('benchmark_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 نتایج در 'benchmark_results.json' ذخیره شد")
    else:
        # ارزیابی تعاملی
        print("🎮 ارزیابی تعاملی")
        print("=" * 60)
        
        while True:
            try:
                initial = input("\n✏️ متن اولیه (یا 'خروج' برای پایان): ").strip()
                if initial.lower() in ['خروج', 'exit', 'quit']:
                    break
                
                target = input("🎯 متن هدف: ").strip()
                
                print("\n🔍 در حال ارزیابی...")
                result = evaluator.evaluate_single(initial, target, use_agent=True)
                
                print(f"\n📊 نتایج:")
                print(f"متن اولیه: {result['initial_text']}")
                print(f"متن هدف:   {result['target_text']}")
                print(f"نتیجه:     {result['final_text']}")
                print(f"شباهت اولیه: {result['initial_similarity']:.4f}")
                print(f"شباهت نهایی: {result['final_similarity']:.4f}")
                print(f"بهبود: {result['improvement']:+.4f}")
                print(f"تعداد مراحل: {result['steps']}")
                print(f"موفقیت: {'✅' if result['success'] else '❌'}")
                print(f"تطابق دقیق: {'✅' if result['exact_match'] else '❌'}")
                
                # نمایش تاریخچه
                if result['history']:
                    print(f"\n📜 تاریخچه ویرایش:")
                    for step in result['history'][-3:]:  # ۳ مرحله آخر
                        print(f"  • {step['action']}: شباهت={step['similarity']:.4f}")
                
            except KeyboardInterrupt:
                print("\n\n👋 پایان ارزیابی")
                break
            except Exception as e:
                print(f"❌ خطا: {e}")


if __name__ == "__main__":
    main()