"""
سیستم کامل ویرایش متن عربی با RL
ادغام کد اصلی با پیاده‌سازی RL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analiys import analyze_your_texts, ArabicTextAnalyzer
from rl import AdvancedArabicTextEnv, ExpertArabicEditor, EditAction
from rl_agent import StringMetricRLAgent
from train_rl import RLTextEditorTrainer
from evaluate_rl import RLTextEditorEvaluator
from utils import RLExperimentLogger, run_complete_experiment

class IntegratedArabicTextEditor:
    """ویرایشگر متون عربی یکپارچه با RL"""
    
    def __init__(self, use_rl: bool = True, rl_model_path: str = None):
        self.use_rl = use_rl
        self.rl_agent = None
        
        if use_rl and rl_model_path:
            self._load_rl_agent(rl_model_path)
    
    def _load_rl_agent(self, model_path: str):
        """بارگذاری عامل RL"""
        try:
            if model_path.endswith('.pth'):
                agent_type = 'dqn'
            else:
                agent_type = 'qlearning'
            
            self.rl_agent = StringMetricRLAgent(agent_type=agent_type)
            self.rl_agent.load(model_path)
            print(f"✅ عامل RL از '{model_path}' بارگذاری شد")
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری عامل RL: {e}")
            self.rl_agent = None
    
    def edit_text(self, initial_text: str, target_text: str, 
                  method: str = 'auto', max_edits: int = 15) -> Dict:
        """
        ویرایش متن با روش انتخابی
        
        پارامترها:
        -----------
        method: 'auto' | 'rl' | 'rule' | 'hybrid'
            روش ویرایش:
            - auto: انتخاب خودکار بهترین روش
            - rl: فقط از RL استفاده کند
            - rule: فقط از rule-based استفاده کند
            - hybrid: ترکیب هر دو
        """
        
        print(f"\n🔧 ویرایش متن:")
        print(f"   📝 اولیه: {initial_text}")
        print(f"   🎯 هدف:   {target_text}")
        
        # محاسبه شباهت اولیه
        initial_sim = Levenshtein.ratio(initial_text, target_text)
        print(f"   🔢 شباهت اولیه: {initial_sim:.4f}")
        
        if initial_sim >= 0.99:
            print("   ✅ متن از قبل مشابه است. نیازی به ویرایش نیست.")
            return {
                'edited_text': initial_text,
                'method': 'none',
                'improvement': 0,
                'success': True
            }
        
        # انتخاب روش
        if method == 'auto':
            # تصمیم‌گیری هوشمند
            if len(initial_text) > 20 and self.rl_agent:
                method = 'rl'  # برای متون طولانی از RL استفاده کن
            else:
                method = 'rule'  # برای متون کوتاه از rule-based
        
        if method == 'rl' and self.rl_agent:
            return self._edit_with_rl(initial_text, target_text, max_edits)
        elif method == 'rule':
            return self._edit_with_rule(initial_text, target_text, max_edits)
        elif method == 'hybrid':
            return self._edit_with_hybrid(initial_text, target_text, max_edits)
        else:
            print("⚠️ روش انتخابی در دسترس نیست. استفاده از rule-based")
            return self._edit_with_rule(initial_text, target_text, max_edits)
    
    def _edit_with_rl(self, initial_text: str, target_text: str, 
                     max_edits: int) -> Dict:
        """ویرایش با RL"""
        print("   🤖 در حال ویرایش با RL...")
        
        env = AdvancedArabicTextEnv(initial_text, target_text, max_edits)
        steps = 0
        
        while not env.done and steps < max_edits:
            # دریافت اعمال ممکن
            possible_actions_objs = env.get_advanced_actions(30)
            possible_actions = list(range(len(possible_actions_objs)))
            
            # دریافت بردار حالت
            state_vector = env.get_state_vector()
            
            # انتخاب عمل (فقط exploitation)
            epsilon_backup = self.rl_agent.agent.epsilon
            self.rl_agent.agent.epsilon = 0.0
            
            action_idx = self.rl_agent.get_action(state_vector, possible_actions)
            action = possible_actions_objs[action_idx]
            
            # بازگرداندن epsilon
            self.rl_agent.agent.epsilon = epsilon_backup
            
            # اعمال عمل
            _, _, done, info = env.step(action)
            steps += 1
            
            # نمایش پیشرفت
            if steps % 3 == 0:
                print(f"     ↪️  گام {steps}: شباهت={info['similarity']:.4f}")
        
        return {
            'edited_text': env.current_text,
            'method': 'rl',
            'initial_similarity': env.initial_similarity,
            'final_similarity': env.current_similarity,
            'improvement': env.current_similarity - env.initial_similarity,
            'steps': steps,
            'success': env.current_similarity >= 0.95,
            'exact_match': env.current_text == target_text
        }
    
    def _edit_with_rule(self, initial_text: str, target_text: str, 
                       max_edits: int) -> Dict:
        """ویرایش با rule-based"""
        print("   ⚙️  در حال ویرایش با rule-based...")
        
        editor = ExpertArabicEditor(use_ai=True)
        result = editor.expert_edit(initial_text, target_text, max_edits)
        
        return {
            'edited_text': result['edited_text'],
            'method': 'rule',
            'initial_similarity': result['initial_similarity'],
            'final_similarity': result['final_similarity'],
            'improvement': result['improvement'],
            'steps': result['edit_count'],
            'success': result['success'],
            'exact_match': result['edited_text'] == target_text
        }
    
    def _edit_with_hybrid(self, initial_text: str, target_text: str, 
                         max_edits: int) -> Dict:
        """ویرایش ترکیبی"""
        print("   🔄 در حال ویرایش ترکیبی...")
        
        # ابتدا با rule-based
        rule_result = self._edit_with_rule(initial_text, target_text, max_edits//2)
        
        # اگر موفق نشد، با RL ادامه بده
        if not rule_result['success'] and self.rl_agent:
            print("   🔄 ادامه با RL...")
            rl_result = self._edit_with_rl(
                rule_result['edited_text'], 
                target_text, 
                max_edits//2
            )
            
            # ترکیب نتایج
            return {
                'edited_text': rl_result['edited_text'],
                'method': 'hybrid',
                'initial_similarity': rule_result['initial_similarity'],
                'final_similarity': rl_result['final_similarity'],
                'improvement': rl_result['final_similarity'] - rule_result['initial_similarity'],
                'steps': rule_result['steps'] + rl_result['steps'],
                'success': rl_result['success'],
                'exact_match': rl_result['exact_match'],
                'rule_steps': rule_result['steps'],
                'rl_steps': rl_result['steps']
            }
        
        return rule_result
    
    def batch_edit(self, text_pairs: List[Tuple[str, str]], 
                   method: str = 'auto') -> List[Dict]:
        """ویرایش دسته‌ای متون"""
        results = []
        
        print(f"\n🔧 شروع ویرایش دسته‌ای ({len(text_pairs)} جفت متن)")
        print("=" * 60)
        
        for i, (initial, target) in enumerate(text_pairs, 1):
            print(f"\n📝 جفت {i}/{len(text_pairs)}:")
            result = self.edit_text(initial, target, method=method)
            results.append(result)
            
            # نمایش خلاصه
            print(f"   ✅ نتیجه: {result['edited_text']}")
            print(f"   📈 بهبود: {result['improvement']:+.4f}")
            print(f"   🎯 موفقیت: {'✅' if result['success'] else '❌'}")
        
        # خلاصه
        success_count = sum(1 for r in results if r['success'])
        exact_match_count = sum(1 for r in results if r['exact_match'])
        avg_improvement = np.mean([r['improvement'] for r in results])
        
        print(f"\n📊 خلاصه ویرایش دسته‌ای:")
        print(f"   تعداد: {len(results)}")
        print(f"   موفق: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"   تطابق دقیق: {exact_match_count} ({exact_match_count/len(results)*100:.1f}%)")
        print(f"   میانگین بهبود: {avg_improvement:.4f}")
        
        return results


def main():
    """برنامه اصلی"""
    print("🧠 ویرایشگر هوشمند متن عربی با RL")
    print("=" * 60)
    
    print("\n🎮 گزینه‌ها:")
    print("1. 🔧 ویرایش متن")
    print("2. 🤖 آموزش مدل RL")
    print("3. 📊 ارزیابی مدل")
    print("4. 🔬 آزمایش کامل برای مقاله")
    print("5. 📝 تحلیل شباهت متون")
    print("6. 🚪 خروج")
    
    try:
        choice = input("\nانتخاب کنید (1-6): ").strip()
    except:
        choice = "1"
    
    if choice == "1":
        # ویرایش متن
        editor = IntegratedArabicTextEditor(
            use_rl=True, 
            rl_model_path='dqn_model.pth'  # مدل پیش‌فرض
        )
        
        # دریافت ورودی
        print("\n✏️ وارد کردن متن:")
        initial = input("متن اولیه: ").strip()
        target = input("متن هدف: ").strip()
        
        print("\n🔄 روش ویرایش:")
        print("1. خودکار (پیشنهادی)")
        print("2. فقط RL")
        print("3. فقط rule-based")
        print("4. ترکیبی")
        
        method_choice = input("انتخاب کنید (1-4): ").strip()
        methods = {
            '1': 'auto',
            '2': 'rl', 
            '3': 'rule',
            '4': 'hybrid'
        }
        
        method = methods.get(method_choice, 'auto')
        
        # اجرای ویرایش
        result = editor.edit_text(initial, target, method=method)
        
        print(f"\n🎯 نتیجه نهایی:")
        print(f"متن ویرایش شده: {result['edited_text']}")
        print(f"متن هدف:        {target}")
        print(f"شباهت اولیه:    {result['initial_similarity']:.4f}")
        print(f"شباهت نهایی:    {result['final_similarity']:.4f}")
        print(f"بهبود:          {result['improvement']:+.4f}")
        print(f"تعداد مراحل:    {result['steps']}")
        print(f"روش:            {result['method']}")
        print(f"موفقیت:         {'✅' if result['success'] else '❌'}")
        print(f"تطابق دقیق:     {'✅' if result['exact_match'] else '❌'}")
        
    elif choice == "2":
        # آموزش مدل RL
        print("\n🤖 آموزش مدل RL")
        print("=" * 40)
        
        print("انتخاب نوع عامل:")
        print("1. Q-Learning (سریع‌تر، نیاز به حافظه کمتر)")
        print("2. DQN (قدرتمندتر، نیاز به GPU)")
        
        agent_choice = input("انتخاب کنید (1-2): ").strip()
        agent_type = 'qlearning' if agent_choice == '1' else 'dqn'
        
        print("\n⚙️ تنظیمات آموزش:")
        episodes = input(f"تعداد اپیزودها [پیش‌فرض: 1000]: ").strip()
        episodes = int(episodes) if episodes.isdigit() else 1000
        
        lr = input(f"نرخ یادگیری [پیش‌فرض: 0.001]: ").strip()
        lr = float(lr) if lr.replace('.', '', 1).isdigit() else 0.001
        
        # ایجاد آموزش‌دهنده
        config = {
            'learning_rate': lr,
            'episodes': episodes,
            'epsilon': 0.3,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
        
        trainer = RLTextEditorTrainer(agent_type=agent_type, config=config)
        
        # آموزش
        print(f"\n🚀 شروع آموزش {agent_type.upper()}...")
        history = trainer.train(num_episodes=episodes)
        
        # ذخیره مدل
        if agent_type == 'dqn':
            save_path = 'dqn_model.pth'
        else:
            save_path = 'qlearning_model.pkl'
        
        trainer.agent.save(save_path)
        
        print(f"\n✅ آموزش کامل شد!")
        print(f"💾 مدل در '{save_path}' ذخیره شد")
        
    elif choice == "3":
        # ارزیابی مدل
        print("\n📊 ارزیابی مدل RL")
        print("=" * 40)
        
        model_path = input("مسیر مدل [پیش‌فرض: dqn_model.pth]: ").strip()
        if not model_path:
            model_path = 'dqn_model.pth'
        
        agent_type = 'dqn' if model_path.endswith('.pth') else 'qlearning'
        
        print("\n🎯 نوع ارزیابی:")
        print("1. بنچمارک کامل")
        print("2. ارزیابی تعاملی")
        print("3. مقایسه با rule-based")
        
        eval_choice = input("انتخاب کنید (1-3): ").strip()
        
        evaluator = RLTextEditorEvaluator(model_path=model_path, agent_type=agent_type)
        
        if eval_choice == '1':
            # بنچمارک کامل
            results = evaluator.run_benchmark()
            
        elif eval_choice == '2':
            # ارزیابی تعاملی
            print("\n🎮 ارزیابی تعاملی")
            print("-" * 40)
            
            while True:
                initial = input("\n✏️ متن اولیه (یا 'خروج'): ").strip()
                if initial.lower() in ['خروج', 'exit', 'quit']:
                    break
                
                target = input("🎯 متن هدف: ").strip()
                
                result = evaluator.evaluate_single(initial, target, use_agent=True)
                
                print(f"\n📊 نتایج:")
                print(f"متن اولیه: {result['initial_text']}")
                print(f"متن هدف:   {result['target_text']}")
                print(f"نتیجه:     {result['final_text']}")
                print(f"شباهت اولیه: {result['initial_similarity']:.4f}")
                print(f"شباهت نهایی: {result['final_similarity']:.4f}")
                print(f"بهبود: {result['improvement']:+.4f}")
                print(f"موفقیت: {'✅' if result['success'] else '❌'}")
        
        elif eval_choice == '3':
            # مقایسه با rule-based
            print("\n⚖️ مقایسه RL با rule-based")
            print("-" * 40)
            
            initial = input("✏️ متن اولیه: ").strip()
            target = input("🎯 متن هدف: ").strip()
            
            # ارزیابی RL
            print("\n🤖 ارزیابی با RL...")
            rl_result = evaluator.evaluate_single(initial, target, use_agent=True)
            
            # ارزیابی rule-based
            print("\n⚙️ ارزیابی با rule-based...")
            rule_result = evaluator.evaluate_single(initial, target, use_agent=False)
            
            print(f"\n📊 مقایسه:")
            print(f"{'':20} {'🤖 RL':10} {'⚙️ Rule':10}")
            print(f"{'='*40}")
            print(f"{'متن نهایی:':20} {rl_result['final_text']:10}")
            print(f"{'':20} {rule_result['final_text']:10}")
            print(f"{'شباهت نهایی:':20} {rl_result['final_similarity']:.4f}      {rule_result['final_similarity']:.4f}")
            print(f"{'تعداد مراحل:':20} {rl_result['steps']:10} {rule_result['steps']:10}")
            print(f"{'موفقیت:':20} {'✅' if rl_result['success'] else '❌':10} {'✅' if rule_result['success'] else '❌':10}")
            
    elif choice == "4":
        # آزمایش کامل برای مقاله
        print("\n🔬 اجرای آزمایش کامل برای مقاله")
        print("=" * 60)
        
        results = run_complete_experiment()
        
        print("\n✅ آزمایش کامل شد!")
        print(f"📋 گزارش در 'experiment_report.txt' ذخیره شد")
        print(f"📊 نتایج در 'paper_results.json' ذخیره شد")
        
    elif choice == "5":
        # تحلیل شباهت متون
        print("\n📝 تحلیل شباهت متون")
        print("=" * 40)
        
        print("📥 وارد کردن متون (خاتمه با خط خالی):")
        texts = []
        i = 1
        
        while True:
            text = input(f"متن {i}: ").strip()
            if not text:
                break
            texts.append(text)
            i += 1
        
        if len(texts) < 2:
            print("❌ حداقل ۲ متن نیاز است")
            return
        
        # تحلیل
        analyzer = ArabicTextAnalyzer(
            texts=texts,
            preprocess=True,
            use_bert=True,
            use_tfidf=True
        )
        
        results = analyzer.analyze_similarities()
        
        # نمایش نتایج
        print(f"\n📊 نتایج تحلیل برای {len(texts)} متن:")
        
        # نمایش ماتریس شباهت
        if 'levenshtein' in results:
            print(f"\n📈 ماتریس Levenshtein:")
            print(results['levenshtein']['matrix'])
            
        # نمایش مقایسات
        if 'comparisons' in results:
            print(f"\n🔍 ۵ مقایسه برتر:")
            sorted_comps = sorted(
                results['comparisons'],
                key=lambda x: x.get('bert_similarity', x['levenshtein_ratio']),
                reverse=True
            )[:5]
            
            for i, comp in enumerate(sorted_comps, 1):
                print(f"\n{i}. {comp['text1_id']} ↔ {comp['text2_id']}:")
                print(f"   Levenshtein: {comp['levenshtein_ratio']:.4f}")
                if comp.get('bert_similarity'):
                    print(f"   BERT: {comp['bert_similarity']:.4f}")
                if comp.get('tfidf_similarity'):
                    print(f"   TF-IDF: {comp['tfidf_similarity']:.4f}")
    
    elif choice == "6":
        print("\n👋 خداحافظ!")
        return
    
    else:
        print("\n❌ انتخاب نامعتبر")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 برنامه متوقف شد")
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        import traceback
        traceback.print_exc()