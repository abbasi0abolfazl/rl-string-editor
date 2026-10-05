import Levenshtein
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
import random
from collections import Counter

# غیرفعال کردن هشدارها
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EditAction:
    """یک عمل ویرایش با اطلاعات کامل"""
    action_type: str  # 'insert', 'delete', 'replace', 'keep'
    position: int
    char: str = ''
    
    def __str__(self):
        if self.action_type == 'keep':
            return f"keep"
        return f"{self.action_type}@{self.position}:{self.char}"


class AdvancedArabicTextEnv:
    """محیط پیشرفته برای ویرایش متن عربی"""
    
    def __init__(self, initial_text: str, target_text: str, max_edits: int = 15):
        self.initial_text = initial_text
        self.current_text = initial_text
        self.target_text = target_text
        self.max_edits = max_edits
        self.edit_count = 0
        self.done = False
        
        # ایجاد لغتنامه پیشرفته
        self._build_advanced_vocabulary()
        
        # محاسبه معیارهای اولیه
        self.initial_similarity = self._calculate_similarity(initial_text, target_text)
        self.current_similarity = self.initial_similarity
        self.best_similarity = self.initial_similarity
        self.best_text = self.initial_text
        
        # حافظه
        self.history = []
        
        # تحلیل عمیق تفاوت‌ها
        self._deep_analyze_texts()
    
    def _build_advanced_vocabulary(self):
        """ایجاد لغتنامه پیشرفته"""
        # تمام کاراکترهای موجود در هر دو متن
        all_chars = set(self.initial_text + self.target_text)
        
        # کاراکترهای عربی رایج
        common_arabic = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 
                        'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 
                        'ن', 'ه', 'و', 'ي', 'ى', 'آ', 'أ', 'إ', 'ئ', 'ؤ', 'ة', ' ',
                        'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ']
        
        # ترکیب
        self.vocabulary = sorted(all_chars.union(set(common_arabic)))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)
    
    def _deep_analyze_texts(self):
        """تحلیل عمیق تفاوت‌ها"""
        # 1. عملیات Levenshtein
        self.edit_ops = Levenshtein.editops(self.initial_text, self.target_text)
        
        # 2. تحلیل کاراکتر به کاراکتر
        self.char_differences = []
        max_len = max(len(self.initial_text), len(self.target_text))
        
        for i in range(max_len):
            char_initial = self.initial_text[i] if i < len(self.initial_text) else None
            char_target = self.target_text[i] if i < len(self.target_text) else None
            
            if char_initial != char_target:
                self.char_differences.append({
                    'position': i,
                    'initial': char_initial,
                    'target': char_target,
                    'type': self._get_diff_type(char_initial, char_target)
                })
        
        # 3. شناسایی الگوهای رایج
        self.patterns = self._identify_patterns()
    
    def _get_diff_type(self, char1, char2):
        """تعیین نوع تفاوت"""
        if char1 is None:
            return 'insert'
        elif char2 is None:
            return 'delete'
        else:
            return 'replace'
    
    def _identify_patterns(self):
        """شناسایی الگوهای رایج خطا"""
        patterns = []
        
        # الگو: حذف همزه (ء)
        if 'ء' in self.target_text and 'ء' not in self.initial_text:
            patterns.append(('hamza_missing', 'همزه حذف شده'))
        
        # الگو: تبدیل ي به ى
        if 'ي' in self.initial_text and 'ى' in self.target_text:
            patterns.append(('ya_to_alif_maqsura', 'ي باید ى باشد'))
        
        # الگو: فاصله اضافی
        if '  ' in self.initial_text and ' ' in self.target_text:
            patterns.append(('extra_space', 'فاصله اضافی'))
        
        return patterns
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """محاسبه شباهت با چند معیار"""
        # 1. Levenshtein ratio
        lev_sim = Levenshtein.ratio(text1, text2)
        
        # 2. Jaro-Winkler (برای اسامی و کلمات کوتاه)
        jaro_sim = Levenshtein.jaro_winkler(text1, text2)
        
        # 3. Sequence ratio
        seq_sim = Levenshtein.seqratio(text1, text2)
        
        # ترکیب وزن‌دار
        combined = (lev_sim * 0.5 + jaro_sim * 0.3 + seq_sim * 0.2)
        
        return combined
    
    def get_state_vector(self) -> np.ndarray:
        """بردار حالت پیشرفته"""
        features = []
        
        # 1. معیارهای شباهت
        features.append(Levenshtein.ratio(self.current_text, self.target_text))
        features.append(Levenshtein.jaro_winkler(self.current_text, self.target_text))
        
        # 2. اطلاعات طول
        features.append(len(self.current_text) / max(len(self.target_text), 1))
        features.append(abs(len(self.current_text) - len(self.target_text)) / 10.0)
        
        # 3. پیشرفت
        features.append(self.best_similarity - self.initial_similarity)
        features.append(self.edit_count / self.max_edits)
        
        # 4. ویژگی‌های متن
        current_chars = Counter(self.current_text)
        target_chars = Counter(self.target_text)
        
        # شباهت توزیع کاراکترها
        common_chars = set(current_chars.keys()).union(set(target_chars.keys()))
        char_sim = 0
        for char in common_chars:
            count_current = current_chars.get(char, 0)
            count_target = target_chars.get(char, 0)
            char_sim += min(count_current, count_target) / max(max(count_current, count_target), 1)
        
        if common_chars:
            char_sim /= len(common_chars)
        
        features.append(char_sim)
        
        return np.array(features)
    
    def get_advanced_actions(self, num_actions: int = 20) -> List[EditAction]:
        """اعمال پیشرفته بر اساس تحلیل عمیق"""
        actions = []
        
        current_sim = self._calculate_similarity(self.current_text, self.target_text)
        if current_sim >= 0.99:
            return [EditAction('keep', 0)]
        
        # 1. اعمال مستقیم از تحلیل Levenshtein
        for op in self.edit_ops[:5]:  # فقط ۵ عملیات اول
            if op[0] == 'insert':
                pos = op[1]
                char = self.target_text[op[2]] if op[2] < len(self.target_text) else ' '
                if pos <= len(self.current_text):
                    actions.append(EditAction('insert', pos, char))
            
            elif op[0] == 'delete':
                pos = op[1]
                if 0 <= pos < len(self.current_text):
                    actions.append(EditAction('delete', pos))
            
            elif op[0] == 'replace':
                pos = op[1]
                char = self.target_text[op[2]] if op[2] < len(self.target_text) else ' '
                if 0 <= pos < len(self.current_text):
                    actions.append(EditAction('replace', pos, char))
        
        # 2. اعمال بر اساس تحلیل کاراکتر به کاراکتر
        for diff in self.char_differences[:5]:
            pos = diff['position']
            
            if diff['type'] == 'insert' and diff['target']:
                if pos <= len(self.current_text):
                    actions.append(EditAction('insert', pos, diff['target']))
            
            elif diff['type'] == 'delete':
                if 0 <= pos < len(self.current_text):
                    actions.append(EditAction('delete', pos))
            
            elif diff['type'] == 'replace' and diff['target']:
                if 0 <= pos < len(self.current_text):
                    actions.append(EditAction('replace', pos, diff['target']))
        
        # 3. اعمال ویژه برای الگوهای شناسایی شده
        for pattern_name, pattern_desc in self.patterns:
            if pattern_name == 'ya_to_alif_maqsura':
                # جایگزینی همه ي با ى
                for i, char in enumerate(self.current_text):
                    if char == 'ي':
                        actions.append(EditAction('replace', i, 'ى'))
            
            elif pattern_name == 'extra_space':
                # حذف فاصله‌های اضافی
                for i in range(len(self.current_text) - 1):
                    if self.current_text[i:i+2] == '  ':
                        actions.append(EditAction('delete', i))
        
        # 4. اعمال عمومی
        # درج کاراکترهای هدف در انتهای متن
        if len(self.current_text) < len(self.target_text):
            for i in range(len(self.current_text), len(self.target_text)):
                if i < len(self.target_text):
                    char = self.target_text[i]
                    actions.append(EditAction('insert', i, char))
        
        # حذف کاراکترهای اضافی از انتها
        if len(self.current_text) > len(self.target_text):
            for i in range(len(self.target_text), len(self.current_text)):
                actions.append(EditAction('delete', i))
        
        # 5. عمل keep (همیشه)
        actions.append(EditAction('keep', 0))
        
        # حذف duplicates و محدود کردن تعداد
        unique_actions = []
        seen = set()
        for action in actions:
            action_str = str(action)
            if action_str not in seen and len(unique_actions) < num_actions:
                seen.add(action_str)
                unique_actions.append(action)
        
        return unique_actions
    
    def apply_action(self, action: EditAction) -> str:
        """اعمال عمل با کنترل خطا"""
        if action.action_type == 'insert':
            pos = min(action.position, len(self.current_text))
            return self.current_text[:pos] + action.char + self.current_text[pos:]
        
        elif action.action_type == 'delete':
            if 0 <= action.position < len(self.current_text):
                return self.current_text[:action.position] + self.current_text[action.position+1:]
        
        elif action.action_type == 'replace':
            if 0 <= action.position < len(self.current_text):
                chars = list(self.current_text)
                chars[action.position] = action.char
                return ''.join(chars)
        
        return self.current_text
    
    def step(self, action: EditAction) -> Tuple[str, float, bool, Dict]:
        """گام در محیط"""
        if self.done or self.edit_count >= self.max_edits:
            return self.current_text, 0.0, True, {}
        
        # حالت قبلی
        old_text = self.current_text
        old_similarity = self._calculate_similarity(old_text, self.target_text)
        
        # اعمال عمل
        new_text = self.apply_action(action)
        self.current_text = new_text
        self.edit_count += 1
        
        # محاسبه جدید
        new_similarity = self._calculate_similarity(new_text, self.target_text)
        self.current_similarity = new_similarity
        
        # پاداش هوشمند
        improvement = new_similarity - old_similarity
        
        # پاداش پایه
        if improvement > 0.05:
            reward = improvement * 15.0
        elif improvement > 0:
            reward = improvement * 8.0
        elif improvement == 0:
            if action.action_type == 'keep':
                reward = 0.05
            else:
                reward = -0.1
        else:
            reward = improvement * 12.0
        
        # پاداش ویژه برای اعمال دقیق
        if action.action_type != 'keep':
            # بررسی آیا این عمل یکی از عملیات مورد نیاز است
            is_required = False
            for op in self.edit_ops:
                if (op[0] == action.action_type and 
                    op[1] == action.position and 
                    (action.action_type != 'delete' or True)):
                    is_required = True
                    break
            
            if is_required and improvement > 0:
                reward += 1.0
        
        # ذخیره تاریخچه
        self.history.append({
            'action': action,
            'old_text': old_text,
            'new_text': new_text,
            'reward': reward,
            'similarity': new_similarity,
            'improvement': improvement
        })
        
        # بروزرسانی بهترین
        if new_similarity > self.best_similarity:
            self.best_similarity = new_similarity
            self.best_text = new_text
        
        # پایان؟
        self.done = (
            new_similarity >= 0.99 or
            self.edit_count >= self.max_edits or
            (improvement <= 0 and self.edit_count >= 5)
        )
        
        # پاداش نهایی
        if new_similarity >= 0.99:
            reward += 10.0
        
        info = {
            'edit_count': self.edit_count,
            'similarity': new_similarity,
            'improvement': improvement,
            'best_similarity': self.best_similarity,
            'required_ops': len(self.edit_ops)
        }
        
        return new_text, reward, self.done, info
    
    def reset(self):
        """بازنشانی"""
        self.current_text = self.initial_text
        self.edit_count = 0
        self.done = False
        self.current_similarity = self.initial_similarity
        self.history = []
        self.best_similarity = self.initial_similarity
        self.best_text = self.initial_text
        return self.current_text


class ExpertArabicEditor:
    """ویرایشگر متخصص متن عربی"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.env = None
    
    def expert_edit(self, initial_text: str, target_text: str, max_edits: int = 15) -> Dict:
        """ویرایش متخصصانه"""
        print("=" * 60)
        print("ویرایشگر متخصص متن عربی")
        print("=" * 60)
        
        print(f"📝 متن اولیه: {initial_text}")
        print(f"🎯 متن هدف:   {target_text}")
        
        # ایجاد محیط
        self.env = AdvancedArabicTextEnv(initial_text, target_text, max_edits)
        
        # تحلیل اولیه
        initial_sim = Levenshtein.ratio(initial_text, target_text)
        print(f"🔢 شباهت اولیه: {initial_sim:.4f}")
        
        # نمایش تحلیل تفاوت‌ها
        self._display_analysis()
        
        # ویرایش هوشمند
        result = self._smart_editing()
        
        # نمایش نتایج
        self._display_results(result)
        
        return result
    
    def _display_analysis(self):
        """نمایش تحلیل تفاوت‌ها"""
        print(f"\n🔍 تحلیل تفاوت‌ها:")
        
        # عملیات Levenshtein
        if self.env.edit_ops:
            print(f"عملیات مورد نیاز ({len(self.env.edit_ops)}):")
            for i, op in enumerate(self.env.edit_ops[:5]):
                if op[0] == 'replace':
                    pos = op[1]
                    old_char = self.env.initial_text[pos] if pos < len(self.env.initial_text) else '?'
                    new_char = self.env.target_text[op[2]] if op[2] < len(self.env.target_text) else '?'
                    print(f"  {i+1}. جایگزینی در {pos}: '{old_char}' → '{new_char}'")
                elif op[0] == 'insert':
                    pos = op[1]
                    new_char = self.env.target_text[op[2]] if op[2] < len(self.env.target_text) else '?'
                    print(f"  {i+1}. درج در {pos}: '{new_char}'")
                elif op[0] == 'delete':
                    pos = op[1]
                    old_char = self.env.initial_text[pos] if pos < len(self.env.initial_text) else '?'
                    print(f"  {i+1}. حذف از {pos}: '{old_char}'")
        
        # الگوهای شناسایی شده
        if self.env.patterns:
            print(f"\n🎯 الگوهای شناسایی شده:")
            for pattern_name, pattern_desc in self.env.patterns:
                print(f"  • {pattern_desc}")
    
    def _smart_editing(self) -> Dict:
        """ویرایش هوشمند"""
        print(f"\n🤖 شروع ویرایش هوشمند...")
        
        # دریافت اعمال پیشرفته
        smart_actions = self.env.get_advanced_actions(25)
        print(f"📋 {len(smart_actions)} عمل پیشرفته شناسایی شد")
        
        # ویرایش تا رسیدن به هدف یا پایان ویرایش‌ها
        while not self.env.done and self.env.edit_count < self.env.max_edits:
            # انتخاب بهترین عمل
            best_action = None
            best_improvement = -float('inf')
            
            for action in smart_actions[:15]:  # فقط ۱۵ عمل اول برای کارایی
                simulated = self.env.apply_action(action)
                sim = Levenshtein.ratio(simulated, self.env.target_text)
                current_sim = Levenshtein.ratio(self.env.current_text, self.env.target_text)
                improvement = sim - current_sim
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = action
            
            # اعمال بهترین عمل
            if best_action and best_improvement > -0.1:  # حداقل بهبود قابل قبول
                print(f"  ✏️ گام {self.env.edit_count + 1}: {best_action} (بهبود: {best_improvement:+.4f})")
                self.env.step(best_action)
                
                # به‌روزرسانی لیست اعمال
                smart_actions = self.env.get_advanced_actions(20)
            else:
                print(f"  ⚠️  هیچ عمل بهبود‌دهنده‌ای یافت نشد")
                break
        
        # جمع‌آوری نتایج
        return {
            'initial_text': self.env.initial_text,
            'target_text': self.env.target_text,
            'edited_text': self.env.best_text,
            'initial_similarity': self.env.initial_similarity,
            'final_similarity': self.env.best_similarity,
            'improvement': self.env.best_similarity - self.env.initial_similarity,
            'edit_count': self.env.edit_count,
            'edit_history': self.env.history,
            'required_ops': len(self.env.edit_ops),
            'success': self.env.best_similarity >= 0.95
        }
    
    def _display_results(self, result: Dict):
        """نمایش نتایج"""
        print(f"\n🎯 نتایج نهایی:")
        print(f"📄 متن ویرایش شده: {result['edited_text']}")
        print(f"🏆 بهترین شباهت: {result['final_similarity']:.4f}")
        print(f"📈 بهبود: {result['improvement']:+.4f}")
        print(f"✏️ تعداد ویرایش‌ها: {result['edit_count']}")
        print(f"🎯 عملیات مورد نیاز: {result['required_ops']}")
        
        if result['success']:
            print(f"✅ موفقیت آمیز!")
        else:
            print(f"⚠️  نیاز به بهبود بیشتر")
        
        # نمایش تاریخچه
        if result['edit_history']:
            print(f"\n📜 تاریخچه ویرایش:")
            for i, step in enumerate(result['edit_history'][-5:]):  # آخرین ۵ مورد
                print(f"  {i+1}. {step['action']}: شباهت={step['similarity']:.4f}, بهبود={step['improvement']:+.4f}")


def run_comprehensive_tests():
    """اجرای تست‌های جامع"""
    print("🧪 تست‌های جامع ویرایشگر متخصص")
    print("=" * 60)
    
    # تست‌های بهبود یافته
    test_cases = [
        {
            'name': 'حذف حرف "ه"',
            'initial': 'الحمد لل رب العالمين',
            'target': 'الحمد لله رب العالمين',
            'description': 'اصلاح حذف ناخواسته'
        },
        {
            'name': 'تکمیل آیه',
            'initial': 'بسم الله الرحمن',
            'target': 'بسم الله الرحمن الرحيم',
            'description': 'تکمیل بخش پایانی'
        },
        {
            'name': 'اصلاح همزه',
            'initial': 'كتابت',
            'target': 'كتابة',
            'description': 'اصلاح همزه روی تاء'
        },
        {
            'name': 'حذف کاراکتر اضافی',
            'initial': 'اللهه اكبر',
            'target': 'الله أكبر',
            'description': 'حذف ه اضافی و اصلاح همزه'
        },
        {
            'name': 'اصلاح فاصله',
            'initial': 'بسم الله الرحم ن',
            'target': 'بسم الله الرحمن',
            'description': 'حذف فاصله اضافی'
        },
        {
            'name': 'ترکیب چند خطا',
            'initial': 'صلاه الجمعه',
            'target': 'صلاة الجمعة',
            'description': 'اصلاح همزه و تنوین'
        }
    ]
    
    editor = ExpertArabicEditor(use_ai=True)
    results = []
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"📋 تست: {test['name']}")
        print(f"📖 {test['description']}")
        print(f"{'='*60}")
        
        result = editor.expert_edit(
            initial_text=test['initial'],
            target_text=test['target'],
            max_edits=10
        )
        
        results.append({
            'test_info': test,
            **result
        })
        
        print("\n" + "-"*40)
    
    # خلاصه
    print("\n" + "=" * 60)
    print("📊 خلاصه نتایج")
    print("=" * 60)
    
    success_count = 0
    total_improvement = 0
    
    for i, res in enumerate(results):
        test = res['test_info']
        print(f"\n{i+1}. {test['name']}:")
        print(f"   📖 {test['description']}")
        print(f"   📝 '{test['initial']}' → '{res['edited_text']}'")
        print(f"   🎯 هدف: '{test['target']}'")
        print(f"   🔢 شباهت: {res['final_similarity']:.4f}")
        print(f"   📈 بهبود: {res['improvement']:+.4f}")
        print(f"   ✏️ ویرایش‌ها: {res['edit_count']}")
        
        if res['success']:
            success_count += 1
            print("   ✅ موفق")
        else:
            print("   ❌ ناموفق")
        
        total_improvement += res['improvement']
    
    # آمار
    success_rate = success_count / len(results) * 100
    avg_improvement = total_improvement / len(results)
    
    print(f"\n📈 آمار کلی:")
    print(f"   تعداد تست‌ها: {len(results)}")
    print(f"   موفقیت: {success_count}/{len(results)} ({success_rate:.1f}%)")
    print(f"   میانگین بهبود: {avg_improvement:+.4f}")
    
    # ارزیابی
    print(f"\n🎯 ارزیابی عملکرد:")
    if success_rate >= 80:
        print("   🏆 عملکرد عالی!")
    elif success_rate >= 60:
        print("   👍 عملکرد خوب")
    elif success_rate >= 40:
        print("   ⚠️  عملکرد متوسط")
    else:
        print("   🔧 نیاز به بهبود اساسی")
    
    return results


def interactive_demo():
    """دموی تعاملی"""
    print("🎮 دموی تعاملی ویرایشگر متن عربی")
    print("=" * 60)
    
    print("📚 مثال‌های آماده:")
    examples = [
        ("1", "صلاه", "صلاة", "اصلاح همزه"),
        ("2", "علي", "على", "اصلاح ياء"),
        ("3", "الرحمن الرحيم", "الرحمن الرحيم", "فاصله اضافی"),
        ("4", "كتابت", "كتابة", "همزه روی تاء"),
        ("5", "اللهه", "الله", "حرف اضافی")
    ]
    
    for num, wrong, correct, desc in examples:
        print(f"{num}. {desc}: '{wrong}' → '{correct}'")
    
    print("\n6. ✏️ وارد کردن مثال دلخواه")
    
    try:
        choice = input("\nانتخاب مثال (1-6): ")
    except:
        choice = "1"
    
    if choice == "6":
        print("\n✏️ وارد کردن مثال دلخواه:")
        wrong = input("متن مشکل‌دار: ").strip()
        correct = input("متن صحیح: ").strip()
        desc = "مثال دلخواه"
    else:
        # یافتن مثال انتخاب شده
        selected = None
        for num, wrong, correct, desc in examples:
            if num == choice:
                selected = (wrong, correct, desc)
                break
        
        if not selected:
            selected = examples[0]
        
        wrong, correct, desc = selected
    
    print(f"\n{'='*60}")
    print(f"🔧 مثال: {desc}")
    print(f"{'='*60}")
    
    editor = ExpertArabicEditor(use_ai=True)
    result = editor.expert_edit(wrong, correct, max_edits=8)
    
    # نمایش مقایسه
    print(f"\n📊 مقایسه نهایی:")
    print(f"متن اولیه: {wrong}")
    print(f"متن هدف:   {correct}")
    print(f"نتیجه:     {result['edited_text']}")
    
    if result['edited_text'] == correct:
        print("🎉 اصلاح کامل شد!")
    else:
        similarity = Levenshtein.ratio(result['edited_text'], correct)
        print(f"شباهت با متن هدف: {similarity:.2%}")
    
    return result


# اجرای اصلی
if __name__ == "__main__":
    print("🧠 ویرایشگر متخصص متن عربی v2.0")
    print("=" * 60)
    
    try:
        import Levenshtein
        
        print("🎮 انتخاب حالت:")
        print("1. دموی تعاملی")
        print("2. تست‌های جامع")
        print("3. تست سریع")
        
        try:
            choice = input("انتخاب کنید (1-3): ").strip()
        except:
            choice = "1"
        
        if choice == "1":
            print("\n🎮 اجرای دموی تعاملی...")
            result = interactive_demo()
            print(f"\n✅ دمو کامل شد!")
            
        elif choice == "2":
            print("\n🧪 اجرای تست‌های جامع...")
            results = run_comprehensive_tests()
            print(f"\n✅ تست‌ها کامل شدند!")
            
        elif choice == "3":
            print("\n⚡ تست سریع...")
            editor = ExpertArabicEditor()
            result = editor.expert_edit(
                initial_text="الحمد لل رب العالمين",
                target_text="الحمد لله رب العالمين",
                max_edits=5
            )
            print(f"\n✅ تست کامل شد!")
            
        else:
            print("\n🎮 اجرای دموی تعاملی...")
            result = interactive_demo()
            
    except ImportError as e:
        print(f"\n❌ خطا: {e}")
        print("\n📦 برای نصب کتابخانه:")
        print("pip install python-Levenshtein")