import Levenshtein
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import re
from collections import Counter
import warnings

# کتابخانه‌های مورد نیاز
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, logging
import torch

# کاهش هشدارها
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

@dataclass
class TextSimilarityResult:
    """کلاس برای نگهداری نتایج تحلیل شباهت متن"""
    
    text1_id: str
    text2_id: str
    
    # پارامترهای اصلی
    levenshtein_distance: int
    levenshtein_ratio: float
    bert_similarity: Optional[float] = None
    tfidf_similarity: Optional[float] = None
    
    # پارامترهای اضافی
    jaro_similarity: Optional[float] = None
    jaro_winkler_similarity: Optional[float] = None
    hamming_distance: Optional[int] = None
    sequence_ratio: Optional[float] = None
    set_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """تبدیل نتایج به دیکشنری"""
        result = {
            'text1_id': self.text1_id,
            'text2_id': self.text2_id,
            'levenshtein_distance': self.levenshtein_distance,
            'levenshtein_ratio': self.levenshtein_ratio,
            'bert_similarity': self.bert_similarity,
            'tfidf_similarity': self.tfidf_similarity
        }
        
        # افزودن مقادیر اختیاری اگر موجود باشند
        optional_fields = ['jaro_similarity', 'jaro_winkler_similarity', 
                          'hamming_distance', 'sequence_ratio', 'set_ratio']
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
                
        return result


class BertEmbeddingGenerator:
    """کلاس برای تولید Embedding با مدل‌های BERT"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-multilingual-cased',
                 device: str = None,
                 max_length: int = 128):
        
        self.model_name = model_name
        self.max_length = max_length
        
        # تعیین دستگاه پردازش
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"📥 بارگذاری مدل BERT: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("✅ مدل BERT با موفقیت بارگذاری شد")
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل BERT: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """تولید Embedding برای دسته‌ای از متون"""
        with torch.no_grad():
            # توکنایز کردن همه متون
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            ).to(self.device)
            
            # محاسبه خروجی
            outputs = self.model(**inputs)
            
            # استفاده از میانگین token embeddings
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # محاسبه میانگین وزنی
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # تبدیل به numpy و نرمال‌سازی
            embeddings_np = embeddings.cpu().numpy()
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings_normalized = embeddings_np / norms
            
            return embeddings_normalized


class ArabicTextAnalyzer:
    """
    کلاس اصلی برای تحلیل متون عربی - نسخه تصحیح شده
    """
    
    def __init__(self, 
                 texts: List[str], 
                 text_ids: Optional[List[str]] = None,
                 preprocess: bool = True,
                 use_bert: bool = True,
                 bert_model: str = 'bert-base-multilingual-cased',
                 use_tfidf: bool = True,
                 max_texts: int = 50):
        
        # بررسی ورودی‌ها
        if not texts:
            raise ValueError("لیست متون نمی‌تواند خالی باشد")
        
        # محدود کردن تعداد متون
        if len(texts) > max_texts:
            print(f"⚠️ تعداد متون ({len(texts)}) از حد مجاز ({max_texts}) بیشتر است")
            print(f"📝 تنها {max_texts} متن اول پردازش می‌شود")
            texts = texts[:max_texts]
            if text_ids:
                text_ids = text_ids[:max_texts]
        
        self.raw_texts = texts
        self.preprocess = preprocess
        self.use_bert = use_bert
        self.use_tfidf = use_tfidf
        
        # تنظیم شناسه‌ها
        if text_ids is None:
            self.text_ids = [f"text_{i}" for i in range(len(texts))]
        else:
            if len(text_ids) != len(texts):
                raise ValueError(f"طول text_ids ({len(text_ids)}) باید با texts ({len(texts)}) برابر باشد")
            self.text_ids = text_ids
        
        # پیش‌پردازش متون
        print("🔄 در حال پیش‌پردازش متون...")
        if preprocess:
            self.processed_texts = self._preprocess_texts_safe(texts)
        else:
            self.processed_texts = texts
        
        # بررسی متون خالی
        self._check_empty_texts()
        
        # ذخیره نتایج
        self.similarity_results: Dict[Tuple[str, str], TextSimilarityResult] = {}
        
        # ماتریس‌های شباهت
        self.levenshtein_matrix: Optional[np.ndarray] = None
        self.bert_matrix: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        
        # مدل BERT
        self.bert_generator: Optional[BertEmbeddingGenerator] = None
        if use_bert:
            try:
                self.bert_generator = BertEmbeddingGenerator(
                    model_name=bert_model,
                    max_length=128
                )
            except Exception as e:
                print(f"⚠️ BERT غیرفعال شد: {e}")
                self.use_bert = False
        
        # TF-IDF Vectorizer
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        if use_tfidf:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=500,
                    min_df=1,
                    max_df=1.0,
                    ngram_range=(1, 1)
                )
            except Exception as e:
                print(f"⚠️ TF-IDF غیرفعال شد: {e}")
                self.use_tfidf = False
    
    def _preprocess_texts_safe(self, texts: List[str]) -> List[str]:
        """پیش‌پردازش متون - نسخه ایمن"""
        processed = []
        
        for idx, original_text in enumerate(texts):
            if not isinstance(original_text, str):
                text = str(original_text)
            else:
                text = original_text
            
            # ذخیره نسخه اصلی
            original_clean = text
            
            # 1. حذف BOM (Byte Order Mark) - کاراکتر \ufeff
            if text.startswith('\ufeff'):
                text = text[1:]
            
            # 2. حذف تگ‌های خاص
            text = re.sub(r'\{[^}]*\}', ' ', text)
            
            # 3. حذف خطوط جدید و فاصله‌های اضافی
            text = text.replace('\n', ' ').replace('\r', ' ')
            
            # 4. نگهداری حروف عربی و فاصله‌ها
            # گسترش محدوده کاراکترهای عربی
            text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF،؛؟]', ' ', text)
            
            # 5. نرمال‌سازی فاصله‌ها
            text = re.sub(r'\s+', ' ', text).strip()
            
            # اگر متن خالی شد، از نسخه اصلی استفاده کن (با حذف تگ‌ها)
            if not text.strip():
                print(f"⚠️ متن {idx} بعد از پیش‌پردازش خالی شد")
                # حذف تگ‌ها از متن اصلی
                text = re.sub(r'\{[^}]*\}', ' ', original_clean)
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF،؛؟]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                # اگر باز هم خالی بود، متن اصلی را برگردان
                if not text.strip():
                    text = original_clean.strip()
                    print(f"  استفاده از متن اصلی: '{text[:50]}...'")
            
            processed.append(text)
        
        return processed
    
    def _check_empty_texts(self):
        """بررسی و گزارش متون خالی"""
        empty_indices = []
        problematic_texts = []
        
        for i, text in enumerate(self.processed_texts):
            if not text or not text.strip():
                empty_indices.append(i)
                problematic_texts.append({
                    'index': i,
                    'text_id': self.text_ids[i],
                    'original_preview': self.raw_texts[i][:100] if len(self.raw_texts[i]) > 100 else self.raw_texts[i],
                    'processed': text
                })
        
        if empty_indices:
            print(f"⚠️ هشدار: {len(empty_indices)} متن خالی شناسایی شد")
            for i, problem in enumerate(problematic_texts[:3]):  # نمایش ۳ مورد اول
                print(f"  {i+1}. متن {problem['index']} (ID: {problem['text_id']})")
                print(f"     اصلی: '{problem['original_preview']}...'")
                print(f"     پردازش شده: '{problem['processed']}'")
            
            if len(problematic_texts) > 3:
                print(f"  ... و {len(problematic_texts) - 3} مورد دیگر")
        
        # حذف متون خالی از لیست
        if empty_indices:
            print(f"🗑️ حذف {len(empty_indices)} متن خالی از تحلیل")
            
            # حذف از لیست‌ها
            self.raw_texts = [text for i, text in enumerate(self.raw_texts) if i not in empty_indices]
            self.processed_texts = [text for i, text in enumerate(self.processed_texts) if i not in empty_indices]
            self.text_ids = [id_ for i, id_ in enumerate(self.text_ids) if i not in empty_indices]
            
            print(f"✅ تعداد متون باقی‌مانده: {len(self.processed_texts)}")
    
    def calculate_levenshtein_matrix(self) -> np.ndarray:
        """محاسبه ماتریس Levenshtein با کنترل خطا"""
        n = len(self.processed_texts)
        if n == 0:
            print("❌ هیچ متنی برای تحلیل وجود ندارد")
            return np.array([])
        
        matrix = np.zeros((n, n))
        
        print(f"🔢 در حال محاسبه ماتریس Levenshtein برای {n} متن...")
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                
                text1 = self.processed_texts[i]
                text2 = self.processed_texts[j]
                
                # بررسی متن‌های خالی
                if not text1.strip() or not text2.strip():
                    matrix[i, j] = 0.0
                    matrix[j, i] = 0.0
                    continue
                
                try:
                    ratio = Levenshtein.ratio(text1, text2)
                    matrix[i, j] = ratio
                    matrix[j, i] = ratio
                except Exception as e:
                    print(f"⚠️ خطا در محاسبه Levenshtein برای ({i},{j}): {e}")
                    print(f"  text1: '{text1[:50]}...'")
                    print(f"  text2: '{text2[:50]}...'")
                    matrix[i, j] = 0.0
                    matrix[j, i] = 0.0
        
        self.levenshtein_matrix = matrix
        
        # گزارش آماری
        if n > 1:
            triu_indices = np.triu_indices_from(matrix, k=1)
            values = matrix[triu_indices]
            print(f"📊 آمار Levenshtein: میانگین={np.mean(values):.4f}, حداقل={np.min(values):.4f}, حداکثر={np.max(values):.4f}")
        
        return matrix
    
    def calculate_bert_matrix(self) -> Optional[np.ndarray]:
        """محاسبه ماتریس شباهت BERT"""
        if not self.use_bert or self.bert_generator is None:
            return None
        
        n = len(self.processed_texts)
        if n == 0:
            return None
        
        print(f"🧠 در حال محاسبه ماتریس BERT برای {n} متن...")
        
        try:
            # تولید Embedding برای همه متون
            embeddings = self.bert_generator.get_embeddings_batch(self.processed_texts)
            
            # محاسبه ماتریس شباهت کسینوسی
            matrix = cosine_similarity(embeddings)
            
            # اطمینان از مقادیر معقول
            matrix = np.clip(matrix, -1.0, 1.0)
            
            self.bert_matrix = matrix
            
            # گزارش آماری
            if n > 1:
                triu_indices = np.triu_indices_from(matrix, k=1)
                values = matrix[triu_indices]
                print(f"📊 آمار BERT: میانگین={np.mean(values):.4f}, حداقل={np.min(values):.4f}, حداکثر={np.max(values):.4f}")
            
            return matrix
            
        except Exception as e:
            print(f"❌ خطا در محاسبه BERT: {e}")
            self.use_bert = False
            return None
    
    def calculate_tfidf_matrix(self) -> Optional[np.ndarray]:
        """محاسبه ماتریس شباهت TF-IDF"""
        if not self.use_tfidf or self.tfidf_vectorizer is None:
            return None
        
        n = len(self.processed_texts)
        if n == 0:
            return None
        
        print(f"📊 در حال محاسبه ماتریس TF-IDF برای {n} متن...")
        
        try:
            # بررسی آیا متن‌ها برای TF-IDF مناسب هستند
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(self.processed_texts):
                words = text.split()
                if len(words) >= 2:  # حداقل ۲ کلمه داشته باشد
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if len(valid_texts) < 2:
                print("⚠️ متون کافی برای TF-IDF وجود ندارد")
                return None
            
            # تبدیل متون به بردارهای TF-IDF
            tfidf_vectors = self.tfidf_vectorizer.fit_transform(valid_texts)
            
            # محاسبه شباهت کسینوسی
            matrix_small = cosine_similarity(tfidf_vectors)
            
            # ایجاد ماتریس کامل
            matrix = np.zeros((n, n))
            for idx_i, i in enumerate(valid_indices):
                for idx_j, j in enumerate(valid_indices):
                    if i <= j:
                        matrix[i, j] = matrix_small[idx_i, idx_j]
                        matrix[j, i] = matrix_small[idx_i, idx_j]
            
            self.tfidf_matrix = matrix
            
            # گزارش آماری
            if len(valid_indices) > 1:
                values = []
                for idx_i, i in enumerate(valid_indices):
                    for idx_j, j in enumerate(valid_indices):
                        if i < j:
                            values.append(matrix[i, j])
                
                if values:
                    print(f"📊 آمار TF-IDF: میانگین={np.mean(values):.4f}, حداقل={np.min(values):.4f}, حداکثر={np.max(values):.4f}")
            
            return matrix
            
        except Exception as e:
            print(f"❌ خطا در محاسبه TF-IDF: {e}")
            self.use_tfidf = False
            return None
    
    def analyze_similarities(self) -> Dict:
        """تحلیل کامل شباهت‌ها"""
        print("\n" + "=" * 60)
        print("شروع تحلیل کامل")
        print("=" * 60)
        
        results = {
            'text_count': len(self.processed_texts),
            'text_ids': self.text_ids,
            'processed_texts_preview': [text[:100] + '...' if len(text) > 100 else text for text in self.processed_texts]
        }
        
        if len(self.processed_texts) < 2:
            print("❌ حداقل ۲ متن برای تحلیل لازم است")
            return results
        
        # 1. محاسبه Levenshtein
        lev_matrix = self.calculate_levenshtein_matrix()
        if lev_matrix.size > 0:
            results['levenshtein'] = {
                'matrix': lev_matrix,
                'mean': np.mean(lev_matrix[np.triu_indices_from(lev_matrix, k=1)]) if lev_matrix.size > 1 else 0,
                'min': np.min(lev_matrix[np.triu_indices_from(lev_matrix, k=1)]) if lev_matrix.size > 1 else 0,
                'max': np.max(lev_matrix[np.triu_indices_from(lev_matrix, k=1)]) if lev_matrix.size > 1 else 0
            }
        
        # 2. محاسبه BERT
        if self.use_bert:
            bert_matrix = self.calculate_bert_matrix()
            if bert_matrix is not None:
                results['bert'] = {
                    'matrix': bert_matrix,
                    'mean': np.mean(bert_matrix[np.triu_indices_from(bert_matrix, k=1)]),
                    'min': np.min(bert_matrix[np.triu_indices_from(bert_matrix, k=1)]),
                    'max': np.max(bert_matrix[np.triu_indices_from(bert_matrix, k=1)])
                }
        
        # 3. محاسبه TF-IDF
        if self.use_tfidf:
            tfidf_matrix = self.calculate_tfidf_matrix()
            if tfidf_matrix is not None:
                results['tfidf'] = {
                    'matrix': tfidf_matrix,
                    'mean': np.mean(tfidf_matrix[np.triu_indices_from(tfidf_matrix, k=1)]) if tfidf_matrix.size > 1 else 0,
                    'min': np.min(tfidf_matrix[np.triu_indices_from(tfidf_matrix, k=1)]) if tfidf_matrix.size > 1 else 0,
                    'max': np.max(tfidf_matrix[np.triu_indices_from(tfidf_matrix, k=1)]) if tfidf_matrix.size > 1 else 0
                }
        
        # 4. جمع‌آوری نتایج مقایسه
        comparison_results = []
        n = len(self.processed_texts)
        
        print(f"\n📈 در حال جمع‌آوری نتایج برای {n} متن...")
        
        for i in range(n):
            for j in range(i + 1, n):
                result = TextSimilarityResult(
                    text1_id=self.text_ids[i],
                    text2_id=self.text_ids[j],
                    levenshtein_distance=Levenshtein.distance(
                        self.processed_texts[i], 
                        self.processed_texts[j]
                    ),
                    levenshtein_ratio=lev_matrix[i, j]
                )
                
                # افزودن BERT similarity
                if 'bert' in results:
                    result.bert_similarity = results['bert']['matrix'][i, j]
                
                # افزودن TF-IDF similarity
                if 'tfidf' in results:
                    result.tfidf_similarity = results['tfidf']['matrix'][i, j]
                
                # ذخیره نتیجه
                self.similarity_results[(self.text_ids[i], self.text_ids[j])] = result
                comparison_results.append(result.to_dict())
        
        results['comparisons'] = comparison_results
        results['comparison_count'] = len(comparison_results)
        
        print(f"✅ تحلیل کامل شد. {len(comparison_results)} مقایسه انجام شد.")
        
        return results


# تابع اصلی برای تحلیل داده‌های شما
def analyze_your_texts(file_readeded: List[str]):
    """
    تحلیل متون خوانده شده از فایل
    
    پارامترها:
    ------------
    file_readeded : List[str]
        لیست متون خوانده شده از فایل
    """
    
    print("🔍 تحلیل متون خوانده شده از فایل")
    print("=" * 60)
    
    # 1. بررسی و تمیز کردن داده‌ها
    print("🧹 در حال تمیز کردن داده‌ها...")
    
    # حذف خطوط خالی و خطوط فقط حاوی تگ
    cleaned_texts = []
    for text in file_readeded:
        stripped = text.strip()
        if stripped and stripped not in ['\ufeff', '', '\n']:
            # حذف BOM اگر وجود دارد
            if stripped.startswith('\ufeff'):
                stripped = stripped[1:]
            cleaned_texts.append(stripped)
    
    print(f"📊 قبل از تمیز کردن: {len(file_readeded)} خط")
    print(f"📊 بعد از تمیز کردن: {len(cleaned_texts)} خط")
    
    # 2. ترکیب خطوط مرتبط
    print("🔗 در حال ترکیب خطوط مرتبط...")
    
    combined_texts = []
    current_text = ""
    
    for line in cleaned_texts:
        # اگر خط فقط حاوی تگ است، متن جاری را ذخیره کن و جدید شروع کن
        if re.match(r'^\{.*\}$', line.strip()):
            if current_text:
                combined_texts.append(current_text.strip())
                current_text = ""
            combined_texts.append(line)
        else:
            # اضافه کردن خط به متن جاری
            current_text += " " + line
    
    # آخرین متن را اضافه کن
    if current_text:
        combined_texts.append(current_text.strip())
    
    print(f"📊 بعد از ترکیب: {len(combined_texts)} متن")
    
    # 3. ایجاد شناسه‌ها
    text_ids = [f"text_{i}" for i in range(len(combined_texts))]
    
    # 4. ایجاد آنالایزر
    analyzer = ArabicTextAnalyzer(
        texts=combined_texts,
        text_ids=text_ids,
        preprocess=True,
        use_bert=True,
        use_tfidf=True,
        max_texts=min(50, len(combined_texts))
    )
    
    # 5. تحلیل کامل
    results = analyzer.analyze_similarities()
    
    # 6. نمایش نتایج
    print("\n📋 نتایج تحلیل:")
    print("-" * 40)
    
    # نمایش اولین ۵ مقایسه
    if 'comparisons' in results and results['comparisons']:
        print("\n🎯 ۵ مقایسه اول:")
        for i, comp in enumerate(results['comparisons'][:5]):
            print(f"\n{i+1}. {comp['text1_id']} ↔ {comp['text2_id']}:")
            print(f"   Levenshtein: {comp['levenshtein_ratio']:.4f}")
            if comp.get('bert_similarity') is not None:
                print(f"   BERT: {comp['bert_similarity']:.4f}")
            if comp.get('tfidf_similarity') is not None:
                print(f"   TF-IDF: {comp['tfidf_similarity']:.4f}")
    
    # نمایش آمار
    print("\n📊 آمار کلی:")
    for method in ['levenshtein', 'bert', 'tfidf']:
        if method in results:
            stats = results[method]
            print(f"\n{method.upper()}:")
            print(f"  میانگین: {stats['mean']:.4f}")
            print(f"  حداقل: {stats['min']:.4f}")
            print(f"  حداکثر: {stats['max']:.4f}")
    
    return {
        'analyzer': analyzer,
        'results': results,
        'combined_texts': combined_texts
    }


# # اجرای تحلیل روی داده‌های شما
# if __name__ == "__main__":

#     # داده‌های نمونه شما
#     file_readeded = [
#         '\ufeff{*~{* 1 *}~*}\n',
#         '{*~{* 2 *}~*}\n',
#         '{*~{* 3 *}~*}\n',
#         '{h1~{h1الإهداءh1}~h1}\n',
#         'يا سيد الشهداء وشهيد الاباء ، اقدم لأعتابك هذه المجموعة الفواحة رمزاً للوفاء ، فما أتمتع به من عزة وكرامة وعافية وسلامة كانت من اشعاعاتك التي تضفيها علي بجاهك العظيم عند الله تعالى. \n',
#         'فتقبل يا سيدي بعض ما يجب من ولدك. \n',
#         'المؤلف  \n',
#         '{*~{* 4 *}~*}\n',
#         '{h1~{h1تصدير : h1}~h1}\n',
#         '-بقلم العلامة :\n',
#         'الشيخ محمد جواد مغنية \n',
#         '{p~{sبسمِ اللهِ الرَحمنِ الرّحيمِs}~p}\n',
#         '{h3~{h3وبعد : h3}~h3}\n'
#     ]
    
#     try:
#         analysis = analyze_your_texts(file_readeded)
#         print("\n✅ تحلیل با موفقیت انجام شد!")
        
#     except Exception as e:
#         print(f"\n❌ خطا در تحلیل: {e}")
#         import traceback
#         traceback.print_exc()


# 1. بارگذاری داده‌ها
file_readeded = [...]  # لیست متون شما

# 2. تحلیل داده‌ها
analysis = analyze_your_texts(file_readeded)

# 3. دسترسی به نتایج
analyzer = analysis['analyzer']
results = analysis['results']
combined_texts = analysis['combined_texts']

# 4. بررسی ماتریس‌ها
lev_matrix = analyzer.levenshtein_matrix
bert_matrix = analyzer.bert_matrix
tfidf_matrix = analyzer.tfidf_matrix

# 5. بررسی نتایج مقایسه
for (id1, id2), result in list(analyzer.similarity_results.items())[:10]:
    print(f"{id1} ↔ {id2}:")
    print(f"  Levenshtein: {result.levenshtein_ratio:.4f}")
    if result.bert_similarity:
        print(f"  BERT: {result.bert_similarity:.4f}")