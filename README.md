#  Akciğer Kanseri BT Görüntüleri Üzerinden Derin Öğrenme Tabanlı Teşhis Sistemi

Bu proje, Chest CT-Scan görüntülerini analiz ederek 3 farklı akciğer kanseri türünü ve sağlıklı dokuları sınıflandırmak amacıyla geliştirilmiş uçtan uca bir yapay zeka çözümüdür.

---

##  1. Proje Konusu ve Önemi

### Seçilme Gerekçesi
Akciğer kanseri, dünya genelinde en yüksek mortalite (ölüm) oranına sahip kanser türüdür. Erken evrede teşhis, hayatta kalma şansını %50'den fazla artırırken, radyolojik görüntülerin manuel incelenmesi zaman alıcı ve hata payına açık olabilmektedir. Bu proje, tıbbi teşhis süreçlerini hızlandırmak ve radyologlara dijital bir karar destek mekanizması sunmak amacıyla seçilmiştir.

### Literatür ve Alanın Önemi
Geleneksel bilgisayarlı görü yöntemlerinin aksine, derin öğrenme (Deep Learning) modelleri dokulardaki mikroskobik paternleri yakalayabilmektedir. Bu alan, "Sağlıkta Yapay Zeka" (AI in Healthcare) vizyonunun en kritik uygulama noktalarından biridir.

---

##  2. Veri Setinin Belirlenmesi 

Projede Kaggle üzerinden sağlanan **"Chest CT-Scan Images"** veri seti kullanılmıştır.
- **Sınıflar:** Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma ve Normal.
- **Veri Dağılımı:** Veri seti dengeli bir dağılıma sahiptir ve modelin her bir sınıfı yeterli düzeyde öğrenmesine olanak tanımaktadır.
- **İşleme:** Görüntüler 224x224 boyutuna standardize edilmiş ve ImageNet normlarına göre normalize edilmiştir.

---

##  3. Uygulanan Yöntem ve Algoritma Seçimi 

### Seçilen Mimari: ResNet18 (Transfer Learning)
Projede sıfırdan bir model eğitmek yerine, önceden milyonlarca görselle eğitilmiş **ResNet18** mimarisi kullanılarak **Transfer Learning** yaklaşımı uygulanmıştır.

### Karşılaştırmalı Analiz ve Seçim Gerekçesi
- **Neden ResNet?** Klasik CNN modellerinde (VGG16 vb.) ağ derinleştikçe gradyan kaybolması (vanishing gradient) yaşanırken, ResNet "Residual Connections" (Atlamalı Bağlantılar) sayesinde çok daha derin özellikleri başarıyla öğrenebilmektedir.
- **Neden Transfer Learning?** Medikal veri setleri genellikle kısıtlıdır. Transfer learning sayesinde modelin genel görsel tanıma yetenekleri, akciğer dokusu tanıma özeline hızlıca aktarılmıştır.

---

##  4. Model Eğitimi ve Değerlendirme 

### Eğitim Detayları
- **Optimizer:** Adam (Dinamik öğrenme hızı ayarı için).
- **Loss:** Cross-Entropy (Çok sınıflı hata hesaplama için).
- **Epoch:** 10 (Kararlı öğrenme eğrisi için).

### Değerlendirme Metrikleri
Model başarısı sadece Accuracy (Doğruluk) ile değil, medikal projelerde kritik olan şu metriklerle ölçülmüştür:
- **Sensitivity (Recall):** Kanserli vakaları kaçırmama oranı (Medikal açıdan en kritik değer).
- **Specificity:** Sağlıklı kişiye yanlış tanı koymama oranı.
- **F1-Score:** Sınıflar arası dengeli performans ölçümü.
- **Confusion Matrix:** Hataların dağılım analizi.

Not: Modelin başarısı sadece genel doğruluk (Accuracy) ile değil, tıbbi teşhislerin doğası gereği kritik olan Recall ve Specificity metriklerine bakılarak yorumlanmalıdır.


---

##  5. Proje Dokümantasyonu ve Dosya Yapısı 

Proje, GitHub üzerinde modüler bir yapıda düzenlenmiştir:
- `analiz_ve_egitim.ipynb`: Veri keşfi, grafikler, eğitim raporu ve tüm analiz metriklerini içerir.
- `model.py`: ResNet18 tabanlı model sınıfı tanımı.
- `serve.py`: Kullanıcı dostu **Gradio** web arayüzü başlatıcı.
- `chest_model_final.pth`: Eğitilmiş model ağırlıkları.
- `requirements.txt`: Projenin çalışması için gerekli kütüphaneler.

---

##  6. Proje Sunumu 

Sunum sırasında modelin öğrenme eğrileri, hata matrisi analizi ve canlı **Gradio** demosu sunulacaktır. Sistem, bir CT görselini saniyeler içinde analiz edip olasılık yüzdeleriyle birlikte teşhis koyma yeteneğine sahiptir.

---

##  Kullanım
1. Kütüphaneleri yükleyin: `pip install -r requirements.txt`
2. Analizi inceleyin: `analiz_ve_egitim.ipynb`
3. Arayüzü çalıştırın: `python serve.py`


