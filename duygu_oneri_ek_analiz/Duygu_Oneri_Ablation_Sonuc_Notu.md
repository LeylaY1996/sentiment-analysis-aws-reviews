# Duygu Analizi Entegrasyonunun Öneri Sistemine Etkisi — Ek Ablation Testi

## Amaç

Tezde (Bölüm 4.2.2 / 4.3.6) içerik tabanlı, işbirlikçi ve hibrit öneri sistemleri yalnızca
duygu-filtreli (Sentiment = 1) etkileşimler üzerinde değerlendirilmiş, ancak duygu bilgisi
*olmadan* aynı yöntemlerin ne sonuç vereceği ayrıca test edilmemişti. Bu not, o boşluğu
kapatmak için yapılan ek, kontrollü bir deneyin sonuçlarını özetler.

## Yöntem

Tezdeki (`23022026.ipynb`) içerik tabanlı / işbirlikçi / hibrit algoritmalar **birebir aynı
kodla** iki koşulda çalıştırıldı:

- **Koşul A (mevcut tez yaklaşımı):** Kullanıcı profili ve etkileşim matrisi yalnızca
  pozitif duygulu (Sentiment = 1) etkileşimlerden kuruldu.
- **Koşul B (temel/karşılaştırma):** Kullanıcı profili ve etkileşim matrisi kullanıcının
  **tüm** etkileşimlerinden (olumlu + olumsuz, duygu bilgisi kullanılmadan) kuruldu.

Her iki koşulda da başarı ölçütü **sabit** tutuldu: kullanıcının gerçekten sevdiği
(Sentiment = 1), elde tutulan (held-out) ürünleri modelin ne kadar iyi bulduğu
(Precision@5 / Recall@5). Adil bir kıyas için, değerlendirmede kullanılan kullanıcı
kümesi ve işbirlikçi filtreleme matrisine giren kullanıcı kümesi **iki koşulda da birebir
aynı** tutuldu — yalnızca bu kullanıcıların etkileşim verisine duygu filtresi uygulanıp
uygulanmadığı değişti.

**Veri:** Orijinal `Reviews.csv` (568.454 satır, kaynak: Kaggle `leylayilmaz/reviews-csv`)
dosyasından rastgele örneklenen 80.000 satırlık bir alt küme kullanıldı (74.031 satır,
Score=3 çıkarıldıktan sonra). 300 kullanıcı üzerinden değerlendirme yapıldı; bu
kullanıcılardan yalnızca 21 tanesinin (%7) en az bir olumsuz etkileşimi vardı — yani
iki koşul arasındaki fark pratikte bu 21 kullanıcının davranışından kaynaklanıyor.

## Sonuçlar

| Yöntem | Precision@5 (duygu VAR) | Precision@5 (duygu YOK) | Fark | Recall@5 (duygu VAR) | Recall@5 (duygu YOK) | Fark |
|---|---|---|---|---|---|---|
| İçerik Tabanlı | 0.0193 | 0.0180 | **+%7,4** | 0.0967 | 0.0900 | **+%7,4** |
| İşbirlikçi Filtreleme | 0.0127 | 0.0133 | −%5,0 | 0.0633 | 0.0667 | −%5,0 |
| Hibrit (α=0,9) | 0.0153 | 0.0140 | **+%9,5** | 0.0767 | 0.0700 | **+%9,5** |

## Yorum

İçerik tabanlı ve hibrit yöntemlerde duygu filtresi eklenmesi, öneri kalitesinde (Precision@5
ve Recall@5) tutarlı, küçük ama pozitif bir iyileşme (~%7-10 göreli artış) sağlamıştır. Bunun
teorik gerekçesi anlamlıdır: kullanıcı profili TF-IDF ile kurulurken yalnızca beğenilen
ürünlerin metinleri kullanıldığında, profil "gerçekten sevilen içerik" imzasını daha az
gürültüyle temsil etmektedir. İşbirlikçi filtrelemede ise fark yönü tersine dönmüş ve
etkisi ihmal edilebilir düzeyde kalmıştır — bu, işbirlikçi filtrelemenin zaten
"etkileşim var mı" sinyaline dayanmasından ve duygu bilgisinin bu yöntemde katkısının
sınırlı olmasından kaynaklanıyor olabilir.

**Önemli sınırlama:** Bu farkın istatistiksel gücü düşüktür — 300 değerlendirme
kullanıcısından yalnızca 21'i iki koşul arasında gerçekten farklı veriyle eğitildi. Bu,
tezin kendi Sınırlılıklar bölümünde (5.2) zaten belirtilen veri seyrekliği sorununun bir
uzantısıdır. Bulgu, "duygu entegrasyonu öneri kalitesini artırır" hipotezini **yönü
itibarıyla destekleyen, ancak kesin/istatistiksel olarak kanıtlamayan** bir ek kanıt
olarak sunulmalıdır.

## Savunmada Kullanılabilecek Cevap (özet)

> "Tez metninde bu karşılaştırma doğrudan raporlanmamıştı; savunma hazırlığı kapsamında
> aynı algoritma ve aynı değerlendirme metodolojisiyle kontrollü bir ek test yaptım.
> Sonuçlar, duygu filtresinin içerik tabanlı ve hibrit öneri sistemlerinde Precision@5 ve
> Recall@5'i yaklaşık %7-10 göreli olarak iyileştirdiğini gösteriyor; işbirlikçi
> filtrelemede ise etkisi ihmal edilebilir düzeyde kaldı. Bu sonuç, tezin sınırlılıklar
> bölümünde belirtilen veri seyrekliği nedeniyle istatistiksel olarak güçlü değil, ancak
> yönü itibarıyla literatürdeki beklentiyle (duygu bilgisinin öneri kalitesini artırması)
> tutarlı. Daha büyük ve dengeli bir veri setiyle bu etkinin daha net ortaya
> çıkacağını düşünüyorum."

## Notlar

- Kullanılan kod: `ablation_experiment.py` (repoya eklenmiş, `sentiment_filter=True/False`
  parametresiyle çalışıyor).
- Daha güçlü bir sonuç isteniyorsa, aynı script tam veri setinin (568.454 satır) tamamı
  veya daha büyük bir örneklemi üzerinde tekrar çalıştırılabilir.
