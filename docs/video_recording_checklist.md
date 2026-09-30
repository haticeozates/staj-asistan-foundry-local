# StajAsistan 2.0 video çekim checklist'i

Bu video için hedef: jürinin 6-8 dakika içinde şunu görmesi: proje çalışıyor,
kanıta bağlı cevap veriyor, özel veriyi koruyor, otomatik mesaj göndermiyor.

## Proje bitti mi?

Evet, geliştirme tarafı bitti:

- Yerel RAG, maskeleme, chunking, hybrid retrieval, citation ve refusal hazır.
- Gelen mesaj triyajı ve üç karar hazır: taslak hazır, insan onayı şart, cevaplama.
- Telegram poller + SQLite onay kuyruğu + onayla gönder yolu kodlandı ve test edildi.
- Branch GitHub'a pushlandı, PR açık: https://github.com/haticeozates/staj-asistan-foundry-local/pull/1
- Test kanıtı: 419 passed, 18 skipped.

Bitmeyen şey canlı operasyon kurulumu:

- BotFather, token, chat ID, gerçek Telegram grubu ve canlı uçtan uca deneme yapılmadı.
- Bu videoda canlı bot göstermek zorunda değilsin. Bunu opsiyonel pilot olarak anlat.

## Videoda ne çekilecek?

### 1. GitHub kanıtı

Ekranda PR'ı aç:

https://github.com/haticeozates/staj-asistan-foundry-local/pull/1

Söyle:

> Kod GitHub'da. Bu branch Telegram operasyon pilotunu ekliyor. Ham WhatsApp
> exportları, özel indeks, .env, token ve gerçek chat ID repoda yok.

Gösterilecek yerler:

- PR başlığı
- Branch adı: `feature/telegram-operations-pilot`
- Changed files veya README/docs kısmı

### 2. Streamlit güvenli demo

Terminal:

```bash
cd /Users/haticeozates/Projects/staj-asistan-foundry-local
source .venv/bin/activate
streamlit run app.py
```

Uygulamada:

1. `Temizle`
2. `Örnek veri`
3. Sidebar'da örnek veri yüklendiğini göster

Söyle:

> Demo örnek veriyle çekiliyor. Gerçek WhatsApp arşivi ve özel indeks yalnız bu
> makinede, git dışında duruyor.

### 3. Grounded answer

Soru:

```text
Final tesliminde ne gerekiyor?
```

Göster:

- Kısa Türkçe cevap
- Citation / kaynak paneli
- Modun teslim kontrol listesine yönlenmesi

Söyle:

> Burada önemli olan cevap metni değil; cevap alttaki kaynaklara bağlı. Kaynak
> yoksa cevap üretmiyor.

### 4. Liste düzeltme örneği

Soru:

```text
Listede projem yanlış görünüyor, Foundry Local olarak güncellenmesini istiyorum. Bu mesaj yeterli mi?
```

Göster:

- Düzeltme isteği olarak sınıflanması
- Eksik bilgi varsa söylemesi
- `auto_apply: false` veya insan onayı mantığı

Söyle:

> Liste veya sertifika gibi kişisel etkisi olan konularda sistem taslak üretse
> bile insan onayı istiyor. Kendisi liste değiştirmiyor.

### 5. Gelen mesaj simülasyonu

Görünüm değiştir:

```text
Gelen Mesaj Simülasyonu
```

Üç örnek göster:

- GitHub repo hazır ama video çekmedim...
- Listede projem yanlış...
- İstanbul'da hava nasıl?

Söyle:

> Gelen mesajda kullanıcı mod seçmiyor. Sistem intent'i buluyor, modu seçiyor ve
> karar veriyor: taslak, insan onayı veya cevaplama.

### 6. Refusal kanıtı

Soru:

```text
İstanbul'da hava nasıl?
```

Göster:

- Kaynak yok
- Citation yok
- Taslak yok

Söyle:

> Bu benim en önemli güvenlik kanıtım. Sistem bilmediği konuda kibarca uydurmuyor;
> retrieval eşiği geçilmediği için model çağrısı yapılmıyor.

### 7. Test kanıtı

Terminalde daha önceki sonucu göster veya yeniden çalıştır:

```bash
.venv/bin/pytest -o addopts= -q --tb=line
```

Beklenen:

```text
419 passed, 18 skipped
```

Söyle:

> Bu sadece UI demosu değil; parser, privacy, retrieval, triage, Telegram kuyruğu
> ve onay akışı offline testlerle korunuyor.

## Kapanış cümlesi

> StajAsistan 2.0 bir bulut chatbot değil. Kişisel verili staj arşivinde,
> kanıta bağlı yerel RAG çalıştırıyor; gelen mesajı sınıflandırıyor, taslak
> üretiyor, riskli konularda insan onayı istiyor ve kanıt yoksa susuyor.

## Videoda gösterme

- Gerçek WhatsApp export dosyaları
- `data/private/config.json`
- `data/index/private/chunks.json`
- `.env`
- Telegram bot token
- Gerçek chat ID
- Gerçek onay kuyruğu ekranı
- Eğitmen adı veya ham mesaj içeren terminal çıktısı

## GitHub'a video ekleme

Video dosyasını repo içine koyacaksan güvenli ad:

```text
delivery/staj-asistan-demo-video.mp4
```

Önce dosya boyutunu kontrol et:

```bash
ls -lh delivery/staj-asistan-demo-video.mp4
```

Çok büyükse GitHub'a doğrudan koymak yerine Releases veya Drive linki kullan.
