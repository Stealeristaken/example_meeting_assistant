# Akıllı Toplantı Asistanı

Bu proje, Azure OpenAI ve LangChain kullanarak geliştirilmiş akıllı bir toplantı asistanıdır. Kullanıcıların doğal dil toplantı taleplerini işleyerek yapılandırılmış JSON çıktısı oluşturur.

## 🚀 Özellikler

- **Doğal Dil İşleme**: Türkçe toplantı taleplerini anlama
- **Vektör Veritabanı**: ChromaDB ile isim çözümleme
- **Azure OpenAI Entegrasyonu**: GPT-4o ile gelişmiş AI
- **Bellek Yönetimi**: ConversationBufferMemory ile konuşma geçmişi
- **Belirsizlik Çözümleme**: Çoklu isim eşleşmelerinde kullanıcıdan açıklama
- **ISO 8601 Zaman Formatı**: Standart zaman formatında çıktı
- **Email Gövdesi Oluşturma**: Otomatik profesyonel email içeriği

## 📋 Gereksinimler

- Python 3.8+
- Azure OpenAI hesabı
- Gerekli Python paketleri (requirements.txt)

## 🛠️ Kurulum

1. **Bağımlılıkları yükleyin:**

```bash
pip install -r requirements.txt
```

2. **Environment değişkenlerini ayarlayın:**

```bash
# env.example dosyasını .env olarak kopyalayın
cp env.example .env

# .env dosyasını düzenleyerek Azure OpenAI bilgilerinizi girin
# AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-api-key-here
```

3. **Uygulamayı çalıştırın:**

```bash
python meeting_assistant_enhanced.py
```

## 🔒 Güvenlik

⚠️ **ÖNEMLİ**: Bu proje artık hardcoded API anahtarları içermiyor. Tüm hassas bilgiler environment değişkenleri ile yönetiliyor.

- API anahtarları ve endpoint'ler `.env` dosyasında saklanır
- `.env` dosyası `.gitignore` ile Git'e dahil edilmez
- Test dosyaları da environment değişkenlerini kullanır

## 🎯 Kullanım

### Örnek Toplantı Talepleri

```
👤 SİZ: Arda Orçun ve Şahin ile yarın 90 dakikalık proje toplantısı organize et
👤 SİZ: Ali ve Ahmet ile pazartesi sabah 10:00'da toplantı planla
👤 SİZ: Ozden ve Emre ile Q3 bütçe görüşmesi yap
👤 SİZ: Yarın saat 14:00'de Arda ile hızlı sync
```

### Çıktı Formatı

```json
{
  "body": "Merhaba,\n\nQ3 bütçe görüşmesi için toplantı planladık.\n\nKatılımınızı bekleriz.\n\nSaygılarımızla,",
  "endTime": "2025-07-15T10:30:00+03:00",
  "meeting_duration": 30,
  "startTime": "2025-07-15T10:00:00+03:00",
  "subject": "Q3 Bütçe Görüşmesi",
  "user_details": [
    {
      "email_address": "ozden.orkon@company.com.tr",
      "full_name": "Özden Gebizli Orkon",
      "id": 12
    },
    {
      "email_address": "emre.celik@company.com.tr",
      "full_name": "Emre Çelik",
      "id": 14
    }
  ]
}
```

## 🔧 Teknik Detaylar

### Mimari Bileşenler

1. **VectorDatabaseManager**: ChromaDB ile isim arama
2. **MeetingAssistantTools**: LangChain araçları
3. **MeetingAssistantAgent**: Ana agent sınıfı
4. **Pydantic Modeller**: Veri doğrulama

### İş Akışı

1. **Bilgi Çıkarma**: Kullanıcı talebinden toplantı bilgilerini çıkar
2. **İsim Arama**: Vektör veritabanında katılımcıları ara
3. **Belirsizlik Çözümleme**: Çoklu eşleşmelerde kullanıcıdan açıklama iste
4. **Zaman Çözümleme**: Tarih/saat bilgilerini ISO 8601 formatına çevir
5. **Email Oluşturma**: Profesyonel email gövdesi oluştur
6. **JSON Çıktısı**: Final toplantı JSON'ını üret

### Örnek Veri

Sistem 25 örnek kullanıcı ile gelir:

- Ahmet Yılmaz, Ahmet Kaya, Ahmet Özkan
- Ali Şahin, Ali Demir, Ali Can Yılmaz
- Şahin Koç, Şahin Nicat, Mehmet Şahin
- Arda Orçun, Ege Gülünay
- Ve daha fazlası...

## 🎨 Özellikler

### Akıllı İsim Çözümleme

- Vektör benzerliği ile isim arama
- Türkçe karakter desteği
- Yazım hatalarını tolere etme
- Çoklu eşleşmelerde açıklama isteme

### Zaman İşleme

- Doğal dil tarih/saat anlama
- İş saatleri kontrolü (09:00-17:00)
- Varsayılan 30 dakika süre
- UTC+3 zaman dilimi

### Email Oluşturma

- Profesyonel Türkçe email gövdesi
- Toplantı amacına uygun içerik
- Katılımcı bilgilerini dahil etme

## 🔍 Test Senaryoları

### Senaryo 1: Basit Toplantı

```
Kullanıcı: "Arda Orçun ile yarın toplantı planla"
Sonuç: Doğrudan JSON çıktısı
```

### Senaryo 2: Belirsiz İsim

```
Kullanıcı: "Ali ile toplantı yap"
Sistem: "Hangi Ali'yi kastettiniz?"
Kullanıcı: "Ali Şahin"
Sonuç: JSON çıktısı
```

### Senaryo 3: Eksik Bilgi

```
Kullanıcı: "Şahin ile pazartesi toplantı"
Sistem: "Toplantının konusu nedir?"
Kullanıcı: "Proje durumu"
Sonuç: JSON çıktısı
```

## 🚨 Hata Yönetimi

- **API Hataları**: Azure OpenAI bağlantı sorunları
- **Veri Doğrulama**: Pydantic ile format kontrolü
- **Belirsizlik**: Kullanıcıdan açıklama isteme
- **Eksik Bilgi**: Eksik alanlar için soru sorma

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📞 İletişim

Sorularınız için issue açabilirsiniz.
