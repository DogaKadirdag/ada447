

# !pip install eco2ai
# !git clone https://github.com/mlco2/impact.git

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from eco2ai import Tracker
import time

# =============================================================================
# GLOBAL SABİTLER — tüm parametreler tek yerden yönetilir
# =============================================================================
LOOK_BACK          = 5       # LSTM pencere boyutu
VERI_SINIRI        = 2000    # Kullanılacak satır sayısı (hız için)
EPOCH_SAYISI       = 100
OGRENME_HIZI       = 0.01
ORTALAMA_KARBON_YG = 450.0   # Türkiye ortalama karbon yoğunluğu (gCO2/kWh)
CBAM_VERGI_TON_EURO = 85.0   # AB ETS güncel karbon ton fiyatı (Euro)
KARBON_ESIGI       = 500.0   # Kapalı döngü eşiği (gCO2/kWh)

# =============================================================================
# AKILLI DONANIM SEÇİMİ — tek noktadan, tüm kodda tutarlı
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan donanım: {device}")

# =============================================================================
# 1. TÜRKİYE EPİAŞ TÜKETİM VERİSİNİN YÜKLENMESİ
# =============================================================================
print("\n--- 1. Gerçek Türkiye EPİAŞ Saatlik Tüketim Verisi Yükleniyor ---")
df_gercek = pd.read_csv("RealTimeConsumption-01012016-04082020.csv")
df_gercek['Consumption (MWh)'] = (
    df_gercek['Consumption (MWh)']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)
turkiye_verisi = df_gercek['Consumption (MWh)'].dropna().values


# =============================================================================
# DİNAMİK ÖLÇEKLENDİRME
# =============================================================================
def create_dynamic_dataset(dataset, look_back=LOOK_BACK):
    """
    Pencere bazlı min-max normalizasyon uygular.
    Sabit global ölçek yerine her pencereye özel ölçek kullanır;
    bu sayede tüketim yüküne duyarlı dinamik katsayı hesabına zemin hazırlar.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        pencere = dataset[i:(i + look_back)]
        hedef   = dataset[i + look_back]
        p_min, p_max = np.min(pencere), np.max(pencere)
        if p_max == p_min:
            X.append(np.zeros(look_back))
            Y.append(0)
        else:
            X.append((pencere - p_min) / (p_max - p_min))
            Y.append((hedef   - p_min) / (p_max - p_min))
    return np.array(X), np.array(Y)


X, y = create_dynamic_dataset(turkiye_verisi[:VERI_SINIRI], look_back=LOOK_BACK)

# .to(device) — .cuda() yerine; CPU ortamında da sorunsuz çalışır
X_train = torch.FloatTensor(X).unsqueeze(-1).to(device)
y_train = torch.FloatTensor(y).unsqueeze(-1).to(device)


# =============================================================================
# 2. LSTM SİNİR AĞI MİMARİSİ
# =============================================================================
class RealWorldLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(RealWorldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# Model da aynı device'a gönderiliyor — X_train ile tutarlı
model_lstm      = RealWorldLSTM().to(device)
criterion_lstm  = nn.MSELoss()
optimizer_lstm  = torch.optim.Adam(model_lstm.parameters(), lr=OGRENME_HIZI)


# =============================================================================
# 3. eco2AI SAYACININ BAŞLATILMASI (TÜRKİYE)
# =============================================================================
tracker = Tracker(
    project_name="Yapay Zeka CBAM Maliyet Optimizasyonu",
    experiment_description="Turkiye verisi uzerinde LSTM egitimi",
    file_name="anlik_tuketim.csv",
    alpha_2_code="TR",
    region="Istanbul"
)
tracker.start_training()


# =============================================================================
# 4. EĞİTİM DÖNGÜSÜ
# =============================================================================
print("\n--- 2. LSTM Modeli Türkiye Şebekesini Öğreniyor ve Enerji Takibi Başladı ---")
for epoch in range(EPOCH_SAYISI):
    model_lstm.train()
    optimizer_lstm.zero_grad()
    predictions = model_lstm(X_train)
    loss        = criterion_lstm(predictions, y_train)
    loss.backward()
    optimizer_lstm.step()
    time.sleep(0.1)

    if (epoch + 1) % 20 == 0:
        tracker.new_epoch({"loss": loss.item()})
        print(f"Adım {epoch+1}/{EPOCH_SAYISI} - Hata Payı: {loss.item():.6f}")

tracker.stop_training()

# Dosyanın diske yazılması için kısa bekleme (eco2AI yazma gecikmesi)
time.sleep(1)

print("\n--- TEBRİKLER: Eğitim Tamamlandı, Türkiye Enerji Tüketimi Kaydedildi! ---")


# =============================================================================
# 5. GELECEK SAATİN TAHMİN EDİLMESİ (LSTM İLE)
# =============================================================================
model_lstm.eval()
with torch.no_grad():
    son_5_saat = turkiye_verisi[-LOOK_BACK:]          # sabit → LOOK_BACK
    p_min, p_max = np.min(son_5_saat), np.max(son_5_saat)

    if p_max == p_min:
        son_5_saat_scaled = np.zeros(LOOK_BACK)
    else:
        son_5_saat_scaled = (son_5_saat - p_min) / (p_max - p_min)

    last_window_tensor = (
        torch.FloatTensor(son_5_saat_scaled)
        .reshape(1, LOOK_BACK, 1)
        .to(device)                                   # model ile aynı device
    )
    lstm_pred_scaled      = model_lstm(last_window_tensor).item()
    tahmini_tuketim_mwh   = (lstm_pred_scaled * (p_max - p_min)) + p_min


# =============================================================================
# 6. EKONOMİK ÇEVİRİ: TÜKETİMİN KARBONA DÖNÜŞMESİ
#    NOT: Bu formül model tabanlı dinamik bir tahmindir;
#    gerçek anlık şebeke emisyon verisi değildir. Tezde bu şekilde belirtiniz.
# =============================================================================
ortalama_tuketim        = np.mean(turkiye_verisi)
dinamik_karbon_katsayisi = (tahmini_tuketim_mwh / ortalama_tuketim) * ORTALAMA_KARBON_YG


# =============================================================================
# 7. KAPALI DÖNGÜ (CLOSED-LOOP) KARAR MEKANİZMASI
#    LSTM çıktısı → karbon yoğunluğu → eğitim zamanlaması kararı
#    Bu blok sistemi gerçek bir karar destek sistemi haline getirir.
# =============================================================================
print("\n--- 3. Kapalı Döngü Karar Mekanizması ---")
print(f"Tahmin edilen şebeke karbon yoğunluğu: {dinamik_karbon_katsayisi:.2f} gCO2/kWh")
print(f"Belirlenen eşik değer               : {KARBON_ESIGI:.2f} gCO2/kWh")

if dinamik_karbon_katsayisi > KARBON_ESIGI:
    karar = "ERTELENDİ"
    print(f"\n⚠️  Şebeke yükü yüksek → Karbon yoğunluğu eşiği aşıldı!")
    print("🔴 SİSTEM KARARI: Bir sonraki model eğitimi ERTELENMELI.")
    print("   → Önerilen eylem: Eğitimi gece düşük yük saatine (örn. 02:00-05:00) kaydır.")
else:
    karar = "ONAYLANDI"
    print(f"\n✅ Şebeke yükü düşük → Karbon yoğunluğu kabul edilebilir seviyede.")
    print("🟢 SİSTEM KARARI: Bir sonraki model eğitimi ŞİMDİ başlatılabilir.")


# =============================================================================
# 8. FİNANSAL MALİYET VE CBAM VERGİSİ HESABI
# =============================================================================
df_tuketim        = pd.read_csv("anlik_tuketim.csv")
toplam_enerji_kwh = df_tuketim['power_consumption(kWh)'].sum()

tahmini_karbon_gram  = toplam_enerji_kwh * dinamik_karbon_katsayisi
tahmini_karbon_ton   = tahmini_karbon_gram / 1_000_000
toplam_cbam_vergisi  = tahmini_karbon_ton * CBAM_VERGI_TON_EURO


# =============================================================================
# 9. YÖNETİCİ RAPORU
# =============================================================================
print("\n")
print("=" * 70)
print("   TÜRKİYE - AVRUPA BİRLİĞİ (CBAM) YAPAY ZEKA MALİYET RAPORU")
print("=" * 70)
print(f"→ Model Eğitiminin Donanım Enerji İhtiyacı : {toplam_enerji_kwh:.6f} kWh")
print(f"→ LSTM Tahmini Şebeke Yükü (Gelecek Saat)  : {tahmini_tuketim_mwh:.2f} MWh")
print(f"→ Dinamik Şebeke Karbon Yoğunluğu (model)  : {dinamik_karbon_katsayisi:.2f} gCO2/kWh")
print(f"→ Kapalı Döngü Sistem Kararı                : {karar}")
print()
print("--- AVRUPA BİRLİĞİ İHRACAT (CBAM) FİNANSAL MALİYETİ ---")
print(f"   Eğitimden Doğan Toplam Karbon   : {tahmini_karbon_gram:.4f} gram"
      f" ({tahmini_karbon_ton:.8f} Ton)")
print(f"   Güncel AB Karbon Vergisi (Ton)  : {CBAM_VERGI_TON_EURO} Euro")
print(f"   ŞİRKETİN ÖDEYECEĞİ CBAM MALİYETİ: {toplam_cbam_vergisi:.8f} Euro")
print()
print(" SİSTEM ÖNERİSİ: Model eğitimi, şebeke yükünün (ve kirliliğin) tavan")
print(" yaptığı bu saat yerine, LSTM'in önereceği gece saatlerinde (düşük yük)")
print(" yapılırsa AB'ye ödenecek CBAM vergisi minimize edilecektir.")
print("=" * 70)
