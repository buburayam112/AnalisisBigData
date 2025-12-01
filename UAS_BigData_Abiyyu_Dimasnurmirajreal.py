import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Wine Quality AI", page_icon="🍷", layout="centered")

# --- STYLE CSS ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .stSlider [data-baseweb="slider"] { padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- KONSTANTA & STATISTIK ---
STATS = {
    'red': {
        'alcohol': {'min': 8.4, 'max': 14.9},
        'volatile acidity': {'min': 0.12, 'max': 1.58},
        'density': {'min': 0.990, 'max': 1.004},
        'chlorides': {'min': 0.012, 'max': 0.611},
        'total sulfur dioxide': {'min': 6.0, 'max': 289.0}
    },
    'white': {
        'alcohol': {'min': 8.0, 'max': 14.2},
        'volatile acidity': {'min': 0.08, 'max': 1.1},
        'density': {'min': 0.987, 'max': 1.039},
        'chlorides': {'min': 0.009, 'max': 0.346},
        'total sulfur dioxide': {'min': 9.0, 'max': 440.0}
    }
}

# --- FUNGSI HELPER ---
def scale_to_actual(score, min_val, max_val):
    """Konversi skala slider 1-10 ke nilai kimia asli"""
    if score == 1: return min_val
    if score == 10: return max_val
    return min_val + (score - 1) * (max_val - min_val) / 9

@st.cache_resource
def load_data():
    try:
        data = joblib.load('wine_model_final.pkl')
        return data
    except FileNotFoundError:
        return None

# --- LOAD DATA & MODEL ---
raw_data = load_data()

if raw_data is None:
    st.error("⚠ File 'wine_model_final.pkl' tidak ditemukan.")
    st.stop()


if isinstance(raw_data, dict):
    model = raw_data['model']
    # Ambil data test jika ada untuk keperluan grafik
    X_test = raw_data.get('X_test')
    y_test = raw_data.get('y_test')
else:
    model = raw_data
    X_test, y_test = None, None

# ==========================================
# MAIN INTERFACE
# ==========================================]
st.title("🍷 Wine Quality Check")

# Membuat Tab Navigasi
tab1, tab2 = st.tabs(["🔮 Simulasi Prediksi", "📊 Evaluasi Model"])

# ==========================================
# TAB 1: SIMULASI (Dashboard User)
# ==========================================
with tab1:
    with st.container(border=True):
        # 1. Pilih Jenis Wine
        wine_type = st.radio("Jenis Wine", ["Red Wine", "White Wine"], horizontal=True, key="w_type")
        w_key = 'red' if wine_type == "Red Wine" else 'white'
        type_num = 0 if wine_type == "Red Wine" else 1

        st.divider()
        st.caption("Geser slider (1 = Sangat Rendah, 10 = Sangat Tinggi)")

        col1, col2 = st.columns(2)

with tab1:
    with st.container(border=True):
        # --- Tampilkan Feature Importance ---
        st.subheader("🔎 Feature Importance (Pengaruh Fitur Terhadap Prediksi)")
        feat_names = ['alcohol', 'volatile acidity', 'density', 'chlorides', 'total sulfur dioxide', 'type_numeric']
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Fitur': feat_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
        sns.barplot(x='Importance', y='Fitur', data=feat_imp_df, ax=ax_imp, palette='viridis')
        ax_imp.set_xlabel('Pengaruh (Semakin tinggi, semakin penting)')
        ax_imp.set_ylabel('Fitur')
        st.pyplot(fig_imp)
        st.caption("Fitur dengan nilai pengaruh tertinggi paling menentukan hasil prediksi kualitas wine.")
        
        


    

        # ...existing code...
        
        # INPUT SKALA 1-10
        with col1:
            s_alc = st.slider("1. Tingkat Alkohol", 1, 10, 5, help="Alkohol tinggi biasanya lebih baik.")
            s_vol = st.slider("2. Tingkat Asam Cuka", 1, 10, 3, help="Rasa asam tajam. Sebaiknya rendah.")
            s_sulf = st.slider("3. Tingkat Sulfur (Pengawet)", 1, 10, 3)

        with col2:
            s_den = st.slider("4. Kepadatan (Density)", 1, 10, 5, help="Semakin cair (rendah) semakin baik.")
            s_chl = st.slider("5. Tingkat Garam", 1, 10, 2, help="Rasa asin. Sebaiknya rendah.")

        # PROSES DATA INPUT
        actual_vals = {}
        feats = ['alcohol', 'volatile acidity', 'density', 'chlorides', 'total sulfur dioxide']
        inputs = [s_alc, s_vol, s_den, s_chl, s_sulf]
        
        for f, score in zip(feats, inputs):
            actual_vals[f] = scale_to_actual(score, STATS[w_key][f]['min'], STATS[w_key][f]['max'])

        # TOMBOL PREDIKSI
        if st.button("🔮 Cek Kualitas", type="primary", use_container_width=True):
            # Format input sesuai format training
            row = [actual_vals['alcohol'], actual_vals['volatile acidity'], actual_vals['density'], 
                   actual_vals['chlorides'], actual_vals['total sulfur dioxide'], type_num]
            
            df_input = pd.DataFrame([row], columns=feats + ['type_numeric'])
            
            # Prediksi
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            
            st.divider()
            if pred == 1:
                st.success(f"## ✅ HASIL: HIGH QUALITY")
                st.write(f"Keyakinan Model: **{proba[1]*100:.1f}%**")
            else:
                st.error(f"## ⚠ HASIL: LOW QUALITY")
                st.write(f"Keyakinan Model: **{proba[0]*100:.1f}%**")
                
                # Logika Saran Sederhana
                st.markdown("### 💡 Saran Perbaikan:")
                suggestions = []
                if s_alc < 7: suggestions.append("- Naikkan tingkat **Alkohol** (ke level 7-8).")
                if s_vol > 4: suggestions.append("- Kurangi **Asam Cuka** (volatile acidity).")
                if s_den > 5: suggestions.append("- Kurangi **Kepadatan** (density).")
                
                if suggestions:
                    for s in suggestions: st.write(s)
                else:
                    st.write("- Kombinasi fitur kimiawi saat ini cenderung menghasilkan wine standar.")

# ==========================================
# TAB 2: EVALUASI (Grafik Kinerja)
# ==========================================
with tab2:
    st.header("Metrik Evaluasi Model")
    
    # Cek apakah Data Test tersedia dalam file Pickle
    if X_test is not None and y_test is not None:
        
        # Hitung Prediksi pada Data Test
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        col_grafik1, col_grafik2 = st.columns(2)

        # --- 1. Confusion Matrix ---
        with col_grafik1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                        xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            plt.ylabel('Aktual (Kenyataan)')
            plt.xlabel('Prediksi Model')
            st.pyplot(fig_cm)
            st.info("Kotak biru gelap di diagonal utama menunjukkan jumlah tebakan yang benar.")

        # --- 2. ROC Curve ---
        with col_grafik2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(alpha=0.3)
            st.pyplot(fig_roc)
            st.info("AUC mendekati 1.00 artinya model sangat bagus membedakan High vs Low quality.")

    