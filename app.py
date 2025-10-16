import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Atur tata letak halaman
st.set_page_config(layout="wide")

# Mengatur kebijakan presisi campuran
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Fungsi untuk memuat model dengan penanganan kesalahan
@st.cache_resource
def load_gan_models():
    try:
        # Tambahkan compile=False untuk menghindari peringatan saat memuat
        generator = load_model('generator_final.h5', compile=False)
        discriminator = load_model('discriminator_final.h5', compile=False)
        return generator, discriminator
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    # Pastikan gambar dalam mode RGB
    img = image.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = (img_array.astype(np.float32) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Memuat model
generator, discriminator = load_gan_models()

# Judul dan deskripsi aplikasi
st.title("Generator Batik Kawung dengan DCGAN")
st.write(
    "Aplikasi ini menggunakan model DCGAN yang dilatih pada dataset Batik Kawung. "
    "Anda dapat menghasilkan gambar batik baru secara acak atau mengunggah gambar Anda sendiri "
    "untuk diperiksa oleh diskriminator."
)

# Tampilkan UI jika model berhasil dimuat
if generator and discriminator:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Generator Batik")
        if st.button("Buat Gambar Batik Baru"):
            # Hasilkan gambar dari noise acak
            latent_dim = 128 # Pastikan dimensi ini sesuai dengan model Anda
            random_latent_vector = tf.random.normal(shape=(1, latent_dim))
            generated_image = generator(random_latent_vector, training=False)
            generated_image = (generated_image + 1) / 2.0  # Ubah skala gambar ke [0, 1]

            st.image(generated_image.numpy(), caption="Batik yang Dihasilkan", use_column_width=True)

    with col2:
        st.header("Diskriminator Batik")
        uploaded_file = st.file_uploader("Unggah gambar batik...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

            # Pra-pemrosesan gambar dan buat prediksi
            processed_image = preprocess_image(image)
            prediction = discriminator(processed_image, training=False)

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi Diskriminator:")
            if prediction[0][0] > 0.5:
                st.success(f"Gambar ini kemungkinan **Asli** (skor: {prediction[0][0]:.2f})")
            else:
                st.error(f"Gambar ini kemungkinan **Palsu** (skor: {prediction[0][0]:.2f})")