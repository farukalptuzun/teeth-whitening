python whiten_teeth.py --input in.jpg --output out.jpg --strength 0.9 --mask-grow 5 --save-mask 

# 🦷 Diş Beyazlatma Script'i (MediaPipe FaceMesh Tabanlı)

Bu script, verilen bir yüz fotoğrafındaki **ağız bölgesini tespit eder**, dişleri maskeleyip **LAB renk uzayında beyazlatma işlemi** uygular.  
**MediaPipe FaceMesh** ile diş bölgesi tespit edilir, dudak ve diş eti tonları hariç tutulur.

---

## 📦 Gereksinimler

Python 3.8+ ile çalışır.  
Aşağıdaki kütüphaneleri yükleyin:

```bash
pip install opencv-python mediapipe numpy

## 📂 Dosya Açıklamaları
whiten_teeth.py → Ana script (diş beyazlatma işlemi)
in.jpg → İşlem yapılacak fotoğraf
out.jpg → İşlem sonucu kaydedilecek fotoğraf

## ⚙️ Kullanım
*1️⃣ Temel Kullanım
python whiten_teeth.py --input in.jpg --output out.jpg

*2️⃣ Beyazlatma Gücünü Ayarlamak
*--strength parametresi 0.0 ile 1.5 arasında olabilir. Varsayılan: 0.6
python whiten_teeth.py --input input.jpg --output output.jpg --strength 1.1

*3️⃣ Maske Ayarları
--mask-grow
Maskeyi piksel cinsinden genişletir.
Daha yüksek değer = daha fazla diş/gum kapsar.
Varsayılan: 3
Önerilen: 4–6 (maske kısa kalıyorsa)

--gum-a-cut
LAB renk uzayında a* kanal eşiği.
Düşük değer = diş eti de dahil daha geniş alan.
Varsayılan: 155
Önerilen: 145–150 (daha geniş kapsama için)

python whiten_teeth.py --input in.jpg --output out.jpg --strength 1.1 --mask-grow 5 --gum-a-cut 150

*4️⃣ Maskeyi Kaydetmek
--save-mask parametresi ile diş maskesi .png olarak kaydedilir
python whiten_teeth.py --input in.jpg --output out.jpg --save-mask

*Notlar
Fotoğrafın net ve iyi aydınlatılmış olması maskenin doğruluğunu artırır.
Eğer maske dişlerin tamamını kapsamıyorsa:
--mask-grow değerini artır
--gum-a-cut değerini 145–150 aralığına düşür
Aşırı beyaz görünüyorsa --strength değerini düşür

Kullanılacak asıl kod python whiten_teeth.py --input in.jpg --output out.jpg --strength 0.9 --mask-grow 5 --save-mask  bunun üsütnden gerekli düzenlemeler yapılarak 

| Girdi (Input) | Maske (Mask) | Çıktı (Output) |
|---------------|--------------|----------------|
| ![input](examples/in.jpg) | ![mask](examples/out_mask.png) | ![output](examples/out.jpg) |

| Girdi (Input) | Maske (Mask) | Çıktı (Output) |
|---------------|--------------|----------------|
| ![input](examples/in1.jpg) | ![mask](examples/out_mask1.png) | ![output](examples/out1.jpg) |


MIT License

Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

