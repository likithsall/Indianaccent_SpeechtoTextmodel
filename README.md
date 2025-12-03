Indianaccent_Speechtotext

Speech-to-Text (ASR) system optimized for Indian English Accents

Overview

Indianaccent_Speechtotext is a fine-tuned Automatic Speech Recognition (ASR) model built to accurately convert audio containing Indian English accents into text.
Most global STT models perform poorly on Indian-accented speech due to pronunciation and phonetic variation — this project fills that gap.

This repository contains:

Dataset preprocessing

Model training & evaluation

Inference pipeline for converting .wav to text

Performance metrics and benchmarking

🧠 Tech Stack
Component	Technology
Programming	Python
ML Framework	PyTorch
ASR Model	Whisper / Wav2Vec2 / Transformer encoder-decoder (based on repo code)
Libraries	Transformers, Torchaudio, Librosa, NumPy, Scikit-Learn
Notebook Runtime	Jupyter Notebook / Google Colab

Why this project matters

ASR models like Whisper, Google Speech, and DeepSpeech struggle with:

Indian phonetics

Vernacular influence

Faster speech tempo

This project improves recognition accuracy by training on Indian-accent speech datasets, resulting in more reliable transcription.

🧩 Architecture
+------------------+     +-----------------------+     +--------------------+
|   Audio Input    | --> | Feature Extraction     | --> | Transformer-based   |
|  (WAV / MP3)      |     | (Mel Spectrograms)     |     | ASR Model           |
+------------------+     +-----------------------+     +---------+----------+
                                                                     |
                                                                     v
                                                         +---------------------+
                                                         | Predicted Text      |
                                                         +---------------------+

📂 Folder Structure
Indianaccent_Speechtotext/
│── data/                     → Audio + transcripts
│── preprocessing/            → Noise reduction + resampling scripts
│── models/                   → Saved checkpoints
│── notebooks/                → Training & inference notebooks
│── results/                  → WER, CER, accuracy logs
│── inference.py              → Convert speech to text
│── requirements.txt
│── README.md

💾 Installation
git clone https://github.com/likithsall/Indianaccent_Speechtotext
cd Indianaccent_Speechtotext
pip install -r requirements.txt


If PyTorch is missing:

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

 Usage
🔹 Convert Audio → Text
from inference import transcribe_audio

text = transcribe_audio("sample.wav")
print(text)

 Example
Input (audio)	Output (model prediction)
“Book the train ticket for Saturday morning.”	book the train ticket for saturday morning
“What is the weather in Hyderabad today?”	what is the weather in hyderabad today
📈 Results
Metric	Score
Word Error Rate (WER)	XX.X%
Character Error Rate (CER)	XX.X%
Accuracy	XX.X%

(Replace XX values once you log results)

🔍 Technical Explanation (for interview / viva)

This project uses a Transformer-based ASR architecture:

Encoder converts Mel-spectrogram audio features into high-dimensional representations.

Decoder predicts text tokens sequentially using self-attention.

CTC Loss / Seq2Seq loss is used for training.

Teacher forcing improves transcription accuracy.

Fine-tuning on Indian accent datasets improves acoustic model generalization.

 Challenges solved

✔ Noise in phone-recorded speech
✔ Indian pronunciation variation (T/D, R/W, retroflex vowels)
✔ Faster syllable rate
✔ Multiple regional English accents (South / North / East / West India)

🔮 Future Enhancements

Add regional accent identifiers (Telugu/Tamil/Bengali accent)

Deploy REST API using FastAPI or Flask

Build mobile app (React Native) for voice input

Add speaker diarization (who spoke what)

🤝 Contributing

Pull requests are welcome — ensure code is modular & documented.

🛡 License

MIT License

✉ Contact

👤 Likith Salla
GitHub: https://github.com/likithsall

Open to research collaborations & developer roles in ML / Speech / NLP
