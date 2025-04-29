

# 🧠 Commercial Courts Research Engine

A smart **PDF-based research assistant** built with **Streamlit**, designed to simplify querying and understanding the **Commercial Courts Act 2015**. The app uses **chunking, vector search, translation, and audio synthesis** to help you study legal documents in both **English** and **French**.

---

## 🚀 Features

- 🔍 **PDF Parsing & Chunking** — Splits legal PDFs into manageable text chunks.
- 📚 **Semantic Search** — Uses embeddings and Pinecone for relevant answer retrieval.
- 🎧 **Text-to-Speech** — Generates an audio response of the answer.
- 🌍 **Multilingual Support** — Translate responses into French using **M2M100** by Facebook AI.
- 📂 **Local PDF Preloading** — Auto-loads `commercialcourtsact2015.pdf` for instant querying.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Pinecone](https://www.pinecone.io/) — Vector database for semantic search.
- OpenCV & other NLP utilities for preprocessing and translation.

---

## 💻 How to Run

1️⃣ Clone the repository:
```bash
git clone https://github.com/manisudhan/chatbot1.git
cd chatbot1
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Make sure you place your **`commercialcourtsact2015.pdf`** file in the project directory.

4️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📝 Usage

- Upload or preload PDFs.
- Ask questions like:
  > *"What is the purpose of Commercial Courts under the Act?"*
- Choose **English** or **French** output.
- Listen to the generated audio or download the response.

---

## ⚠️ Notes

- **Secrets:** Make sure you don't push Hugging Face API tokens, Pinecone keys, or any sensitive information into your commits.
- The code is set to preload `commercialcourtsact2015.pdf` but you can change this path in `app.py` if you want to process another document.

---

## 📌 Screenshot

<img width="959" alt="image" src="https://github.com/user-attachments/assets/4b332a22-6ae6-4a83-bf17-097ac0987534" />


---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

