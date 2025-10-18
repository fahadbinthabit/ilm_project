# 🧠 ILM Project — Insurance Language Model

A full pipeline for building a domain-specialized Insurance Language Model (ILM).  
It extracts PDFs, chunks text, generates Q&A pairs using local **Ollama**, and fine-tunes a local **Llama 3** model using **Unsloth**.

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/ilm_project.git
cd ilm_project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
