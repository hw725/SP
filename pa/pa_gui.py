import tkinter as tk
from tkinter import filedialog, messagebox
import threading

def run_pa(input_file, output_file, embedder, threshold, max_length, splitter, device, openai_model, openai_api_key):
    try:
        from main import main
        import sys
        sys.argv = [
            "main.py",
            input_file,
            output_file,
            "--embedder", embedder,
            "--threshold", str(threshold),
            "--max-length", str(max_length),
            "--splitter", splitter,
            "--device", device,
            "--openai-model", openai_model,
        ]
        if openai_api_key:
            sys.argv += ["--openai-api-key", openai_api_key]
        main()
        messagebox.showinfo("완료", "PA 처리가 완료되었습니다.")
    except Exception as e:
        messagebox.showerror("오류", f"실행 중 오류 발생: {e}")

def start_pa():
    input_file = input_entry.get()
    output_file = output_entry.get()
    embedder = embedder_var.get()
    threshold = threshold_var.get()
    max_length = max_length_var.get()
    splitter = splitter_var.get()
    device = device_var.get()
    openai_model = openai_model_var.get()
    openai_api_key = openai_api_key_var.get()
    if not input_file or not output_file:
        messagebox.showerror("오류", "입력/출력 파일을 지정하세요.")
        return
    threading.Thread(target=run_pa, args=(input_file, output_file, embedder, threshold, max_length, splitter, device, openai_model, openai_api_key)).start()

def toggle_openai_options(*args):
    if embedder_var.get() == "openai":
        openai_model_label.grid(row=8, column=0)
        openai_model_menu.grid(row=8, column=1)
        openai_api_key_label.grid(row=9, column=0)
        openai_api_key_entry.grid(row=9, column=1)
    else:
        openai_model_label.grid_remove()
        openai_model_menu.grid_remove()
        openai_api_key_label.grid_remove()
        openai_api_key_entry.grid_remove()

root = tk.Tk()
root.title("PA Paragraph Aligner GUI")

tk.Label(root, text="입력 파일:").grid(row=0, column=0)
input_entry = tk.Entry(root, width=40)
input_entry.grid(row=0, column=1)
tk.Button(root, text="찾기", command=lambda: input_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2)

tk.Label(root, text="출력 파일:").grid(row=1, column=0)
output_entry = tk.Entry(root, width=40)
output_entry.grid(row=1, column=1)
tk.Button(root, text="찾기", command=lambda: output_entry.insert(0, filedialog.asksaveasfilename(defaultextension=".xlsx"))).grid(row=1, column=2)

tk.Label(root, text="임베더:").grid(row=2, column=0)
embedder_var = tk.StringVar(value="bge")
embedder_menu = tk.OptionMenu(root, embedder_var, "bge", "st", "openai")
embedder_menu.grid(row=2, column=1)

tk.Label(root, text="유사도 임계값:").grid(row=3, column=0)
threshold_var = tk.DoubleVar(value=0.3)
tk.Entry(root, textvariable=threshold_var).grid(row=3, column=1)

tk.Label(root, text="최대 문장 길이:").grid(row=4, column=0)
max_length_var = tk.IntVar(value=150)
tk.Entry(root, textvariable=max_length_var).grid(row=4, column=1)

tk.Label(root, text="분할기:").grid(row=5, column=0)
splitter_var = tk.StringVar(value="spacy")
tk.OptionMenu(root, splitter_var, "spacy", "stanza").grid(row=5, column=1)

tk.Label(root, text="디바이스:").grid(row=6, column=0)
device_var = tk.StringVar(value="cuda")
tk.OptionMenu(root, device_var, "cuda", "cpu").grid(row=6, column=1)

# OpenAI 옵션 (초기에는 숨김)
openai_model_label = tk.Label(root, text="OpenAI 모델명:")
openai_model_var = tk.StringVar(value="text-embedding-3-large")
openai_model_menu = tk.OptionMenu(
    root, openai_model_var,
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
)
openai_api_key_label = tk.Label(root, text="OpenAI API 키:")
openai_api_key_var = tk.StringVar()
openai_api_key_entry = tk.Entry(root, textvariable=openai_api_key_var, show="*")

tk.Button(root, text="실행", command=start_pa).grid(row=10, column=1, pady=10)

# 임베더 선택 시 OpenAI 옵션 표시/숨김
embedder_var.trace_add("write", toggle_openai_options)
toggle_openai_options()

root.mainloop()