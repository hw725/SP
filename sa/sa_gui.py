import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# 중지 플래그
stop_flag = threading.Event()

def run_sa(input_file, output_file, tgt_tokenizer, embedder, min_tokens, max_tokens, use_semantic, parallel, verbose, openai_model, openai_api_key):
    try:
        from main import main
        import sys

        progress_var.set(0)
        progress_bar.update()

        def progress_callback(current, total):
            percent = int((current / total) * 100)
            progress_var.set(percent)
            progress_bar.update()
            # 중지 요청 시 예외 발생
            if stop_flag.is_set():
                raise KeyboardInterrupt("사용자에 의해 중지됨")

        sys.argv = [
            "main.py",
            input_file,
            output_file,
            "--tokenizer", tgt_tokenizer,
            "--embedder", embedder,
            "--min-tokens", str(min_tokens),
            "--max-tokens", str(max_tokens),
        ]
        if not use_semantic:
            sys.argv.append("--no-semantic")
        if parallel:
            sys.argv.append("--parallel")
        if verbose:
            sys.argv.append("--verbose")
        if embedder == "openai":
            sys.argv += [
                "--openai-model", openai_model,
            ]
            if openai_api_key:
                sys.argv += ["--openai-api-key", openai_api_key]
        sys.argv.append("--save-phrase")
        # main 함수에 progress_callback, stop_flag 전달 필요 (main/processor에서 지원해야 함)
        main(progress_callback=progress_callback, stop_flag=stop_flag)
        progress_var.set(100)
        progress_bar.update()
        messagebox.showinfo("완료", "SA 처리가 완료되었습니다.")
    except KeyboardInterrupt:
        messagebox.showinfo("중지", "사용자에 의해 처리가 중지되었습니다.")
    except Exception as e:
        messagebox.showerror("오류", f"실행 중 오류 발생: {e}")

def start_sa():
    stop_flag.clear()
    input_file = input_entry.get()
    output_file = output_entry.get()
    tgt_tokenizer = tgt_tokenizer_var.get()
    embedder = embedder_var.get()
    min_tokens = min_tokens_var.get()
    max_tokens = max_tokens_var.get()
    use_semantic = semantic_var.get()
    parallel = parallel_var.get()
    verbose = verbose_var.get()
    openai_model = openai_model_var.get()
    openai_api_key = openai_api_key_var.get()
    if not input_file or not output_file:
        messagebox.showerror("오류", "입력/출력 파일을 지정하세요.")
        return
    progress_var.set(0)
    progress_bar.update()
    threading.Thread(target=run_sa, args=(
        input_file, output_file, tgt_tokenizer, embedder, min_tokens, max_tokens, use_semantic, parallel, verbose, openai_model, openai_api_key
    )).start()

def stop_sa():
    stop_flag.set()

def toggle_openai_options(*args):
    if embedder_var.get() == "openai":
        openai_model_label.grid(row=10, column=0)
        openai_model_menu.grid(row=10, column=1)
        openai_api_key_label.grid(row=11, column=0)
        openai_api_key_entry.grid(row=11, column=1)
    else:
        openai_model_label.grid_remove()
        openai_model_menu.grid_remove()
        openai_api_key_label.grid_remove()
        openai_api_key_entry.grid_remove()

root = tk.Tk()
root.title("SA Sentence Aligner GUI")

tk.Label(root, text="입력 파일:").grid(row=0, column=0)
input_entry = tk.Entry(root, width=40)
input_entry.grid(row=0, column=1)
tk.Button(root, text="찾기", command=lambda: input_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2)

tk.Label(root, text="출력 파일:").grid(row=1, column=0)
output_entry = tk.Entry(root, width=40)
output_entry.grid(row=1, column=1)
tk.Button(root, text="찾기", command=lambda: output_entry.insert(0, filedialog.asksaveasfilename(defaultextension=".xlsx"))).grid(row=1, column=2)

tk.Label(root, text="원문 토크나이저:").grid(row=2, column=0)
tk.Label(root, text="jieba (고정)").grid(row=2, column=1)

tk.Label(root, text="번역문 토크나이저:").grid(row=3, column=0)
tgt_tokenizer_var = tk.StringVar(value="mecab")
tk.OptionMenu(root, tgt_tokenizer_var, "mecab").grid(row=3, column=1)

tk.Label(root, text="임베더:").grid(row=4, column=0)
embedder_var = tk.StringVar(value="bge")
tk.OptionMenu(root, embedder_var, "bge", "openai").grid(row=4, column=1)

tk.Label(root, text="최소 토큰 수:").grid(row=5, column=0)
min_tokens_var = tk.IntVar(value=1)
tk.Entry(root, textvariable=min_tokens_var).grid(row=5, column=1)

tk.Label(root, text="최대 토큰 수:").grid(row=6, column=0)
max_tokens_var = tk.IntVar(value=10)
tk.Entry(root, textvariable=max_tokens_var).grid(row=6, column=1)

semantic_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="의미 기반 매칭 사용", variable=semantic_var).grid(row=7, column=1, sticky="w")

parallel_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="병렬 처리", variable=parallel_var).grid(row=8, column=1, sticky="w")

verbose_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="상세 로그(Verbose)", variable=verbose_var).grid(row=9, column=1, sticky="w")

# 빈 줄
tk.Label(root, text="").grid(row=10, column=0)

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

tk.Button(root, text="실행", command=start_sa).grid(row=12, column=1, pady=10)
tk.Button(root, text="중지", command=stop_sa).grid(row=12, column=2, pady=10)

# 진행률 바 추가
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=300)
progress_bar.grid(row=13, column=0, columnspan=3, pady=10)

# 임베더 선택 시 OpenAI 옵션 표시/숨김
embedder_var.trace_add("write", toggle_openai_options)
toggle_openai_options()

root.mainloop()