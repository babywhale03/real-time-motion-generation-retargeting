# prompt_queue를 통해 prompt를 실시간으로 입력하기 위한 파일
# clear 입력시 prompt queue를 비우고, exit 입력시 종료

from pathlib import Path

prompt_path = "/home/bong/CLoSD/closd/custom_t2m/prompt_queue.txt"

print("### RoboGo: Real-time prompt writer (txt mode) started ###")
print("Type a prompt and press Enter. Type 'exit' to quit, or 'clear' to empty the queue.\n")

while True:
    user_input = input("[Prompt] > ").strip()
    
    if user_input.lower() == "exit":
        print("Exiting prompt writer.")
        break
    elif user_input.lower() == "clear":
        with open(prompt_path, "w") as f:
            pass  # simply truncate the file
        print("prompt_queue.txt has been cleared.")
    elif user_input:
        with open(prompt_path, "a") as f:
            f.write(user_input + "\n")
        print(f"Appended to prompt_queue.txt: '{user_input}'")
